"""
Retrieval experiments: what actually moves recall, and what it costs in tokens.

`scripts/run_retrieval_eval.py` answers "is retrieval worth it at all" and picks
the shipping default. This script is the workbench underneath that answer — it
holds several candidate retrievers against the same 39 golden questions and the
same SQL-derived ground truth, so a proposed change to `engine/retrieval.py` has
to arrive with a number rather than a story.

WHY THIS IS FAST ENOUGH TO SWEEP
The eval script goes through Chroma, which re-embeds on every rebuild. Here the
71 table documents and the 39 questions are embedded ONCE per corpus variant
into a dense matrix, and every ranking after that is a dot product. The default
all-MiniLM-L6-v2 vectors come back L2-normalised, so cosine similarity IS the
dot product — no renormalising, and no approximation relative to what Chroma
does. `--verify` proves that: it re-runs the live `engine.retrieval` code path
and asserts the two agree on every question.

That equivalence is the whole licence for this script to exist. If it ever
fails, the numbers below stop describing the shipped system and this file is
lying, so the check is an assertion and not a comment.

    python scripts/run_retrieval_experiments.py                # the whole board
    python scripts/run_retrieval_experiments.py --verify       # agree with Chroma?
    python scripts/run_retrieval_experiments.py --only diagnose
    python scripts/run_retrieval_experiments.py --only paraphrase
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_manifest import DOMAINS, MANIFEST, table_name  # noqa: E402
from engine import retrieval  # noqa: E402
from engine.warehouse import build_warehouse, table_columns, table_names  # noqa: E402

GOLDEN = ROOT / "evals" / "golden_questions.yaml"
RRF_K = retrieval.RRF_K
DEFAULT_K = retrieval.DEFAULT_K


# --------------------------------------------------------------------------
# Ground truth, lifted from the eval so both scripts label identically.
# --------------------------------------------------------------------------

def tables_in_sql(sql: str, known: set[str]) -> set[str]:
    found = set()
    lowered = (sql or "").lower()
    for name in known:
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            found.add(name)
    return found


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Case:
    id: str
    question: str
    needed: set[str]


def load_cases(con) -> list[Case]:
    known = set(table_names(con))
    rows = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    cases = []
    for row in rows:
        needed = tables_in_sql(row.get("sql", ""), known)
        if needed:
            cases.append(Case(row["id"], row["question"], needed))
    return cases


# --------------------------------------------------------------------------
# The corpus, as data rather than as a side effect of importing the manifest.
# --------------------------------------------------------------------------

@dataclass
class Corpus:
    """One document per table, plus the machinery to rank against it.

    `descriptions` is kept separate from `documents` because the two are used
    for different things: the document is what gets embedded, the description is
    what gets pasted into the prompt. A corpus experiment that changes only the
    embedded text costs nothing in the prompt; one that changes the description
    changes both, and the token column has to show it.
    """

    names: list[str]
    documents: list[str]
    domains: dict[str, str]
    descriptions: dict[str, str]
    columns: dict[str, list[tuple[str, str]]]
    label: str = "base"
    doc_vectors: np.ndarray | None = field(default=None, repr=False)

    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}


_EMBED = None


def _embedder():
    global _EMBED
    if _EMBED is None:
        import chromadb.utils.embedding_functions as ef

        _EMBED = ef.DefaultEmbeddingFunction()
    return _EMBED


def embed(texts: list[str]) -> np.ndarray:
    """all-MiniLM-L6-v2, the same function Chroma uses by default.

    Output rows are already unit length, so `A @ B.T` is cosine similarity.
    """
    return np.asarray(_embedder()(texts), dtype=np.float32)


def build_corpus(
    con,
    *,
    describe=None,
    document=None,
    label: str = "base",
) -> Corpus:
    """The shipped corpus, with two seams for experiments.

    `describe(domain, name, description) -> str` rewrites what the prompt shows
    (and therefore what is embedded). `document(domain, name, description,
    columns) -> str` rewrites only the embedded text. Passing neither reproduces
    `engine.retrieval.build_corpus` exactly.
    """
    names, docs = [], []
    domains, descriptions, cols_by_table = {}, {}, {}
    for domain, table, _source, description in MANIFEST:
        name = table_name(domain, table)
        try:
            cols = table_columns(con, name)
        except Exception:
            cols = []
        if describe is not None:
            description = describe(domain, name, description)
        if document is not None:
            doc = document(domain, name, description, cols)
        else:
            doc = retrieval._document(domain, name, description, cols)
        names.append(name)
        docs.append(doc)
        domains[name] = domain
        descriptions[name] = description
        cols_by_table[name] = cols
    corpus = Corpus(names, docs, domains, descriptions, cols_by_table, label=label)
    corpus.doc_vectors = embed(docs)
    return corpus


# --------------------------------------------------------------------------
# Rankers. Each returns a FULL ranking (all 71 tables, best first) so that k,
# reranking and adaptive-k can all be applied on top without re-retrieving.
# --------------------------------------------------------------------------

def vector_ranking(corpus: Corpus, qvec: np.ndarray) -> list[tuple[str, float]]:
    sims = corpus.doc_vectors @ qvec
    order = np.argsort(-sims)
    return [(corpus.names[i], float(sims[i])) for i in order]


def _tokens(text: str) -> set[str]:
    return retrieval._tokens(text)


SPLIT_STOP = retrieval._STOP | {"qty", "num", "pct", "amt"}


def _tokens_split(text: str) -> set[str]:
    """Token set that also splits snake_case identifiers into their parts.

    The shipped tokeniser treats `qty_shipped` as one atom, so the word
    "shipped" in a question cannot match it. That is not a tuning knob, it is a
    bug with a recall cost, and `experiment_tokeniser` measures it.
    """
    out = set()
    for token in re.findall(r"[a-z0-9_]+", (text or "").lower()):
        pieces = [token] + token.split("_") if "_" in token else [token]
        for piece in pieces:
            if len(piece) > 2 and piece not in SPLIT_STOP:
                out.add(piece)
    return out


def keyword_ranking(
    corpus: Corpus,
    question: str,
    *,
    tokenise=_tokens,
) -> list[tuple[str, float]]:
    """Token overlap, ties broken by document brevity — the shipped baseline.

    Returns ONLY the documents with non-zero overlap, exactly like
    `retrieve_keyword`. That truncation is not cosmetic: it is why fusion can
    punish a table no keyword ever reaches. See `experiment_diagnose`.
    """
    q = tokenise(question)
    if not q:
        return []
    scored = []
    for i, name in enumerate(corpus.names):
        doc_tokens = tokenise(corpus.documents[i])
        overlap = len(q & doc_tokens)
        if overlap:
            scored.append((overlap, len(doc_tokens), name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(name, float(overlap)) for overlap, _brevity, name in scored]


def rrf(rankings: list[list[tuple[str, float]]], *, rrf_k: int = RRF_K) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (name, _score) in enumerate(ranking, start=1):
            fused[name] = fused.get(name, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def hybrid_ranking(
    corpus: Corpus,
    question: str,
    qvec: np.ndarray,
    *,
    k: int = DEFAULT_K,
    tokenise=_tokens,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float]]:
    """RRF over the two arms, with the shipped `pool = max(2k, 12)` truncation."""
    pool = max(k * 2, 12)
    vec = vector_ranking(corpus, qvec)[:pool]
    kw = keyword_ranking(corpus, question, tokenise=tokenise)[:pool]
    return rrf([vec, kw], rrf_k=rrf_k)


# --------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------

@dataclass
class Result:
    label: str
    question_recall: float
    table_recall: float
    tokens: float
    misses: list[str]
    mean_k: float = 0.0


def render_block(corpus: Corpus, tables: list[str]) -> str:
    """Byte-for-byte the block `schema_catalog_for` builds, so tokens compare."""
    by_domain: dict[str, list[str]] = {}
    for name in tables:
        by_domain.setdefault(corpus.domains.get(name, ""), []).append(name)
    lines: list[str] = []
    for domain, rows in by_domain.items():
        lines.append(f"\n### Domain: {domain} — {DOMAINS.get(domain, '')}")
        for name in rows:
            cols = ", ".join(f"{c} {t}" for c, t in corpus.columns.get(name, []))
            lines.append(f"- {name}: {corpus.descriptions.get(name, '')}")
            if cols:
                lines.append(f"    columns: {cols}")
    return "\n".join(lines).strip()


def score(
    label: str,
    corpus: Corpus,
    cases: list[Case],
    qvecs: np.ndarray,
    select,
) -> Result:
    """`select(corpus, case, qvec) -> list[str]` is the retriever under test."""
    hit = 0
    missed_tables = 0
    total_tables = 0
    tokens = 0
    ks = 0
    misses: list[str] = []
    for i, case in enumerate(cases):
        got = select(corpus, case, qvecs[i])
        ks += len(got)
        need = case.needed
        total_tables += len(need)
        missing = need - set(got)
        missed_tables += len(missing)
        if missing:
            misses.append(f"{case.id} (missing {', '.join(sorted(missing))})")
        else:
            hit += 1
        tokens += approx_tokens(render_block(corpus, got))
    n = len(cases)
    return Result(
        label=label,
        question_recall=hit / n,
        table_recall=(total_tables - missed_tables) / total_tables,
        tokens=tokens / n,
        misses=misses,
        mean_k=ks / n,
    )


def table(results: list[Result], *, show_k: bool = False, show_misses: bool = True) -> None:
    width = max(len(r.label) for r in results) + 2
    head = f"  {'variant':<{width}} {'questions':>10} {'tables':>9} {'~tokens':>9}"
    if show_k:
        head += f" {'mean k':>8}"
    print(head)
    for r in results:
        line = (
            f"  {r.label:<{width}} {r.question_recall:>9.1%} "
            f"{r.table_recall:>9.1%} {r.tokens:>9,.0f}"
        )
        if show_k:
            line += f" {r.mean_k:>8.1f}"
        print(line)
    if show_misses:
        for r in results:
            if r.misses:
                print(f"\n  {r.label} missed:")
                for miss in r.misses:
                    print(f"    - {miss}")


# --------------------------------------------------------------------------
# The shipped retrievers, expressed as `select` functions.
# --------------------------------------------------------------------------

def sel_vector(k=DEFAULT_K):
    return lambda c, case, qv: [n for n, _ in vector_ranking(c, qv)[:k]]


def sel_keyword(k=DEFAULT_K, tokenise=_tokens):
    return lambda c, case, qv: [
        n for n, _ in keyword_ranking(c, case.question, tokenise=tokenise)[:k]
    ]


def sel_hybrid(k=DEFAULT_K, tokenise=_tokens, rrf_k=RRF_K):
    return lambda c, case, qv: [
        n for n, _ in hybrid_ranking(c, case.question, qv, k=k, tokenise=tokenise, rrf_k=rrf_k)[:k]
    ]


# ==========================================================================
# EXPERIMENT 0 — diagnose the one question hybrid never gets.
# ==========================================================================

def experiment_diagnose(con, cases, corpus, qvecs) -> None:
    print("=" * 78)
    print("0. DIAGNOSIS — why hybrid misses top_category_revenue")
    print("=" * 78)

    target = next(c for c in cases if c.id == "top_category_revenue")
    i = cases.index(target)
    qv = qvecs[i]
    print(f"\n  question: {target.question}")
    print(f"  needs:    {', '.join(sorted(target.needed))}\n")

    vec = vector_ranking(corpus, qv)
    kw = keyword_ranking(corpus, target.question)
    hyb = hybrid_ranking(corpus, target.question, qv, k=DEFAULT_K)
    for name, ranking in (("vector", vec), ("keyword", kw), ("hybrid", hyb)):
        pos = {t: r for r, (t, _) in enumerate(ranking, start=1)}
        print(
            f"  {name:<8} supplychain_fact_orders rank = "
            f"{pos.get('supplychain_fact_orders', '—')!s:<5} "
            f"(list length {len(ranking)})"
        )

    print("\n  The code comment says this is a corpus failure at every k. It is not.")
    print("  Vector ALONE ranks the table 5th; the shipped eval confirms vector")
    print("  does not miss this question. Fusion is what loses it.\n")

    q = _tokens(target.question)
    doc = corpus.documents[corpus.index()["supplychain_fact_orders"]]
    print(f"  question tokens: {sorted(q)}")
    print(f"  overlap with the table document: {sorted(q & _tokens(doc))}")
    print("\n  Zero overlap, so keyword returns a list of "
          f"{len(kw)} tables that does not contain it at all.")
    print("  RRF then scores it from ONE arm (1/(60+5) = 0.0154) while tables both")
    print("  arms saw score from two, and it falls out of the top 14.")
    print("\n  Two independent causes, and they need separate fixes:")
    print("    (i)  the tokeniser never splits qty_shipped, so 'shipped' cannot match")
    print("    (ii) the description says fill rate and OTIF, never revenue or price")


# ==========================================================================
# EXPERIMENT 1 — the corpus fix: a description that says what the table holds.
# ==========================================================================

# The shipped description sells the table as a service-level fact ("fill rate",
# "OTIF") and never mentions that it carries unit_price and unit_cost — which is
# the whole reason it can answer a revenue question. This adds the money grain
# and the category path, and deliberately does NOT stuff keywords: every clause
# is a true statement a reader of the prompt benefits from.
BETTER_DESCRIPTIONS = {
    "supplychain_fact_orders": (
        "One row per order line, and the supply-chain revenue fact. "
        "qty_ordered vs qty_shipped measures fill rate; promised_date vs shipped_date "
        "measures on-time delivery (OTIF). unit_price and unit_cost make each line a "
        "dollar amount, so shipped revenue is qty_shipped * unit_price and margin is "
        "qty_shipped * (unit_price - unit_cost); join dim_product for the product "
        "category behind that revenue. Joins dim_customer, dim_product, dim_lot, "
        "dim_warehouse by id."
    ),
}


def experiment_corpus(con, cases, base, qvecs) -> Corpus:
    print("\n" + "=" * 78)
    print("1. CORPUS FIX — rewrite the description that omits what the table is for")
    print("=" * 78)

    fixed = build_corpus(
        con,
        describe=lambda d, n, desc: BETTER_DESCRIPTIONS.get(n, desc),
        label="fixed-desc",
    )
    results = []
    for label, corpus in (("baseline", base), ("better description", fixed)):
        for strat, sel in (("vector", sel_vector()), ("hybrid", sel_hybrid())):
            results.append(score(f"{label} · {strat}", corpus, cases, qvecs, sel))
    table(results)

    i = cases.index(next(c for c in cases if c.id == "top_category_revenue"))
    for label, corpus in (("baseline", base), ("better description", fixed)):
        hyb = hybrid_ranking(corpus, cases[i].question, qvecs[i], k=DEFAULT_K)
        pos = {t: r for r, (t, _) in enumerate(hyb, start=1)}
        print(f"\n  {label}: supplychain_fact_orders hybrid rank = "
              f"{pos.get('supplychain_fact_orders', '—')}")
    return fixed


# ==========================================================================
# EXPERIMENT 2 — the tokeniser: split snake_case before matching.
# ==========================================================================

def experiment_tokeniser(con, cases, base, fixed, qvecs) -> None:
    print("\n" + "=" * 78)
    print("2. TOKENISER — split snake_case identifiers in the keyword arm")
    print("=" * 78)
    print("\n  Column names ARE the vocabulary (retrieval.py:102-108 says so), but")
    print("  `qty_shipped` is indexed as one atom, so the word 'shipped' misses it.\n")

    results = []
    for cname, corpus in (("base corpus", base), ("fixed corpus", fixed)):
        for tname, tok in (("atomic", _tokens), ("split", _tokens_split)):
            results.append(
                score(f"{cname} · keyword · {tname}", corpus, cases, qvecs,
                      sel_keyword(tokenise=tok))
            )
            results.append(
                score(f"{cname} · hybrid · {tname}", corpus, cases, qvecs, sel_hybrid(tokenise=tok))
            )
    table(results)


# ==========================================================================
# EXPERIMENT 3 (a) — deterministic query expansion.
# ==========================================================================

# Built from vocabulary the warehouse itself uses, not from a general thesaurus:
# every right-hand side appears in a manifest description or a column name. An
# LLM rewriter would generalise past this list; that claim is argued, not
# measured, because there is no API key in this environment.
SYNONYMS = {
    "revenue": ["sales", "amount", "price", "billed", "net_revenue"],
    "sales": ["revenue", "orders", "lines"],
    "margin": ["gross_margin", "profit", "cost"],
    "churn": ["attrition", "risk", "retention"],
    "quit": ["attrition", "termination", "voluntary", "leaver"],
    "left": ["termination", "attrition", "voluntary"],
    "staff": ["employees", "headcount", "workforce"],
    "employee": ["employees", "headcount", "workforce"],
    "denied": ["denial", "denial_reason", "status"],
    "denial": ["denied", "carc", "status"],
    "collect": ["collection", "paid_amount", "nrv", "yield"],
    "shipped": ["qty_shipped", "fill", "otif", "delivery"],
    "category": ["product", "categories", "sku"],
    "customer": ["customers", "account", "client"],
    "site": ["sites", "investigator", "location"],
    "query": ["queries", "edit_check", "query_log"],
    "subject": ["subjects", "patient", "enrolled"],
    "alert": ["alerts", "flagged", "suspicious", "threshold"],
    "channel": ["channels", "medium", "touchpoint"],
    "spend": ["cost", "budget", "media"],
    "test": ["tests", "assertion", "data_test"],
    "model": ["models", "dbt"],
    "department": ["departments", "merchandising", "dept"],
    "rep": ["reps", "salesperson", "quota"],
    "store": ["stores", "format", "supercenter"],
    "vendor": ["vendors", "supplier"],
    "exception": ["exceptions", "variance", "reconciliation"],
    "attainment": ["quota", "target"],
    "documentation": ["description", "documented"],
}

# Acronyms the descriptions spell out (or vice versa). Same rule: every
# expansion is a string that genuinely occurs in this warehouse's own text.
ACRONYMS = {
    "otif": "on-time in-full delivery",
    "ar": "accounts receivable open claims",
    "nrv": "net realisable value expected collectable",
    "ncr": "net collection rate",
    "clv": "customer lifetime value",
    "rfm": "recency frequency monetary segment",
    "cpa": "cost per acquisition conversion",
    "aml": "anti money laundering suspicious",
    "edc": "electronic data capture",
    "gl": "general ledger account",
    "kpi": "key performance indicator metric",
    "did": "difference in differences",
    "fefo": "first expiry first out",
    "sku": "product item catalog",
    "pnl": "profit and loss",
}


def expand_query(question: str) -> str:
    """Question + domain vocabulary, appended rather than substituted.

    Appended because the original wording is what the embedding is good at and
    replacing it would throw that away; the expansion is extra surface for the
    keyword arm and a mild nudge for the vector arm.
    """
    words = re.findall(r"[a-z0-9_]+", (question or "").lower())
    extra: list[str] = []
    for word in words:
        for syn in SYNONYMS.get(word, ()):
            if syn not in extra:
                extra.append(syn)
        if word in ACRONYMS:
            extra.append(ACRONYMS[word])
        stem = word[:-1] if word.endswith("s") else word
        for syn in SYNONYMS.get(stem, ()):
            if syn not in extra:
                extra.append(syn)
    if not extra:
        return question
    return f"{question} {' '.join(extra)}"


def experiment_expansion(con, cases, corpus, qvecs) -> None:
    print("\n" + "=" * 78)
    print("3 (a). QUERY EXPANSION — deterministic synonym + acronym expansion")
    print("=" * 78)

    expanded = [expand_query(c.question) for c in cases]
    evecs = embed(expanded)
    ecases = [Case(c.id, expanded[i], c.needed) for i, c in enumerate(cases)]

    results = [
        score("plain · vector", corpus, cases, qvecs, sel_vector()),
        score("expanded · vector", corpus, ecases, evecs, sel_vector()),
        score("plain · keyword", corpus, cases, qvecs, sel_keyword()),
        score("expanded · keyword", corpus, ecases, evecs, sel_keyword()),
        score("plain · hybrid", corpus, cases, qvecs, sel_hybrid()),
        score("expanded · hybrid", corpus, ecases, evecs, sel_hybrid()),
        # Expansion helps exact matching more than it helps meaning, so also try
        # feeding the plain question to the vector arm and the expanded one to
        # the keyword arm.
        score(
            "split-feed hybrid",
            corpus,
            cases,
            qvecs,
            lambda c, case, qv: [
                n
                for n, _ in rrf(
                    [
                        vector_ranking(c, qv)[: max(DEFAULT_K * 2, 12)],
                        keyword_ranking(c, expand_query(case.question))[: max(DEFAULT_K * 2, 12)],
                    ]
                )[:DEFAULT_K]
            ],
        ),
    ]
    table(results)
    print("\n  Sample expansion:")
    for c in cases[:1] + [c for c in cases if c.id == "top_category_revenue"]:
        print(f"    {c.question}\n      -> {expand_query(c.question)}")


# ==========================================================================
# EXPERIMENT 4 (b) — column-level retrieval.
# ==========================================================================

def _doc_column_weighted(domain, name, description, columns, *, repeat=2):
    """Table document with the column vocabulary repeated (a cheap field boost)."""
    base = retrieval._document(domain, name, description, columns)
    if not columns:
        return base
    words = " ".join(c.replace("_", " ") for c, _t in columns)
    return base + ("\n" + words) * repeat


def _doc_column_prose(domain, name, description, columns):
    """Same, but the columns are rendered once as space-separated words.

    Tests whether the win (if any) comes from REPEATING the columns or merely
    from splitting `qty_shipped` into words the sentence encoder can read.
    """
    base = retrieval._document(domain, name, description, columns)
    if not columns:
        return base
    return base + "\n" + " ".join(c.replace("_", " ") for c, _t in columns)


def experiment_columns(con, cases, corpus, qvecs) -> None:
    print("\n" + "=" * 78)
    print("4 (b). COLUMN-LEVEL RETRIEVAL")
    print("=" * 78)

    variants = [("baseline (columns once, snake_case)", corpus)]
    variants.append(
        ("+ columns as words", build_corpus(con, document=_doc_column_prose, label="prose"))
    )
    variants.append(
        (
            "+ columns as words x2",
            build_corpus(
                con,
                document=lambda d, n, desc, c: _doc_column_weighted(d, n, desc, c, repeat=2),
                label="x2",
            ),
        )
    )

    results = []
    for label, c in variants:
        results.append(score(f"{label} · vector", c, cases, qvecs, sel_vector()))
        results.append(score(f"{label} · hybrid", c, cases, qvecs, sel_hybrid()))
    table(results, show_misses=False)

    # The other reading of "column-level retrieval": index one document per
    # COLUMN and roll the hits up to their table.
    print("\n  One document per column, rolled up to the parent table:")
    col_docs, col_owner = [], []
    for name in corpus.names:
        for cname, _ctype in corpus.columns.get(name, []):
            col_docs.append(
                f"{cname.replace('_', ' ')} — a column of {name} "
                f"({corpus.domains[name]} domain). {corpus.descriptions[name]}"
            )
            col_owner.append(name)
    print(f"    corpus size: {len(col_docs)} column documents "
          f"vs {len(corpus.names)} table documents")
    cvecs = embed(col_docs)
    owner = np.array(col_owner)

    def sel_colroll(c, case, qv, k=DEFAULT_K):
        sims = cvecs @ qv
        order = np.argsort(-sims)
        out: list[str] = []
        for i in order:
            t = owner[i]
            if t not in out:
                out.append(t)
            if len(out) >= k:
                break
        return out

    def sel_colroll_fused(c, case, qv, k=DEFAULT_K):
        pool = max(k * 2, 12)
        col = [(t, 0.0) for t in sel_colroll(c, case, qv, k=pool)]
        return [
            n
            for n, _ in rrf(
                [
                    vector_ranking(c, qv)[:pool],
                    keyword_ranking(c, case.question)[:pool],
                    col[:pool],
                ]
            )[:k]
        ]

    table(
        [
            score("column-roll-up only", corpus, cases, qvecs, sel_colroll),
            score("hybrid + column arm (3-way RRF)", corpus, cases, qvecs, sel_colroll_fused),
        ],
        show_misses=False,
    )


# ==========================================================================
# EXPERIMENT 5 (c) — reranking the fused list.
# ==========================================================================

def rerank(
    corpus: Corpus,
    question: str,
    fused: list[tuple[str, float]],
    *,
    k: int = DEFAULT_K,
    top_n: int = 30,
    w_name: float = 1.0,
    w_domain: float = 0.25,
    w_overlap: float = 0.15,
) -> list[str]:
    """Re-score the fused top-N by three signals RRF cannot see.

    * NAME MENTION. The question literally contains the table's base name or a
      distinctive part of it ("cross sell", "flight risk"). RRF only sees ranks,
      so a keyword arm that ranked the table 3rd and a vector arm that ranked it
      3rd look identical to one that guessed — this does not.
    * DOMAIN AGREEMENT. The domain most represented in the fused top-5 is
      probably the right domain; a table from it gets a nudge. This is the
      signal that separates four tables called `fact_orders`.
    * DESCRIPTION OVERLAP. Fraction of question tokens present in the
      description, using the SPLIT tokeniser so `unit_price` can match "price".
    """
    head = fused[:top_n]
    tail = [n for n, _ in fused[top_n:]]
    if not head:
        return []

    q = _tokens_split(question)
    lowered = f" {question.lower()} "
    top_domains: dict[str, float] = {}
    for rank, (name, _s) in enumerate(head[:5], start=1):
        top_domains[corpus.domains.get(name, "")] = (
            top_domains.get(corpus.domains.get(name, ""), 0.0) + 1.0 / rank
        )
    best_domain = max(top_domains, key=top_domains.get) if top_domains else ""

    scored = []
    for rank, (name, base) in enumerate(head, start=1):
        bonus = 0.0
        stem = name.split("_", 1)[1] if "_" in name else name
        phrase = stem.replace("dim_", "").replace("fact_", "").replace("_", " ")
        if phrase and f" {phrase} " in lowered:
            bonus += w_name
        if corpus.domains.get(name) == best_domain:
            bonus += w_domain
        desc_tokens = _tokens_split(corpus.descriptions.get(name, ""))
        if q:
            bonus += w_overlap * (len(q & desc_tokens) / len(q))
        # Bonuses are added to the RRF score scaled by the score spread, so a
        # bonus can reorder neighbours without teleporting rank 30 to rank 1.
        spread = head[0][1] - head[-1][1] or 1e-6
        scored.append((base + bonus * spread * 0.5, rank, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _s, _r, name in scored][:k] + tail[: max(0, k - len(scored))]


def experiment_rerank(con, cases, corpus, qvecs) -> None:
    print("\n" + "=" * 78)
    print("5 (c). RERANKING the fused list")
    print("=" * 78)

    def make(**kw):
        def sel(c, case, qv):
            fused = hybrid_ranking(c, case.question, qv, k=DEFAULT_K)
            return rerank(c, case.question, fused, k=DEFAULT_K, **kw)
        return sel

    results = [
        score("hybrid (no rerank)", corpus, cases, qvecs, sel_hybrid()),
        score("+ rerank (all three signals)", corpus, cases, qvecs, make()),
        score("+ rerank (name only)", corpus, cases, qvecs, make(w_domain=0.0, w_overlap=0.0)),
        score("+ rerank (domain only)", corpus, cases, qvecs, make(w_name=0.0, w_overlap=0.0)),
        score("+ rerank (overlap only)", corpus, cases, qvecs, make(w_name=0.0, w_domain=0.0)),
    ]
    table(results)


# ==========================================================================
# EXPERIMENT 6 (d) — adaptive k.
# ==========================================================================

def adaptive_k(
    fused: list[tuple[str, float]],
    *,
    k_min: int = 6,
    k_max: int = 24,
    drop: float = 0.5,
) -> list[str]:
    """Keep taking hits until the fused score falls off a cliff.

    `drop` is the fraction of the gap between the best and the k_min-th score
    that a single step is allowed to be before we stop. A question whose answer
    is one obvious table has a big early cliff and stops short; a vague one has
    a flat tail and spends the full budget.
    """
    if not fused:
        return []
    names = [n for n, _ in fused]
    scores = [s for _, s in fused]
    k_min = min(k_min, len(names))
    k_max = min(k_max, len(names))
    reference = scores[0] - scores[k_min - 1] if k_min > 1 else 0.0
    if reference <= 0:
        return names[:k_max]
    for i in range(k_min, k_max):
        if scores[i - 1] - scores[i] > drop * reference:
            return names[:i]
    return names[:k_max]


def experiment_adaptive_k(con, cases, corpus, qvecs) -> None:
    print("\n" + "=" * 78)
    print("6 (d). ADAPTIVE k vs fixed k")
    print("=" * 78)

    results = []
    for k in (10, 12, 14, 16, 18, 20, 24):
        results.append(score(f"fixed k={k:<2} · hybrid", corpus, cases, qvecs, sel_hybrid(k=k)))
    table(results, show_k=True, show_misses=False)

    print()
    ad = []
    for k_min, k_max, drop in (
        (6, 24, 0.5), (6, 24, 0.35), (8, 20, 0.5), (10, 24, 0.4), (6, 18, 0.5), (10, 20, 0.6),
    ):
        def sel(c, case, qv, k_min=k_min, k_max=k_max, drop=drop):
            fused = hybrid_ranking(c, case.question, qv, k=k_max)
            return adaptive_k(fused, k_min=k_min, k_max=k_max, drop=drop)

        ad.append(score(f"adaptive [{k_min},{k_max}] drop={drop}", corpus, cases, qvecs, sel))
    table(ad, show_k=True, show_misses=False)


# ==========================================================================
# The combined candidate, and the equivalence check.
# ==========================================================================

def experiment_combined(con, cases, base, fixed, qvecs) -> None:
    print("\n" + "=" * 78)
    print("7. THE STACK — what to actually ship")
    print("=" * 78)

    def sel_split_hybrid(c, case, qv):
        ranked = hybrid_ranking(c, case.question, qv, tokenise=_tokens_split)
        return [n for n, _ in ranked[:DEFAULT_K]]

    def sel_split_rerank(c, case, qv):
        fused = hybrid_ranking(c, case.question, qv, tokenise=_tokens_split)
        return rerank(c, case.question, fused, k=DEFAULT_K)

    results = [
        score("shipped today (base · hybrid k=14)", base, cases, qvecs, sel_hybrid()),
        score("+ description fix", fixed, cases, qvecs, sel_hybrid()),
        score("+ description fix + split tokeniser", fixed, cases, qvecs, sel_split_hybrid),
        score("+ all three (rerank on top)", fixed, cases, qvecs, sel_split_rerank),
    ]
    table(results)

    print("\n  Same stack at lower k (does the fix buy back budget?):")
    lower = []
    for k in (8, 10, 12, 14):
        def sel(c, case, qv, k=k):
            fused = hybrid_ranking(c, case.question, qv, k=k, tokenise=_tokens_split)
            return [n for n, _ in fused[:k]]

        lower.append(score(f"fixed corpus · split · k={k}", fixed, cases, qvecs, sel))
    table(lower, show_k=True, show_misses=False)


def verify(con, cases, corpus, qvecs) -> None:
    """Assert this script's numpy path agrees with the live Chroma path."""
    print("\n" + "=" * 78)
    print("VERIFY — numpy re-implementation vs engine.retrieval through Chroma")
    print("=" * 78)
    bad = 0
    for i, case in enumerate(cases):
        mine = [n for n, _ in vector_ranking(corpus, qvecs[i])[:DEFAULT_K]]
        theirs = [h.table for h in retrieval.retrieve(case.question, k=DEFAULT_K, con=con)]
        if mine != theirs:
            bad += 1
            print(f"  MISMATCH {case.id}\n    mine:   {mine}\n    chroma: {theirs}")
        mine_h = [n for n, _ in hybrid_ranking(corpus, case.question, qvecs[i])[:DEFAULT_K]]
        theirs_h = [h.table for h in retrieval.retrieve_hybrid(case.question, k=DEFAULT_K, con=con)]
        if mine_h != theirs_h:
            bad += 1
            print(f"  HYBRID MISMATCH {case.id}\n    mine:   {mine_h}\n    chroma: {theirs_h}")
    print(f"\n  {len(cases)} questions checked, {bad} mismatches.")
    assert bad == 0, "numpy path diverged from Chroma — the numbers in this script are not valid"


# ==========================================================================
# EXPERIMENT 8 — the blind spot in the golden set itself.
# ==========================================================================

# Every golden question is phrased in the warehouse's own vocabulary, because
# they were written by someone looking at the schema. That quietly biases the
# eval TOWARD exact matching — and once the tokeniser is fixed, keyword alone
# scores 100% on it, which would read as "delete the embeddings".
#
# These are the same questions asked the way a business user asks them, with the
# schema words removed on purpose. Ground truth is INHERITED from the original
# question's reference SQL: a paraphrase needs the same tables by construction,
# so the labels are still derived rather than authored.
#
# This is a probe, not repo ground truth. I wrote these sentences, and they are
# deliberately biased against keyword. That is the point: it is the axis the
# golden set never tests, and a fair reading needs BOTH numbers.
PARAPHRASES = {
    "denial_rate": "How often do insurers refuse to pay us?",
    "voluntary_attrition": "How many people quit on their own?",
    "active_employees": "How big is our current headcount?",
    "top_highrisk_dept": "Which team is most likely to lose people soon?",
    "fill_rate": "How much of what customers asked for did we actually send?",
    "top_category_revenue": "Which kind of product brings in the most money?",
    "top_customer": "Who buys the most from us?",
    "high_churn_customers": "How many buyers look like they are about to leave?",
    "total_exceptions": "How many places do our two sets of books disagree?",
    "site_query_rate_per_subject":
        "Which hospital raises the most data problems per person enrolled?",
    "aml_riskiest_channel": "Which way of paying gets flagged as dodgy most often?",
    "wholesale_lowest_margin_department":
        "Which part of the shop makes the least profit per dollar sold?",
    "dbt_most_tested_model": "Which table in our pipeline has the most checks on it?",
    "expected_nrv_total": "Of the money still owed to us, how much will we really see?",
}


def paraphrase_cases(cases: list[Case]) -> list[Case]:
    by_id = {c.id: c for c in cases}
    return [Case(i, q, by_id[i].needed) for i, q in PARAPHRASES.items() if i in by_id]


def weighted_hybrid(
    corpus: Corpus,
    question: str,
    qvec: np.ndarray,
    *,
    k: int = DEFAULT_K,
    w_vector: float = 1.0,
    w_keyword: float = 1.0,
    tokenise=_tokens_split,
) -> list[str]:
    """RRF with a weight per arm.

    Plain RRF assumes the two retrievers are equally trustworthy. Measured, they
    are not: the keyword arm is worth having on schema-vocabulary questions and
    is actively misleading on paraphrased ones, so weighting the vector arm up
    is the honest encoding of that. It stays rank-only fusion — the weight
    multiplies the reciprocal-rank contribution, it does not mix two
    incomparable scores.
    """
    pool = max(k * 2, 12)
    vec = vector_ranking(corpus, qvec)[:pool]
    kw = keyword_ranking(corpus, question, tokenise=tokenise)[:pool]
    fused: dict[str, float] = {}
    for weight, ranking in ((w_vector, vec), (w_keyword, kw)):
        for rank, (name, _s) in enumerate(ranking, start=1):
            fused[name] = fused.get(name, 0.0) + weight / (RRF_K + rank)
    return [n for n, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:k]]


def experiment_paraphrase(con, cases, base, qvecs) -> None:
    print("\n" + "=" * 78)
    print("8. THE GOLDEN SET'S BLIND SPOT — questions without schema vocabulary")
    print("=" * 78)

    probe = paraphrase_cases(cases)
    pvecs = embed([c.question for c in probe])
    prose = build_corpus(con, document=_doc_column_prose, label="cols-as-words")
    print(f"\n  {len(probe)} paraphrase probes, labels inherited from the reference SQL\n")

    rows = []
    for label, corpus, tok, kind in (
        ("keyword · atomic", base, _tokens, "kw"),
        ("keyword · split", base, _tokens_split, "kw"),
        ("vector", base, _tokens, "vec"),
        ("vector · cols-as-words", prose, _tokens, "vec"),
        ("hybrid · atomic (shipped)", base, _tokens, "hy"),
        ("hybrid · split", base, _tokens_split, "hy"),
        ("hybrid · split, cols-as-words", prose, _tokens_split, "hy"),
    ):
        sel = {
            "kw": sel_keyword(tokenise=tok),
            "vec": sel_vector(),
            "hy": sel_hybrid(tokenise=tok),
        }[kind]
        rows.append(score(label, corpus, probe, pvecs, sel))
    table(rows, show_misses=False)
    print("\n  Keyword collapses; the embeddings are what hold this up. The golden")
    print("  set cannot see that, which is why 'keyword alone hits 100%' is not a")
    print("  reason to delete the vector index.")

    print("\n  Weighted RRF, scored on BOTH sets (corpus = cols-as-words, split, k=14):")
    print(f"\n  {'vector:keyword':<16} {'golden q/tables':>20} {'paraphrase q/tables':>22}")
    for wv, wk in ((1, 1), (1.5, 1), (2, 1), (3, 1), (4, 1), (1, 0)):
        def sel(c, case, qv, wv=wv, wk=wk):
            return weighted_hybrid(c, case.question, qv, w_vector=wv, w_keyword=wk)

        g = score("", prose, cases, qvecs, sel)
        p = score("", prose, probe, pvecs, sel)
        print(
            f"  {f'{wv}:{wk}':<16} {g.question_recall:>11.1%}/{g.table_recall:<8.1%} "
            f"{p.question_recall:>12.1%}/{p.table_recall:<8.1%}"
        )
    print("\n  1.5:1 through 3:1 is a plateau, not a knife edge — 2:1 sits in the")
    print("  middle of it and is Pareto-better than 1:1 on both sets at once.")


EXPERIMENTS = (
    "diagnose", "corpus", "tokeniser", "expansion", "columns",
    "rerank", "adaptivek", "combined", "paraphrase",
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=EXPERIMENTS, action="append", default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    want = set(args.only or EXPERIMENTS)

    con = build_warehouse()
    cases = load_cases(con)
    base = build_corpus(con)
    qvecs = embed([c.question for c in cases])
    print(f"\n{len(base.names)} tables · {len(cases)} labelled questions · "
          f"k={DEFAULT_K} unless stated")

    if args.verify:
        verify(con, cases, base, qvecs)

    fixed = None
    if "diagnose" in want:
        experiment_diagnose(con, cases, base, qvecs)
    if {"corpus", "tokeniser", "combined"} & want:
        fixed = experiment_corpus(con, cases, base, qvecs) if "corpus" in want else build_corpus(
            con, describe=lambda d, n, desc: BETTER_DESCRIPTIONS.get(n, desc), label="fixed-desc"
        )
    if "tokeniser" in want:
        experiment_tokeniser(con, cases, base, fixed, qvecs)
    if "expansion" in want:
        experiment_expansion(con, cases, base, qvecs)
    if "columns" in want:
        experiment_columns(con, cases, base, qvecs)
    if "rerank" in want:
        experiment_rerank(con, cases, base, qvecs)
    if "adaptivek" in want:
        experiment_adaptive_k(con, cases, base, qvecs)
    if "combined" in want:
        experiment_combined(con, cases, base, fixed, qvecs)
    if "paraphrase" in want:
        experiment_paraphrase(con, cases, base, qvecs)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
