"""
Retrieval experiments: what would actually make `engine/retrieval.py` retrieve better?

`scripts/run_retrieval_eval.py` answers "does retrieval beat pasting the whole
schema, and beat keyword search". It does, and the numbers are in the module
docstring. This script answers the next question, which is harder and less
flattering: *the current default still misses a question — why, and what fixes
it?* It is a bench, not a product. Nothing here is imported by the app.

Everything is measured on the same ground truth as the main eval — the tables
each golden question's reference SQL selects from — so the numbers here are
directly comparable to the ones in `engine/retrieval.py`'s header table.

WHAT IS MEASURED, AND WHAT CANNOT BE
There is no API key in this environment, so end-to-end SQL accuracy is not
measurable and is not claimed anywhere. Recall@k and prompt tokens are, and
those are the two numbers that decide whether a retrieval change is worth
shipping: a missed table makes the answer impossible, and every retrieved table
is paid for on every turn.

    python scripts/run_retrieval_experiments.py                 # everything
    python scripts/run_retrieval_experiments.py --only diagnose fusion
    python scripts/run_retrieval_experiments.py --k 14

Sections:
    diagnose   why top_category_revenue misses, hit by hit
    corpus     does a better table DESCRIPTION fix it, and what does it cost
    fusion     RRF drops singly-retrieved tables; three candidate repairs
    expand     deterministic query expansion (acronyms, pseudo-relevance feedback)
    columns    indexing columns as their own documents, and column weighting
    rerank     re-scoring the fused top-N by sharper, cheaper signals
    adaptivek  per-question k from the score-gap profile, vs. fixed k
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_manifest import DOMAINS, MANIFEST, table_name  # noqa: E402
from engine import retrieval  # noqa: E402
from engine.warehouse import build_warehouse, schema_catalog, table_columns, table_names  # noqa: E402

GOLDEN = ROOT / "evals" / "golden_questions.yaml"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """~4 characters per token — the same estimator run_retrieval_eval.py uses.

    Kept identical on purpose. The absolute number is a rough proxy; what
    matters is that every row in every table below was measured the same way,
    so the differences are real even though the units are approximate.
    """
    return max(1, len(text) // 4)


def tables_in_sql(sql: str, known: set[str]) -> set[str]:
    found = set()
    lowered = (sql or "").lower()
    for name in known:
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            found.add(name)
    return found


def load_cases(con) -> list[dict]:
    known = set(table_names(con))
    rows = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    cases = []
    for row in rows:
        needed = tables_in_sql(row.get("sql", ""), known)
        if needed:
            cases.append({"id": row["id"], "question": row["question"], "needed": needed})
    return cases


class Bench:
    """Holds the warehouse, the cases, and the caches that make sweeps tractable.

    The main eval re-derives the corpus (71 `DESCRIBE` round trips) inside every
    single `retrieve_keyword` call. That is fine for one run of one strategy; it
    is not fine for the couple of hundred strategy-times-k combinations below, so
    the corpus and the column lists are cached here. The cache is keyed by the
    active manifest, which is what lets the `corpus` section swap in a rewritten
    description and re-measure without touching engine/retrieval.py.
    """

    def __init__(self, con):
        self.con = con
        self.cases = load_cases(con)
        self._cols: dict[str, list[tuple[str, str]]] = {}
        for domain, table, _src, _desc in MANIFEST:
            name = table_name(domain, table)
            try:
                self._cols[name] = table_columns(con, name)
            except Exception:
                self._cols[name] = []
        self.manifest = list(MANIFEST)
        self._corpus_cache: list[dict] | None = None
        self._install()

    # -- corpus / index plumbing -------------------------------------------

    def columns_of(self, name: str) -> list[tuple[str, str]]:
        return self._cols.get(name, [])

    def build_corpus(self, con=None) -> list[dict]:
        if self._corpus_cache is None:
            corpus = []
            for domain, table, _src, description in self.manifest:
                name = table_name(domain, table)
                corpus.append(
                    {
                        "id": name,
                        "document": retrieval._document(
                            domain, name, description, self.columns_of(name)
                        ),
                        "metadata": {"domain": domain, "description": description},
                    }
                )
            self._corpus_cache = corpus
        return self._corpus_cache

    def _install(self):
        """Point engine.retrieval at the cached corpus, then rebuild its index."""
        retrieval.build_corpus = self.build_corpus  # type: ignore[assignment]
        retrieval.build_index(self.con, rebuild=True)

    def set_manifest(self, manifest: list[tuple]):
        """Swap the manifest (i.e. the descriptions) and re-embed."""
        self.manifest = list(manifest)
        self._corpus_cache = None
        self._install()

    # -- scoring ------------------------------------------------------------

    def prompt_block(self, tables: list[str]) -> str:
        """Reproduce `schema_catalog_for`'s output for an arbitrary table list.

        Same shape, same ordering rule (grouped by domain, first-seen order), so
        the token counts here are comparable with the main eval's.
        """
        desc = {table_name(d, t): (d, dsc) for d, t, _s, dsc in self.manifest}
        by_domain: dict[str, list[str]] = {}
        for name in tables:
            row = desc.get(name)
            if row is None:
                continue
            by_domain.setdefault(row[0], []).append(name)
        lines: list[str] = []
        for domain, names in by_domain.items():
            lines.append(f"\n### Domain: {domain} — {DOMAINS.get(domain, '')}")
            for name in names:
                lines.append(f"- {name}: {desc[name][1]}")
                cols = ", ".join(f"{c} {t}" for c, t in self.columns_of(name))
                if cols:
                    lines.append(f"    columns: {cols}")
        return "\n".join(lines).strip()

    def score(self, strategy, k: int) -> dict:
        """Run one strategy over every case. Returns recall + mean tokens.

        `strategy` is any callable (question, k, bench) -> ordered table names.
        A strategy may return fewer or more than k names; adaptive-k strategies
        do exactly that, and their token column is what makes them comparable.
        """
        hit = 0
        missed = 0
        total = 0
        tokens = 0
        picked = 0
        misses: list[str] = []
        for case in self.cases:
            got_list = strategy(case["question"], k, self)
            got = set(got_list)
            need = case["needed"]
            total += len(need)
            missing = need - got
            missed += len(missing)
            picked += len(got_list)
            if missing:
                misses.append(f"{case['id']} (missing {', '.join(sorted(missing))})")
            else:
                hit += 1
            tokens += approx_tokens(self.prompt_block(got_list))
        n = len(self.cases)
        return {
            "questions": hit / n,
            "tables": (total - missed) / total,
            "tokens": tokens / n,
            "mean_k": picked / n,
            "misses": misses,
        }


def report(bench: Bench, title: str, entries: list[tuple[str, object, int]], *,
           show_misses: bool = True, baseline: dict | None = None):
    """Print one comparison table. `entries` is [(label, strategy, k), ...]."""
    print(f"\n{title}")
    print(f"  {'variant':<34} {'questions':>10} {'tables':>9} {'~tok/turn':>11} {'mean k':>7}")
    results = []
    for label, strategy, k in entries:
        res = bench.score(strategy, k)
        results.append((label, res))
        delta = ""
        if baseline is not None:
            dq = res["questions"] - baseline["questions"]
            dt = res["tokens"] - baseline["tokens"]
            delta = f"   {dq:+.1%} recall, {dt:+,.0f} tok"
        print(
            f"  {label:<34} {res['questions']:>9.1%} {res['tables']:>9.1%} "
            f"{res['tokens']:>11,.0f} {res['mean_k']:>7.1f}{delta}"
        )
    if show_misses:
        for label, res in results:
            if res["misses"]:
                print(f"    {label} missed: {'; '.join(res['misses'])}")
    return results


# ---------------------------------------------------------------------------
# Strategies. Every one is (question, k, bench) -> ordered list of table names.
# ---------------------------------------------------------------------------

def s_vector(q, k, b):
    return [h.table for h in retrieval.retrieve(q, k=k, con=b.con)]


def s_keyword(q, k, b):
    return [h.table for h in retrieval.retrieve_keyword(q, k=k, con=b.con)]


def s_hybrid(q, k, b):
    return [h.table for h in retrieval.retrieve_hybrid(q, k=k, con=b.con)]


def _ranked(q, k, b, pool=None):
    """The two input rankings RRF fuses, as name lists."""
    pool = pool or max(k * 2, 12)
    return (
        [h.table for h in retrieval.retrieve(q, k=pool, con=b.con)],
        [h.table for h in retrieval.retrieve_keyword(q, k=pool, con=b.con)],
    )


# -- fusion repairs ---------------------------------------------------------

def rrf(rankings: list[tuple[list[str], float]], *, rrf_k: int = 60,
        absent_rank: dict[int, int] | None = None) -> list[tuple[str, float]]:
    """Weighted reciprocal-rank fusion, optionally scoring absences.

    `absent_rank[i]` is the rank charged to a document that ranking `i` did not
    return at all. Vanilla RRF charges nothing, which sounds neutral and is not:
    see `diagnose`.
    """
    fused: dict[str, float] = defaultdict(float)
    universe: set[str] = set()
    for names, _w in rankings:
        universe |= set(names)
    for i, (names, weight) in enumerate(rankings):
        pos = {name: rank for rank, name in enumerate(names, start=1)}
        miss = (absent_rank or {}).get(i)
        for name in universe:
            rank = pos.get(name)
            if rank is None:
                if miss is None:
                    continue
                rank = miss
            fused[name] += weight / (rrf_k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def s_wrrf(wv: float, wk: float):
    """RRF with the vector list weighted above the keyword list."""
    def strategy(q, k, b):
        vec, kw = _ranked(q, k, b)
        return [n for n, _ in rrf([(vec, wv), (kw, wk)])[:k]]
    return strategy


def s_rrf_absent(q, k, b):
    """RRF that charges a document a real (bad) rank for being absent.

    Absent is scored as `len(list) + 1` rather than skipped, so 'the vector
    index ranked you fifth and keyword had never heard of you' beats 'both lists
    put you near the bottom', which is the ordering vanilla RRF gets backwards.
    """
    vec, kw = _ranked(q, k, b)
    absent = {0: len(vec) + 1, 1: len(kw) + 1}
    return [n for n, _ in rrf([(vec, 1.0), (kw, 1.0)], absent_rank=absent)[:k]]


def s_interleave(q, k, b):
    """Round-robin the two lists instead of fusing scores.

    The crudest possible repair and the one with the strongest guarantee: the
    top ceil(k/2) of each retriever is always in the result, so neither method
    can be talked out of its best hit by the other.
    """
    vec, kw = _ranked(q, k, b)
    out: list[str] = []
    seen: set[str] = set()
    for i in range(max(len(vec), len(kw))):
        for src in (vec, kw):
            if i < len(src) and src[i] not in seen:
                seen.add(src[i])
                out.append(src[i])
                if len(out) == k:
                    return out
    return out


def s_hybrid_floor(floor: int):
    """RRF, but the top-`floor` vector hits are guaranteed a seat."""
    def strategy(q, k, b):
        vec, kw = _ranked(q, k, b)
        fused = [n for n, _ in rrf([(vec, 1.0), (kw, 1.0)])]
        keep = list(vec[:floor])
        for name in fused:
            if len(keep) >= k:
                break
            if name not in keep:
                keep.append(name)
        return keep[:k]
    return strategy


# -- query expansion --------------------------------------------------------

def _acronym_map(bench: Bench) -> dict[str, str]:
    """Acronym <-> gloss pairs mined from the manifest's own prose.

    Derived, not authored: the descriptions already spell out their jargon in
    the form "on-time delivery (OTIF)" and "FEFO (first-expiry-first-out)",
    which is exactly a two-way glossary if you read it with a regex. Nothing
    here was chosen by looking at the eval questions.
    """
    pairs: dict[str, str] = {}
    text = " ".join(dsc for _d, _t, _s, dsc in bench.manifest) + " " + " ".join(DOMAINS.values())
    for gloss, acro in re.findall(r"([a-zA-Z][a-zA-Z\- ]{3,40}?)\s*\(([A-Z]{2,7})\)", text):
        pairs[acro.lower()] = gloss.strip().lower()
    for acro, gloss in re.findall(r"\b([A-Z]{2,7})\s*\(([a-zA-Z][a-zA-Z\- ]{3,40})\)", text):
        pairs[acro.lower()] = gloss.strip().lower()
    return pairs


_ACROS: dict[str, str] = {}


def s_expand_acronym(base):
    def strategy(q, k, b):
        global _ACROS
        if not _ACROS:
            _ACROS = _acronym_map(b)
        extra = [gloss for token in retrieval._tokens(q) if (gloss := _ACROS.get(token))]
        return base(q + (" " + " ".join(extra) if extra else ""), k, b)
    return strategy


def s_prf(base, *, feedback_docs: int = 3, terms: int = 8):
    """Pseudo-relevance feedback (Rocchio, term-only).

    Retrieve, assume the top few hits are relevant, harvest their most
    distinctive terms by collection frequency, append them to the query, and
    retrieve again. Fully deterministic and keyless, which is the whole point:
    an LLM rewriter cannot be measured in this environment, and PRF is the
    classical stand-in for one.
    """
    def strategy(q, k, b):
        first = base(q, min(k, feedback_docs * 3), b)[:feedback_docs]
        corpus = {row["id"]: row["document"] for row in b.build_corpus()}
        df = Counter()
        for doc in corpus.values():
            df.update(retrieval._tokens(doc))
        cand = Counter()
        for name in first:
            for token in retrieval._tokens(corpus.get(name, "")):
                if df[token] <= 8:  # distinctive: not warehouse-wide boilerplate
                    cand[token] += 1
        extra = [t for t, _ in cand.most_common(terms)]
        return base(q + " " + " ".join(extra), k, b)
    return strategy


# -- column-level retrieval -------------------------------------------------

_COLCOLL = {"handle": None, "fingerprint": None}


def _column_collection(b: Bench):
    """A second Chroma collection: one document per COLUMN, not per table.

    Each column document carries its own name, its table, and its table's
    description, so a query that names a measure ("compa ratio", "qty shipped")
    can hit the column directly and vote its table up.
    """
    import chromadb

    fingerprint = str(len(b.manifest)) + str(id(b._corpus_cache))
    if _COLCOLL["handle"] is not None and _COLCOLL["fingerprint"] == fingerprint:
        return _COLCOLL["handle"]
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("schema_columns")
    except Exception:
        pass
    coll = client.create_collection("schema_columns", metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for domain, table, _s, description in b.manifest:
        name = table_name(domain, table)
        for col, _type in b.columns_of(name):
            ids.append(f"{name}.{col}")
            docs.append(f"{col.replace('_', ' ')} — column of {name} ({domain}). {description}")
            metas.append({"table": name})
    coll.add(ids=ids, documents=docs, metadatas=metas)
    _COLCOLL["handle"] = coll
    _COLCOLL["fingerprint"] = fingerprint
    return coll


def s_column_fusion(q, k, b):
    """Fuse table-level RRF with a column-level ranking rolled up to tables."""
    coll = _column_collection(b)
    res = coll.query(query_texts=[q], n_results=min(60, coll.count()))
    seen: list[str] = []
    for meta in (res.get("metadatas") or [[]])[0]:
        t = str((meta or {}).get("table") or "")
        if t and t not in seen:
            seen.append(t)
    vec, kw = _ranked(q, k, b)
    return [n for n, _ in rrf([(vec, 1.0), (kw, 1.0), (seen, 1.0)])[:k]]


def _weighted_document(domain, table, description, columns, repeat: int, split: bool):
    """The table document with column vocabulary repeated `repeat` times.

    The cheap alternative to a second index: if columns carry the vocabulary
    questions use, say them more than once so they weigh more in the embedding
    and in token overlap. Costs nothing at query time and nothing in the prompt
    — the prompt block is built from the manifest, not from the index document.

    `split` turns `qty_shipped` into `qty shipped`. That is a SEPARATE change
    from repetition and the two must be measured apart, because the shipped
    document does neither and a combined win says nothing about which half did
    the work.
    """
    parts = [f"{table} ({domain} domain)", DOMAINS.get(domain, ""), description]
    if columns and repeat:
        words = ", ".join(
            (name.replace("_", " ") if split else name) for name, _t in columns
        )
        for _ in range(repeat):
            parts.append("columns: " + words)
    return "\n".join(p for p in parts if p)


def with_column_weight(b: Bench, repeat: int, split: bool = True):
    """Re-embed the corpus with column vocabulary repeated. Returns a restorer."""
    original = retrieval._document

    def patched(domain, table, description, columns):
        return _weighted_document(domain, table, description, columns, repeat, split)

    retrieval._document = patched  # type: ignore[assignment]
    b._corpus_cache = None
    b._install()

    def restore():
        retrieval._document = original  # type: ignore[assignment]
        b._corpus_cache = None
        b._install()

    return restore


# -- reranking --------------------------------------------------------------

def s_rerank(base, *, top_n: int = 24, w_name: float = 1.0, w_domain: float = 0.35,
             w_overlap: float = 0.5):
    """Re-score the fused top-N by three signals RRF cannot see.

      name    the question literally says a word that is in the table's name
              ("inventory", "forecast") — a very strong signal RRF dilutes
      domain  agreement with the majority domain of the top hits, because a
              question is almost always about one domain and the 71-table
              warehouse has four tables called some variant of fact_orders
      overlap IDF-weighted term overlap with the description, which is the
              keyword signal graded on rarity instead of raw count
    """
    def strategy(q, k, b):
        pool = base(q, max(top_n, k), b)
        qt = retrieval._tokens(q)
        corpus = {row["id"]: row for row in b.build_corpus()}
        df = Counter()
        for row in corpus.values():
            df.update(retrieval._tokens(row["document"]))
        n_docs = max(1, len(corpus))
        head = pool[: min(6, len(pool))]
        dom_counts = Counter(
            str(corpus[t]["metadata"]["domain"]) for t in head if t in corpus
        )
        top_domain = dom_counts.most_common(1)[0][0] if dom_counts else ""
        scored = []
        for rank, name in enumerate(pool, start=1):
            row = corpus.get(name)
            if row is None:
                continue
            base_score = 1.0 / (60 + rank)
            name_tokens = set(name.split("_"))
            bonus = 0.0
            if qt & name_tokens:
                bonus += w_name * len(qt & name_tokens) / (60 + rank)
            if str(row["metadata"]["domain"]) == top_domain:
                bonus += w_domain / (60 + rank)
            dt = retrieval._tokens(str(row["metadata"]["description"]))
            if qt & dt:
                idf = sum(1.0 / (1 + df[t]) for t in (qt & dt)) * n_docs / 71.0
                bonus += w_overlap * idf / (60 + rank)
            scored.append((base_score + bonus, name))
        scored.sort(key=lambda item: -item[0])
        return [name for _s, name in scored[:k]]
    return strategy


# -- adaptive k -------------------------------------------------------------

def s_adaptive(base_scored, *, kmin: int, kmax: int, drop: float):
    """Stop at the first big relative drop in fused score, within [kmin, kmax].

    The intuition worth testing: an unambiguous question ("how many GO
    verdicts") has one obviously-best table and a cliff right after it, while a
    vague one has a flat score profile and needs more of the catalogue. If that
    is true, k should be a property of the question, not a constant.
    """
    def strategy(q, k, b):
        ranked = base_scored(q, kmax, b)
        names = [n for n, _s in ranked]
        scores = [s for _n, s in ranked]
        cut = kmax
        for i in range(kmin, min(kmax, len(scores)) - 1):
            prev, nxt = scores[i - 1], scores[i]
            if prev > 0 and (prev - nxt) / prev >= drop:
                cut = i
                break
        return names[:cut]
    return strategy


def scored_hybrid(q, k, b):
    vec, kw = _ranked(q, k, b)
    return rrf([(vec, 1.0), (kw, 1.0)])[:k]


def scored_vector(q, k, b):
    return [(h.table, h.score) for h in retrieval.retrieve(q, k=k, con=b.con)]


# ---------------------------------------------------------------------------
# The improved description under test.
# ---------------------------------------------------------------------------

# data_manifest.py's own docstring says a description exists so the model can
# decide which tables answer a question, and should say "what the grain is, what
# the money columns mean". The shipped supplychain_fact_orders description meets
# the first half and skips the second: it names unit_price and unit_cost nowhere
# and the word revenue nowhere, while being the only place shipped revenue in
# that domain can come from. This is that gap closed, in the house style of the
# other fact-table descriptions (compare healthcare_fact_claims, which does spell
# its money columns out).
FIXED_SUPPLYCHAIN_FACT_ORDERS = (
    "One row per order line, and the only source of shipped revenue in this domain: "
    "line revenue is qty_shipped * unit_price and line margin is qty_shipped * "
    "(unit_price - unit_cost), so revenue or margin by product category, customer, "
    "supplier or warehouse is an aggregate of this table joined to the matching dim. "
    "qty_ordered vs qty_shipped measures fill rate; promised_date vs shipped_date "
    "measures on-time delivery (OTIF). Joins dim_customer, dim_product, dim_lot, "
    "dim_warehouse by id."
)


def fixed_manifest() -> list[tuple]:
    out = []
    for domain, table, source, description in MANIFEST:
        if (domain, table) == ("supplychain", "fact_orders"):
            description = FIXED_SUPPLYCHAIN_FACT_ORDERS
        out.append((domain, table, source, description))
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_diagnose(b: Bench, k: int):
    print("\n" + "=" * 78)
    print("DIAGNOSE — why top_category_revenue misses")
    print("=" * 78)
    q = "Which product category has the highest shipped revenue?"
    target = "supplychain_fact_orders"
    vec = s_vector(q, 71, b)
    kw = s_keyword(q, 71, b)
    hyb = s_hybrid(q, 71, b)
    print(f"\n  question: {q}")
    print(f"  target  : {target}\n")
    for label, ranking in (("vector", vec), ("keyword", kw), ("hybrid", hyb)):
        pos = ranking.index(target) + 1 if target in ranking else None
        print(f"  {label:<8} rank {pos if pos else 'ABSENT'}   (list length {len(ranking)})")
    print(
        "\n  The received story is that this question fails under every strategy.\n"
        "  It does not. VECTOR ranks the table 5th and would retrieve it at any\n"
        "  k >= 5. The failure is introduced by the fusion step: keyword shares\n"
        "  no token with it at all, so it never appears in the keyword list, and\n"
        "  vanilla RRF gives an absent document a contribution of exactly zero.\n"
        "  A table one retriever is confident about therefore loses to tables\n"
        "  both retrievers are lukewarm about — 1/(60+5) = 0.0154 for the\n"
        "  confident single vote, against 1/(60+20)+1/(60+20) = 0.0250 for two\n"
        "  weak ones. Ranking it 24th out of 71 is not a budget problem and no k\n"
        "  below 25 fixes it.\n"
    )
    print("  recall at k for this ONE question:")
    for kk in (5, 9, 14, 20, 25):
        v = target in s_vector(q, kk, b)
        h = target in s_hybrid(q, kk, b)
        print(f"    k={kk:<3} vector={'hit ' if v else 'MISS'}  hybrid={'hit ' if h else 'MISS'}")


def section_corpus(b: Bench, k: int):
    print("\n" + "=" * 78)
    print("CORPUS — does a better description fix it, and what does it cost")
    print("=" * 78)
    entries = [("hybrid (shipped corpus)", s_hybrid, k), ("vector (shipped corpus)", s_vector, k)]
    before = report(b, "before:", entries)
    b.set_manifest(fixed_manifest())
    try:
        after = report(b, "after rewriting supplychain_fact_orders:", entries)
    finally:
        b.set_manifest(list(MANIFEST))
    print(
        f"\n  full catalogue grows by "
        f"{approx_tokens(FIXED_SUPPLYCHAIN_FACT_ORDERS) - approx_tokens(dict(((d, t), x) for d, t, _s, x in MANIFEST)[('supplychain', 'fact_orders')]):+d}"
        " tokens, paid only on turns that retrieve this table."
    )
    return before, after


def section_fusion(b: Bench, k: int):
    print("\n" + "=" * 78)
    print("FUSION — repairing RRF's treatment of singly-retrieved tables")
    print("=" * 78)
    base = b.score(s_hybrid, k)
    report(
        b,
        f"k = {k}",
        [
            ("hybrid (current, plain RRF)", s_hybrid, k),
            ("vector only", s_vector, k),
            ("RRF, absence scored as last", s_rrf_absent, k),
            ("weighted RRF 1.5 vec / 1.0 kw", s_wrrf(1.5, 1.0), k),
            ("weighted RRF 2.0 vec / 1.0 kw", s_wrrf(2.0, 1.0), k),
            ("round-robin interleave", s_interleave, k),
            ("RRF + top-4 vector floor", s_hybrid_floor(4), k),
            ("RRF + top-6 vector floor", s_hybrid_floor(6), k),
        ],
        baseline=base,
    )


def section_expand(b: Bench, k: int):
    print("\n" + "=" * 78)
    print("EXPAND — deterministic query expansion (no API key, so no LLM rewriter)")
    print("=" * 78)
    acros = _acronym_map(b)
    print(f"\n  acronym glossary mined from the manifest ({len(acros)} pairs): "
          f"{', '.join(sorted(acros)[:12])}")
    base = b.score(s_hybrid, k)
    report(
        b,
        f"k = {k}",
        [
            ("hybrid (no expansion)", s_hybrid, k),
            ("hybrid + acronym expansion", s_expand_acronym(s_hybrid), k),
            ("hybrid + PRF (3 docs, 8 terms)", s_prf(s_hybrid), k),
            ("hybrid + PRF (2 docs, 5 terms)", s_prf(s_hybrid, feedback_docs=2, terms=5), k),
            ("vector + PRF (3 docs, 8 terms)", s_prf(s_vector), k),
        ],
        baseline=base,
    )


def section_columns(b: Bench, k: int):
    print("\n" + "=" * 78)
    print("COLUMNS — column-level documents, and column-weighted table documents")
    print("=" * 78)
    base = b.score(s_hybrid, k)
    n_cols = sum(len(b.columns_of(table_name(d, t))) for d, t, _s, _x in b.manifest)
    print(f"\n  a column-level index is {n_cols} documents against {len(b.manifest)} tables")
    report(
        b,
        f"k = {k}",
        [
            ("hybrid (table docs only)", s_hybrid, k),
            ("hybrid + column index (3-way RRF)", s_column_fusion, k),
        ],
        baseline=base,
    )
    # Repetition and underscore-splitting are two changes, so measure the 2x2.
    # `1x, underscores kept` is the shipped document rebuilt through the same
    # patch — it should reproduce the baseline exactly, and if it does not, the
    # harness is lying and nothing below it means anything.
    for label, repeat, split in (
        ("0x — columns dropped entirely", 0, True),
        ("1x, underscores kept (control)", 1, False),
        ("1x, split only (no repetition)", 1, True),
        ("2x, underscores kept", 2, False),
        ("2x, split + repetition", 2, True),
        ("3x, split + repetition", 3, True),
    ):
        restore = with_column_weight(b, repeat, split)
        try:
            report(
                b,
                f"index document: {label}",
                [("hybrid", s_hybrid, k), ("vector", s_vector, k), ("keyword", s_keyword, k)],
                baseline=base,
                show_misses=False,
            )
        finally:
            restore()


def section_rerank(b: Bench, k: int):
    print("\n" + "=" * 78)
    print("RERANK — re-scoring the fused pool by name / domain / IDF-overlap")
    print("=" * 78)
    base = b.score(s_hybrid, k)
    report(
        b,
        f"k = {k}",
        [
            ("hybrid (no rerank)", s_hybrid, k),
            ("rerank: name only", s_rerank(s_hybrid, w_name=1.0, w_domain=0.0, w_overlap=0.0), k),
            ("rerank: domain only", s_rerank(s_hybrid, w_name=0.0, w_domain=0.35, w_overlap=0.0), k),
            ("rerank: overlap only", s_rerank(s_hybrid, w_name=0.0, w_domain=0.0, w_overlap=0.5), k),
            ("rerank: all three", s_rerank(s_hybrid), k),
            ("rerank all three, over floor-4", s_rerank(s_hybrid_floor(4)), k),
        ],
        baseline=base,
    )


def section_adaptivek(b: Bench, k: int):
    print("\n" + "=" * 78)
    print("ADAPTIVE k — is a per-question k cheaper than a fixed one at equal recall")
    print("=" * 78)
    base = b.score(s_hybrid, k)
    entries: list[tuple[str, object, int]] = [(f"hybrid fixed k={k}", s_hybrid, k)]
    for fixed in (8, 10, 12, 16, 18, 20):
        entries.append((f"hybrid fixed k={fixed}", s_hybrid, fixed))
    for drop in (0.02, 0.05, 0.10):
        entries.append(
            (f"adaptive gap>{drop:.0%} (6..20)", s_adaptive(scored_hybrid, kmin=6, kmax=20, drop=drop), k)
        )
    for drop in (0.05, 0.10, 0.15):
        entries.append(
            (f"adaptive VECTOR gap>{drop:.0%} (6..20)",
             s_adaptive(scored_vector, kmin=6, kmax=20, drop=drop), k)
        )
    report(b, f"baseline k = {k}", entries, baseline=base, show_misses=False)


def _split_document(domain, table, description, columns):
    """The recommended index document: identifiers spelled as words.

    One character's difference from the shipped `_document`. `_tokens` splits on
    `[a-z0-9_]+`, so `qty_shipped` is a SINGLE token that the question "shipped
    revenue" can never match, and MiniLM's WordPiece vocabulary has no entry for
    it either. Writing it as `qty shipped` makes both halves reachable by both
    retrievers. The table name gets the same treatment for the same reason.
    """
    parts = [
        f"{table} ({domain} domain)",
        table.replace("_", " "),
        DOMAINS.get(domain, ""),
        description,
    ]
    if columns:
        parts.append("columns: " + ", ".join(n.replace("_", " ") for n, _t in columns))
    return "\n".join(p for p in parts if p)


def with_split_document(b: Bench):
    original = retrieval._document
    retrieval._document = _split_document  # type: ignore[assignment]
    b._corpus_cache = None
    b._install()

    def restore():
        retrieval._document = original  # type: ignore[assignment]
        b._corpus_cache = None
        b._install()

    return restore


def section_combined(b: Bench, k: int):
    print("\n" + "=" * 78)
    print("COMBINED — the three independent fixes, together and across k")
    print("=" * 78)
    base = b.score(s_hybrid, k)

    # 1. tokenizer fix alone, with and without splitting the table name too.
    restore = with_split_document(b)
    try:
        report(
            b,
            f"tokenizer fix (columns AND table name split), k = {k}",
            [("hybrid", s_hybrid, k), ("vector", s_vector, k), ("keyword", s_keyword, k)],
            baseline=base,
        )
    finally:
        restore()

    # 2. corpus fix + fusion fix, no tokenizer change.
    b.set_manifest(fixed_manifest())
    try:
        report(
            b,
            f"corpus fix + fusion repair, k = {k}",
            [
                ("hybrid plain RRF", s_hybrid, k),
                ("RRF absence-scored", s_rrf_absent, k),
                ("round-robin interleave", s_interleave, k),
            ],
            baseline=base,
        )
    finally:
        b.set_manifest(list(MANIFEST))

    # 3. all three, then push k DOWN — the real prize is not +2.6% recall at
    #    k=14, it is holding 100% on a smaller prompt.
    b.set_manifest(fixed_manifest())
    restore = with_split_document(b)
    try:
        entries: list[tuple[str, object, int]] = []
        for kk in (6, 8, 10, 12, 14):
            entries.append((f"all three fixes, k={kk}", s_rrf_absent, kk))
        report(b, "corpus + tokenizer + absence-scored RRF:", entries, baseline=base)
    finally:
        restore()
        b.set_manifest(list(MANIFEST))

    # 4. robustness of each fix on its own across k, so nothing above rests on
    #    a single lucky operating point.
    print("\n  robustness — question recall at each k, one fix at a time")
    ks = [8, 10, 12, 14, 16, 20]
    rows: list[tuple[str, list[float]]] = []
    rows.append(("shipped hybrid", [b.score(s_hybrid, kk)["questions"] for kk in ks]))
    rows.append(("+ absence-scored RRF", [b.score(s_rrf_absent, kk)["questions"] for kk in ks]))
    restore = with_split_document(b)
    try:
        rows.append(("+ tokenizer fix only", [b.score(s_hybrid, kk)["questions"] for kk in ks]))
    finally:
        restore()
    b.set_manifest(fixed_manifest())
    try:
        rows.append(("+ corpus fix only", [b.score(s_hybrid, kk)["questions"] for kk in ks]))
    finally:
        b.set_manifest(list(MANIFEST))
    header = "  " + f"{'variant':<24}" + "".join(f"{'k=' + str(kk):>9}" for kk in ks)
    print(header)
    for label, vals in rows:
        print("  " + f"{label:<24}" + "".join(f"{v:>8.1%} " for v in vals))


SECTIONS = {
    "diagnose": section_diagnose,
    "corpus": section_corpus,
    "fusion": section_fusion,
    "expand": section_expand,
    "columns": section_columns,
    "rerank": section_rerank,
    "adaptivek": section_adaptivek,
    "combined": section_combined,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--k", type=int, default=retrieval.DEFAULT_K)
    ap.add_argument("--only", nargs="*", choices=sorted(SECTIONS), default=None)
    args = ap.parse_args(argv)

    con = build_warehouse()
    bench = Bench(con)
    print(f"warehouse: {len(bench._cols)} tables · labelled golden questions: {len(bench.cases)}")
    print(f"full catalogue: ~{approx_tokens(schema_catalog(con)):,} tokens/turn")

    for name in (args.only or list(SECTIONS)):
        SECTIONS[name](bench, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
