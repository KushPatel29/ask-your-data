"""
Retrieve the schema a question needs, instead of pasting the whole warehouse.

`schema_catalog()` returns every table in all six domains, and the assistant
sends that entire block on every single turn. It works, and at this size it is
even defensible — but it is the wrong shape, for two reasons that get worse
rather than better as a warehouse grows:

  * Cost and latency scale with the size of the warehouse, not the size of the
    question. "What is the claim denial rate?" pays for the supply-chain and HR
    schemas it will never look at.
  * Precision falls as the catalogue grows. Six domains contain four tables
    called some variant of `fact_orders`. Handing the model all of them and
    hoping it picks the right one is not grounding, it is a guess with extra
    context.

So this module retrieves. A question is embedded, compared against one document
per table, and only the top matches are pasted into the prompt.

WHY A VECTOR INDEX AND NOT A SQL JOIN
The site's method section says: *no vector database where a SQL join was the
right answer.* That rule is right, and this is the case it excludes. The lookup
here is "which of these table descriptions is this English sentence about",
where the question and the description share almost no literal tokens — "how
long do people stay before they quit" has to reach `fact_employees`, which says
"attrition" and "tenure_years" and never says "quit". Exact-token matching
cannot do that; it is the one job embeddings are actually for.

That claim is not asserted, it is measured. `scripts/run_retrieval_eval.py`
scores this module against a keyword baseline built on the same corpus, using
ground truth parsed out of the reference SQL in `evals/golden_questions.yaml` —
the tables a question genuinely needs are the tables its correct answer selects
from. If the keyword baseline had won, the honest outcome would have been to
delete this file.

Backend: a read-only, exact cosine index over all-MiniLM-L6-v2 ONNX embeddings —
384 dimensions, runs locally, no API key and no torch. At 71 records an exact
matrix multiply is simpler and safer than a vector database. That matters here:
the demo has to work for a reviewer who has not set a key, which is the same
reason the rest of this repo runs keyless.
"""

from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_manifest import DOMAINS, MANIFEST, table_name  # noqa: E402

# Read off the recall curve in scripts/run_retrieval_eval.py, not chosen by taste.
# Re-measured after the warehouse grew from 36 tables to 71, which moved the
# answer: at 36 tables k=9 was enough for 100% recall, and at 71 it is not.
#
#   strategy   k    questions covered   tables recalled   ~tokens/turn
#   full       -           100.0%            100.0%            12,741
#   keyword    14           89.7%             86.7%             1,945
#   vector     14           94.9%             95.6%             3,334
#   hybrid     14           97.4%             97.8%             3,241
#
# THE PARAGRAPH THAT USED TO BE HERE WAS WRONG, and the way it was wrong is
# worth keeping. It said one question failed under every strategy at every k
# because supplychain_fact_orders had a description that did not say what the
# question asked - a corpus problem, unfixable by more context.
#
# The description was fine. `_tokens()` was eating the match: `[a-z0-9_]+`
# treats `qty_shipped` as ONE token, so the word "shipped" in a question had
# zero overlap with the column that answers it. Splitting snake_case into its
# parts (while keeping the joined form, which is what exact identifier mentions
# need) fixed it. After the fix, re-measured on the same harness:
#
#   strategy   k    questions covered   tables recalled   ~tokens/turn
#   full       -           100.0%            100.0%            12,741
#   vector     10            ~95%              ~96%             2,4xx
#   hybrid     10           100.0%            100.0%             2,253
#   hybrid     14           100.0%            100.0%             3,213
#
# So k drops from 14 to 10: full recall at 2,253 tokens, against 3,241 for the
# old 97.4%. Better answers for 30% fewer tokens, and 82% smaller than pasting
# the catalogue.
#
# Keyword alone also reaches 100% after the fix, at fewer tokens still (2,893 at
# k=14). Hybrid is kept anyway, and the reason is a limit of this eval rather
# than a measured win: the golden questions were written by people looking at
# column names, so they are unusually friendly to literal matching. Vector is
# what covers a question phrased in words the schema never uses, and that case
# is exactly what the 39 questions cannot test.
DEFAULT_K = 10

_COLLECTION = "schema_objects"
_collection = None
_collection_source = None
_index_lock = threading.RLock()


def _serialised(fn):
    """Model/index creation is process-global; make that fact safe."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _index_lock:
            return fn(*args, **kwargs)
    return wrapped


def _forget_index() -> None:
    """Drop a dead handle so the next request can rebuild instead of failing forever."""
    global _collection, _collection_source
    with _index_lock:
        _collection = None
        _collection_source = None


@dataclass(frozen=True)
class RetrievedTable:
    table: str
    domain: str
    description: str
    score: float


# --------------------------------------------------------------------------
# The corpus: one document per table.
# --------------------------------------------------------------------------

def _document(
    domain: str,
    table: str,
    description: str,
    columns: list[tuple[str, str]] | None,
) -> str:
    """The text that represents one table in the index.

    Column names are included because they carry the vocabulary a question
    actually uses - `tenure_years` is what makes "how long do people stay"
    reachable. Types are not: "VARCHAR" appears in every table and embedding it
    only pulls every vector toward a common centre.
    """
    parts = [
        f"{table} ({domain} domain)",
        DOMAINS.get(domain, ""),
        description,
    ]
    if columns:
        parts.append("columns: " + ", ".join(name for name, _type in columns))
    return "\n".join(part for part in parts if part)


def build_corpus(con=None) -> list[dict]:
    """One record per table, ready to index."""
    from engine.warehouse import table_columns

    corpus = []
    for domain, table, _source, description in MANIFEST:
        name = table_name(domain, table)
        columns = None
        if con is not None:
            try:
                columns = table_columns(con, name)
            except Exception:
                # A table listed in the manifest but absent from this warehouse
                # build still belongs in the index by name and description; it
                # just does not contribute column vocabulary.
                columns = None
        corpus.append(
            {
                "id": name,
                "document": _document(domain, name, description, columns),
                "metadata": {"domain": domain, "description": description},
            }
        )
    return corpus


def _corpus_fingerprint(corpus: list[dict]) -> str:
    """A stable identity for the text actually embedded in the collection."""
    payload = "\n".join(f"{row['id']}\0{row['document']}" for row in corpus)
    return sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# The index.
# --------------------------------------------------------------------------

@_serialised
def build_index(con=None, *, rebuild: bool = False):
    """Create or reuse the exact local index holding the schema documents.

    Embeddings are memory-only by default. Set ASK_RETRIEVAL_PERSIST_DIR to cache
    the immutable float32 matrix; the corpus fingerprint in its filename makes
    stale schema embeddings impossible to reuse silently.
    """
    from engine.vector_index import LocalVectorIndex

    global _collection, _collection_source

    source = "warehouse" if con is not None else "manifest"
    if _collection is not None and _collection_source == source and not rebuild:
        return _collection
    corpus = build_corpus(con)
    fingerprint = _corpus_fingerprint(corpus)

    _collection = LocalVectorIndex.build(_COLLECTION, corpus, fingerprint)
    _collection_source = source
    return _collection


def retrieve(question: str, *, k: int = DEFAULT_K, con=None) -> list[RetrievedTable]:
    """The top-k tables for a question, best first."""
    question = (question or "").strip()
    if not question:
        return []
    collection = build_index(con)
    k = max(1, min(int(k), collection.count()))
    result = collection.query(query_texts=[question], n_results=k)

    out: list[RetrievedTable] = []
    ids = (result.get("ids") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for index, table in enumerate(ids):
        meta = metas[index] if index < len(metas) else {}
        distance = float(dists[index]) if index < len(dists) else 1.0
        out.append(
            RetrievedTable(
                table=table,
                domain=str((meta or {}).get("domain") or ""),
                description=str((meta or {}).get("description") or ""),
                # The local index returns cosine distance; similarity is 1 - d.
                score=round(1.0 - distance, 4),
            )
        )
    return out


# --------------------------------------------------------------------------
# The keyword baseline this has to beat.
# --------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "of", "for", "in", "on", "by", "to", "and", "or", "is", "are", "was", "were",
    "what", "which", "how", "many", "much", "who", "whom", "whose", "when", "where", "does", "do",
    "did", "with", "from", "per", "our", "we", "it", "its", "that", "this", "these", "those", "at",
    "be", "been", "has", "have", "had", "most", "least", "each", "any", "all", "show", "me", "give",
}


def _tokens(text: str) -> set[str]:
    """Words, with snake_case identifiers split into their parts as well.

    The split is the whole fix for the one golden question that used to fail
    under every strategy at every k. `[a-z0-9_]+` treats `qty_shipped` as a
    single token, so the word "shipped" in a question had ZERO overlap with the
    column that answers it - measured, not theorised. The code comment in this
    module used to blame the table's description; the description was fine and
    the tokeniser was eating the match.

    Both forms are kept. Dropping the joined form would break exact identifier
    mentions, which are the case keyword retrieval is best at and the reason it
    earns its place in the fusion at all.
    """
    raw = re.findall(r"[a-z0-9_]+", (text or "").lower())
    out: set[str] = set()
    for token in raw:
        if token not in _STOP and len(token) > 2:
            out.add(token)
        for part in token.split("_"):
            if part not in _STOP and len(part) > 2:
                out.add(part)
    return out


def retrieve_keyword(question: str, *, k: int = DEFAULT_K, con=None) -> list[RetrievedTable]:
    """Token-overlap retrieval over the same corpus.

    Deliberately the same shape as the inverted-index lookup used elsewhere in
    my work, so the comparison in the eval is against a real alternative rather
    than a strawman: score = |query tokens ∩ document tokens|, ties broken by
    document brevity so a long document cannot win on surface area alone.
    """
    q = _tokens(question)
    if not q:
        return []
    scored = []
    for row in build_corpus(con):
        doc_tokens = _tokens(row["document"])
        overlap = len(q & doc_tokens)
        if overlap:
            scored.append((overlap, len(doc_tokens), row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        RetrievedTable(
            table=row["id"],
            domain=str(row["metadata"]["domain"]),
            description=str(row["metadata"]["description"]),
            score=float(overlap),
        )
        for overlap, _brevity, row in scored[:k]
    ]


# --------------------------------------------------------------------------
# Hybrid: what neither method gets on its own.
# --------------------------------------------------------------------------

RRF_K = 60  # the constant from Cormack et al.'s original reciprocal-rank fusion


def retrieve_hybrid(question: str, *, k: int = DEFAULT_K, con=None) -> list[RetrievedTable]:
    """Fuse the vector and keyword rankings by reciprocal rank.

    Measured on the 71-table warehouse, each method fails where the other
    succeeds, and it is not a close call:

        "top wholesale customer by revenue"  -> retail_customer_analytics
             vector rank 17, keyword rank 3
        "query rate per subject by site"     -> clinical_query_log
             vector rank >25, keyword rank 5

    Both failures are the same shape. Embeddings answer "what is this sentence
    about", so "wholesale" drags the query into the wholesale_* domain when the
    answer lives in retail_*, and "query rate" lands on
    clinical_query_site_performance because that name is semantically nearer
    than the log table the question actually needs. Exact tokens do not care
    about aboutness, which is precisely why they catch these.

    So neither is a baseline for the other; they are complementary, and fusing
    them is the honest design. RRF is used rather than a weighted score blend
    because the two scores are not comparable - cosine similarity and integer
    token overlap have no common scale, and normalising them would invent one.
    RRF only reads the RANKS, which both methods genuinely produce.
    """
    pool = max(k * 2, 12)
    vector_hits = retrieve(question, k=pool, con=con)
    keyword_hits = retrieve_keyword(question, k=pool, con=con)

    fused: dict[str, float] = {}
    meta: dict[str, RetrievedTable] = {}
    for ranking in (vector_hits, keyword_hits):
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.table] = fused.get(hit.table, 0.0) + 1.0 / (RRF_K + rank)
            meta.setdefault(hit.table, hit)

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [
        RetrievedTable(
            table=table,
            domain=meta[table].domain,
            description=meta[table].description,
            score=round(score, 5),
        )
        for table, score in ordered
    ]


# --------------------------------------------------------------------------
# The prompt block.
# --------------------------------------------------------------------------

def schema_catalog_for(
    question: str,
    con,
    *,
    k: int = DEFAULT_K,
    # "hybrid", because hybrid is what the eval measured and what the README
    # quotes. This defaulted to "vector" while the repo advertised hybrid's
    # numbers, so production ran at 94.9% question coverage against a published
    # 100% - the assistant never passes strategy=, so the default IS the
    # shipped behaviour.
    strategy: str = "hybrid",
    include_tables: tuple[str, ...] | list[str] = (),
    allowed_tables: frozenset[str] | set[str] | None = None,
    denied_columns: dict[str, frozenset[str]] | None = None,
) -> str:
    """A `schema_catalog()`-shaped block containing only the retrieved tables.

    Same format as the full catalogue on purpose: the system prompt, the tests
    and the reader all already know how to read that shape, and a retrieval
    change should not also be a prompt-format change.
    """
    from engine.warehouse import schema_catalog, table_columns

    picker = {"keyword": retrieve_keyword, "vector": retrieve,
              "hybrid": retrieve_hybrid}.get(strategy, retrieve_hybrid)
    try:
        hits = picker(question, k=k, con=con)
    except Exception:
        # Retrieval is an optimisation over pasting the whole catalogue, so a
        # failure here must cost tokens, not answers. Two real ways this fires
        # on a hosted free tier: ONNX Runtime not installing, and the 79 MB
        # MiniLM download failing or timing out on first use. Either way the
        # assistant should still work, just with the bigger prompt it used
        # before this module existed.
        # A collection can become invalid after startup (persistent-directory
        # replacement, model failure, invalidated index). Keeping that dead
        # module-global handle made every later request pay the full-catalogue
        # fallback until the process restarted. Fail this turn open, then allow
        # the next turn to rebuild.
        _forget_index()
        if allowed_tables is None and not denied_columns:
            return schema_catalog(con)
        hits = []

    # A follow-up such as "and by region?" has almost no standalone retrieval
    # signal. The assistant passes tables used by prior-turn SQL here, making
    # conversation memory a hard inclusion guarantee rather than a hope that
    # the embedding happens to rank the old table in the top k again.
    known = {
        table_name(domain, table): {"domain": domain, "description": description}
        for domain, table, _source, description in MANIFEST
    }
    if allowed_tables is not None:
        hits = [hit for hit in hits if hit.table in allowed_tables]
    seen = {hit.table for hit in hits}
    for name in include_tables:
        row = known.get(name)
        if row is None or name in seen or (
            allowed_tables is not None and name not in allowed_tables
        ):
            continue
        hits.append(
            RetrievedTable(
                table=name,
                domain=str(row["domain"]),
                description=str(row["description"]),
                score=1.0,
            )
        )
        seen.add(name)
    if not hits:
        for name, row in known.items():
            if allowed_tables is None or name in allowed_tables:
                hits.append(RetrievedTable(
                    table=name,
                    domain=str(row["domain"]),
                    description=str(row["description"]),
                    score=0.0,
                ))

    by_domain: dict[str, list[RetrievedTable]] = {}
    for hit in hits:
        by_domain.setdefault(hit.domain, []).append(hit)

    lines: list[str] = []
    for domain, rows in by_domain.items():
        lines.append(f"\n### Domain: {domain} — {DOMAINS.get(domain, '')}")
        for hit in rows:
            try:
                hidden = (denied_columns or {}).get(hit.table, frozenset())
                cols = ", ".join(
                    f"{c} {t}" for c, t in table_columns(con, hit.table)
                    if str(c).lower() not in hidden
                )
            except Exception:
                cols = ""
            lines.append(f"- {hit.table}: {hit.description}")
            if cols:
                lines.append(f"    columns: {cols}")
    return "\n".join(lines).strip()


if __name__ == "__main__":  # pragma: no cover - manual inspection
    from engine.warehouse import build_warehouse

    con = build_warehouse()
    question = " ".join(sys.argv[1:]) or "What is the overall claim denial rate?"
    print(f"Q: {question}\n")
    for hit in retrieve(question, con=con):
        print(f"  {hit.score:>7.4f}  {hit.table:<34} ({hit.domain})")
