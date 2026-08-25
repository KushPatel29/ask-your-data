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

Backend: Chroma, with its default all-MiniLM-L6-v2 ONNX embedding function —
384 dimensions, runs locally, no API key and no torch. That matters here: the
demo has to work for a reviewer who has not set a key, which is the same reason
the rest of this repo runs keyless.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_manifest import DOMAINS, MANIFEST, table_name  # noqa: E402

# Read off the recall curve in scripts/run_retrieval_eval.py, not chosen by taste.
# k=9 is the smallest k at which every golden question's reference tables are all
# retrieved (100% recall) - k=8 leaves one question short. It costs ~857 tokens
# against the full catalogue's ~2,738, so perfect recall for a third of the
# context. The corrected keyword baseline tops out at 75% table recall through
# k=10, which is the finding that justifies an embedding index being here.
DEFAULT_K = 9

_COLLECTION = "schema_objects"
_client = None
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

def _persist_dir() -> str | None:
    raw = (os.getenv("ASK_RETRIEVAL_PERSIST_DIR") or "").strip()
    return raw or None


def build_index(con=None, *, rebuild: bool = False):
    """Create (or reuse) the Chroma collection holding the schema documents.

    Ephemeral by default. Thirty-odd documents embed in well under a second, so
    persisting buys nothing at this size and costs a whole class of bug - an
    index that silently describes a warehouse the code no longer builds. Set
    ASK_RETRIEVAL_PERSIST_DIR when the corpus is big enough for that trade to
    flip.
    """
    global _client, _collection, _collection_source

    source = "warehouse" if con is not None else "manifest"
    if _collection is not None and _collection_source == source and not rebuild:
        return _collection
    corpus = build_corpus(con)
    fingerprint = _corpus_fingerprint(corpus)

    import chromadb

    persist = _persist_dir()
    _client = chromadb.PersistentClient(path=persist) if persist else chromadb.EphemeralClient()

    if rebuild:
        try:
            _client.delete_collection(_COLLECTION)
        except Exception:
            pass

    try:
        _collection = _client.get_collection(_COLLECTION)
        metadata = _collection.metadata or {}
        if (
            _collection.count() == len(corpus)
            and metadata.get("corpus_fingerprint") == fingerprint
            and not rebuild
        ):
            _collection_source = source
            return _collection
        _client.delete_collection(_COLLECTION)
    except Exception:
        pass

    # Cosine, not the L2 default: these documents differ a lot in length (a
    # one-line dimension against a six-line fact table) and L2 would rank on
    # magnitude as much as on meaning.
    _collection = _client.create_collection(
        _COLLECTION,
        metadata={"hnsw:space": "cosine", "corpus_fingerprint": fingerprint},
    )
    _collection.add(
        ids=[row["id"] for row in corpus],
        documents=[row["document"] for row in corpus],
        metadatas=[row["metadata"] for row in corpus],
    )
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
                # Chroma returns cosine DISTANCE; similarity is 1 - d.
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
    raw = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return {token for token in raw if token not in _STOP and len(token) > 2}


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
# The prompt block.
# --------------------------------------------------------------------------

def schema_catalog_for(
    question: str,
    con,
    *,
    k: int = DEFAULT_K,
    strategy: str = "vector",
    include_tables: tuple[str, ...] | list[str] = (),
) -> str:
    """A `schema_catalog()`-shaped block containing only the retrieved tables.

    Same format as the full catalogue on purpose: the system prompt, the tests
    and the reader all already know how to read that shape, and a retrieval
    change should not also be a prompt-format change.
    """
    from engine.warehouse import table_columns

    picker = retrieve_keyword if strategy == "keyword" else retrieve
    hits = picker(question, k=k, con=con)

    # A follow-up such as "and by region?" has almost no standalone retrieval
    # signal. The assistant passes tables used by prior-turn SQL here, making
    # conversation memory a hard inclusion guarantee rather than a hope that
    # the embedding happens to rank the old table in the top k again.
    known = {
        table_name(domain, table): {"domain": domain, "description": description}
        for domain, table, _source, description in MANIFEST
    }
    seen = {hit.table for hit in hits}
    for name in include_tables:
        row = known.get(name)
        if row is None or name in seen:
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
        from engine.warehouse import schema_catalog

        return schema_catalog(con)

    by_domain: dict[str, list[RetrievedTable]] = {}
    for hit in hits:
        by_domain.setdefault(hit.domain, []).append(hit)

    lines: list[str] = []
    for domain, rows in by_domain.items():
        lines.append(f"\n### Domain: {domain} — {DOMAINS.get(domain, '')}")
        for hit in rows:
            try:
                cols = ", ".join(f"{c} {t}" for c, t in table_columns(con, hit.table))
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
