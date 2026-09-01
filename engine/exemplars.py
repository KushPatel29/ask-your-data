"""
Few-shot exemplars: show the model solved questions before asking it a new one.

WHY THIS EXISTS
The assistant already grounds the model in *schema* — `engine/retrieval.py`
selects the tables a question needs. Schema tells the model what columns exist.
It does not tell the model what a *good query against this warehouse looks
like*: that `status IN ('Paid', 'Denied')` is the denominator for a denial rate,
that a "top N" answer is a subquery plus LIMIT 1, that money gets ROUNDed in the
SQL rather than in prose. Those are conventions of this warehouse, and the
cheapest way to teach a convention is to show it. In-context examples of solved
question -> SQL pairs are the standard lever in the text-to-SQL literature for
exactly this, and this repo is already sitting on the corpus:
`evals/golden_questions.yaml` holds 39 verified pairs across 11 domains.

The examples are RETRIEVED, not fixed. A fixed handful would spend tokens on
healthcare SQL while answering a supply-chain question. The same embedding
index that picks tables can pick neighbours, so the examples the model sees are
the ones nearest the question it was actually asked.

THE LEAKAGE PROBLEM, HANDLED IN CODE RATHER THAN IN INTENT
The golden set is also the evaluation set. If `run_live_eval.py` asks "What is
the overall claim denial rate?" and this module helpfully pastes that exact
question's reference SQL into the prompt, the model is copying an answer key and
the eval score means nothing. So leave-one-out is not a convention to remember —
it is enforced twice, on every call:

  1. Automatic: any candidate whose normalised question text equals the incoming
     question is dropped. This needs no cooperation from the caller and is the
     layer that protects a harness someone writes later and forgets to wire up.
  2. Explicit: `exclude_ids` drops candidates by golden-question id. The live
     eval passes the id of the case under test, which also covers a paraphrased
     or reworded probe of the same case.

`tests/test_exemplars.py` asserts that no golden question can retrieve itself.

FAILURE IS SILENT AND CHEAP
A dead index is rebuilt once. If local inference is genuinely unavailable (the
same failure modes retrieval.py documents — no ONNX Runtime, or the MiniLM
download timing out on a free tier), this returns no exemplars. That is exactly the
prompt the assistant sent before this module existed. An optimisation may cost
tokens when it fails; it must not cost answers.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "evals" / "golden_questions.yaml"

# Three, not five, and the number is read off a measurement rather than chosen
# by taste. Selecting neighbours for all 39 golden questions and scoring each
# RANK separately (self excluded), against the reference-SQL tables:
#
#   neighbour rank   from the same domain   shares a table with the question
#         1                 51.3%                       28.2%
#         2                 33.3%                       10.3%
#         3                 35.9%                       10.3%
#         4                 30.8%                        5.1%
#         5                 20.5%                        5.1%
#
# Ranks 4 and 5 are where table overlap collapses to the random-baseline floor
# (3.4%), so they are paying ~77 tokens each to show the model an unrelated
# query. k=3 costs ~231 tokens per turn — 7% on top of the ~3,241-token schema
# block — and stops before that cliff.
DEFAULT_K = 3

# Reciprocal-rank fusion constant, the same one engine/retrieval.py uses.
RRF_K = 60

_COLLECTION = "golden_exemplars"
_collection = None
_cases: list[dict] | None = None
_index_lock = threading.RLock()


def _serialised(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _index_lock:
            return fn(*args, **kwargs)
    return wrapped


def _forget_index() -> None:
    global _collection
    with _index_lock:
        _collection = None


@dataclass(frozen=True)
class Exemplar:
    """One verified question -> SQL pair, selected for a new question."""
    id: str
    domain: str
    question: str
    sql: str
    score: float


# --------------------------------------------------------------------------
# The corpus: one document per golden question.
# --------------------------------------------------------------------------

def _normalise(question: str) -> str:
    """Comparison key for the automatic leave-one-out check.

    Case, surrounding whitespace, internal run-length and a trailing question
    mark are all noise for "is this the same question"; anything else is signal
    and is deliberately left alone. A looser rule (token-set overlap, say) would
    start dropping *legitimately* similar questions, which is the one thing an
    exemplar selector must not do.
    """
    return re.sub(r"\s+", " ", (question or "").strip().lower()).rstrip("?!. ")


def _collapse(sql: str) -> str:
    """One-line, single-spaced SQL.

    Byte-stability matters here: this text lands in the prompt prefix, and the
    caching strategy depends on the same question producing the same bytes. The
    YAML folded scalars already vary in trailing whitespace between entries.
    """
    return re.sub(r"\s+", " ", (sql or "").strip())


def load_cases(path: Path | None = None) -> list[dict]:
    """The golden set, as plain dicts. Cached — the file does not change at run time."""
    global _cases
    if _cases is not None and path is None:
        return _cases
    rows = yaml.safe_load((path or GOLDEN_PATH).read_text(encoding="utf-8")) or []
    cases = [
        {
            "id": str(row["id"]),
            "domain": str(row.get("domain", "")),
            "question": str(row["question"]),
            "sql": _collapse(row.get("sql", "")),
            "key": _normalise(row["question"]),
        }
        for row in rows
        if row.get("question") and row.get("sql")
    ]
    if path is None:
        _cases = cases
    return cases


def _document(case: dict) -> str:
    """The text a golden question is indexed under.

    Question text plus domain, and NOT the SQL. Indexing the SQL would match on
    table names, which pulls in questions that merely touch the same table
    rather than questions that ask the same *shape* of thing — and the shape is
    what an exemplar teaches.
    """
    return f"{case['question']} ({case['domain']} domain)"


def _fingerprint(cases: list[dict]) -> str:
    payload = "\n".join(f"{c['id']}\0{_document(c)}" for c in cases)
    return sha256(payload.encode("utf-8")).hexdigest()


@_serialised
def build_index(*, rebuild: bool = False):
    """Create or reuse the exact local index holding the golden questions.

    Ephemeral and separate from the schema collection on purpose: a question
    document and a table document are different corpora, and mixing them into
    one collection would make every `retrieve()` call rank exemplars against
    tables on a single similarity scale that means nothing.
    """
    from engine.vector_index import LocalVectorIndex

    global _collection

    if _collection is not None and not rebuild:
        return _collection

    cases = load_cases()
    fingerprint = _fingerprint(cases)

    rows = [
        {
            "id": case["id"],
            "document": _document(case),
            "metadata": {
                "domain": case["domain"],
                "question": case["question"],
                "sql": case["sql"],
            },
        }
        for case in cases
    ]
    _collection = LocalVectorIndex.build(_COLLECTION, rows, fingerprint)
    return _collection


# --------------------------------------------------------------------------
# Selection.
# --------------------------------------------------------------------------

def _tables_in(sql: str, known) -> set[str]:
    """Warehouse tables named by a reference query.

    Same substring-on-word-boundary approach `scripts/run_retrieval_eval.py`
    uses to derive its labels, and for the same reason: these queries nest CTEs
    and subqueries, and a regex that tries to parse FROM/JOIN structure quietly
    misses one.
    """
    lowered = (sql or "").lower()
    return {name for name in known
            if re.search(rf"\b{re.escape(name.lower())}\b", lowered)}


def select_exemplars(
    question: str,
    *,
    k: int = DEFAULT_K,
    exclude_ids: object = (),
    retrieved_tables: object = None,
    query_embedding=None,
) -> list[Exemplar]:
    """The k golden questions most similar to `question`, best first.

    Leave-one-out is applied here, not by the caller:
      * a candidate whose normalised text equals `question` is always dropped;
      * a candidate whose id is in `exclude_ids` is always dropped.

    Over-fetches before filtering so that dropping the self-match still returns
    k neighbours rather than k-1.

    `retrieved_tables` — the table names `engine.retrieval` already selected for
    this question — turns on a second ranking signal, fused by reciprocal rank
    with the text similarity. Measured over all 39 golden questions at k=3:

        selector                     same domain   shares a table   table IoU
        text only                        40.2%          16.2%          0.14
        text + retrieved-table RRF       51.3%          19.7%          0.17
        random neighbour (baseline)       8.5%           3.4%          0.03
        (achievable ceiling)             84.6%

    The ceiling is not 100%: `migration` has a single golden question and so has
    no same-domain neighbour to find at all, and three more domains have two.

    This stays leak-free: the re-rank signal is derived from what retrieval
    picks out of the QUESTION, never from the question's own reference SQL.

    `query_embedding` lets a caller that has already embedded the question (the
    schema retrieval on the same turn) hand the vector over instead of paying
    for a second MiniLM forward pass. Embedding, not search, is what costs:
    ~450 ms against ~5 ms for the HNSW query on this corpus, so sharing one
    embedding across both collections roughly halves per-turn retrieval time.
    """
    question = (question or "").strip()
    if not question or k < 1:
        return []

    excluded = {str(i) for i in (exclude_ids or ())}
    self_key = _normalise(question)
    cases = {c["id"]: c for c in load_cases()}

    try:
        collection = build_index()
        # Over-fetch: the whole corpus when re-ranking (RRF needs full rankings
        # to fuse), otherwise just enough to survive the leave-one-out drops.
        pool = len(cases) if retrieved_tables else min(
            len(cases), k + len(excluded) + 2)
        query_args = ({"query_embeddings": query_embedding}
                      if query_embedding is not None else {"query_texts": [question]})
        result = collection.query(**query_args, n_results=max(1, pool))
    except Exception:
        # A collection can be deleted or invalidated after startup. Retry once
        # with a fresh handle; only a genuine local-model failure should
        # degrade to the pre-exemplar prompt.
        _forget_index()
        try:
            collection = build_index(rebuild=True)
            result = collection.query(**query_args, n_results=max(1, pool))
        except Exception:
            return []

    ids = [i for i in (result.get("ids") or [[]])[0] if i in cases]
    dists = (result.get("distances") or [[]])[0]
    similarity = {
        case_id: round(1.0 - float(dists[index]), 4) if index < len(dists) else 0.0
        for index, case_id in enumerate((result.get("ids") or [[]])[0])
    }

    if retrieved_tables:
        wanted = {str(t) for t in retrieved_tables}
        overlap = sorted(
            ((len(_tables_in(case["sql"], wanted)), case_id)
             for case_id, case in cases.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        by_overlap = [case_id for count, case_id in overlap if count]
        fused: dict[str, float] = {}
        for ranking in (ids, by_overlap):
            for rank, case_id in enumerate(ranking, start=1):
                fused[case_id] = fused.get(case_id, 0.0) + 1.0 / (RRF_K + rank)
        ids = [case_id for case_id, _ in sorted(fused.items(),
                                                key=lambda kv: -kv[1])]

    out: list[Exemplar] = []
    for case_id in ids:
        case = cases[case_id]
        if case_id in excluded or case["key"] == self_key:
            continue
        out.append(Exemplar(
            id=case_id,
            domain=case["domain"],
            question=case["question"],
            sql=case["sql"],
            score=similarity.get(case_id, 0.0),
        ))
        if len(out) == k:
            break
    return out


# --------------------------------------------------------------------------
# The prompt block.
# --------------------------------------------------------------------------

EXEMPLAR_HEADER = """SOLVED EXAMPLES
===============
Verified question/SQL pairs from this same warehouse, closest to the question \
being asked. They show the conventions this data expects — they are NOT the \
answer to the current question, and their tables may not be the right ones.
"""


def exemplar_block(
    question: str,
    *,
    k: int = DEFAULT_K,
    exclude_ids: object = (),
    retrieved_tables: object = None,
    query_embedding=None,
) -> str:
    """The formatted few-shot block, or "" when there is nothing to show.

    Returning "" rather than a header with no examples matters: an empty
    section reads to the model as "there are no similar examples, so this
    question is unusual", which is not what a local-model failure means.
    """
    picks = select_exemplars(question, k=k, exclude_ids=exclude_ids,
                             retrieved_tables=retrieved_tables,
                             query_embedding=query_embedding)
    if not picks:
        return ""
    lines = [EXEMPLAR_HEADER]
    for pick in picks:
        lines.append(f"-- {pick.question}")
        lines.append(pick.sql)
        lines.append("")
    return "\n".join(lines).strip()


if __name__ == "__main__":  # pragma: no cover - manual inspection
    import sys

    q = " ".join(sys.argv[1:]) or "how many people quit last year?"
    print(f"Q: {q}\n")
    print(exemplar_block(q))
