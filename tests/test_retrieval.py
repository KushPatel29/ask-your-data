"""
The retrieval contract.

These run without an API key, like almost everything else here: the embedding
model is all-MiniLM-L6-v2 running locally through Chroma's ONNX backend, so
retrieval is fully testable offline. That is a deliberate property of the
design, not a happy accident - a demo that cannot be evaluated without someone
else's billing account cannot be evaluated by a reviewer.

What is pinned here is the SHAPE of the contract, not the exact numbers. Recall
figures belong in scripts/run_retrieval_eval.py where they are reported with
their method; asserting "recall == 97.4%" in a unit test would turn a corpus
improvement into a red build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_manifest import MANIFEST  # noqa: E402
from engine import retrieval  # noqa: E402
from engine.warehouse import build_warehouse, schema_catalog  # noqa: E402


@pytest.fixture(scope="module")
def con():
    return build_warehouse()


def test_corpus_covers_every_manifest_table(con):
    """Every table the warehouse loads must be retrievable.

    A table missing from the index is invisible to the model no matter how the
    question is phrased, and nothing else in the suite would notice.
    """
    corpus = {row["id"] for row in retrieval.build_corpus(con)}
    assert len(corpus) == len(MANIFEST)


def test_retrieval_narrows_the_prompt(con):
    """The whole point: fewer tokens than pasting the catalogue."""
    question = "What is the overall claim denial rate?"
    narrowed = retrieval.schema_catalog_for(question, con)
    assert len(narrowed) < len(schema_catalog(con)), "retrieval did not shrink the prompt"


def test_retrieval_finds_the_obvious_table(con):
    """A question naming its own domain should retrieve that domain's fact table."""
    hits = retrieval.retrieve("What is the overall claim denial rate?", con=con)
    assert "healthcare_fact_claims" in {h.table for h in hits}


def test_scores_are_ordered_and_bounded(con):
    hits = retrieval.retrieve("Which payer type collects the least of what it bills?", con=con)
    assert hits, "no hits returned"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "hits are not ranked best-first"
    # Cosine similarity, so 1 - distance. Allow a little float slack either side.
    assert all(-1.01 <= s <= 1.01 for s in scores)


def test_hybrid_beats_its_parts_on_the_cases_that_motivated_it(con):
    """The two questions that justified fusing the rankings.

    Vector ranks these low (17 and 24 of 71) because the question's wording
    points somewhere else semantically; exact tokens find them immediately.
    If hybrid ever stops covering both, the fusion has stopped earning its keep
    and this test should fail loudly rather than the recall quietly sagging.
    """
    cases = [
        ("Who is the top wholesale customer by revenue?", "retail_customer_analytics"),
        ("Which site generates the most data queries per enrolled subject?", "clinical_query_log"),
    ]
    for question, needed in cases:
        got = {h.table for h in retrieval.retrieve_hybrid(question, con=con)}
        assert needed in got, f"hybrid lost {needed} for {question!r}"


def test_keyword_baseline_is_a_real_alternative(con):
    """The baseline has to actually work, or beating it proves nothing."""
    hits = retrieval.retrieve_keyword(
        "Which site generates the most data queries per enrolled subject?", con=con)
    assert "clinical_query_log" in {h.table for h in hits}


def test_empty_question_returns_nothing(con):
    assert retrieval.retrieve("", con=con) == []
    assert retrieval.retrieve_keyword("   ", con=con) == []


def test_prompt_block_keeps_the_catalogue_format(con):
    """Same shape as schema_catalog(), so the system prompt does not change."""
    block = retrieval.schema_catalog_for("How many employees left voluntarily?", con)
    assert "### Domain:" in block
    assert "    columns:" in block


def test_unknown_strategy_falls_back_rather_than_raising(con):
    """A typo in a strategy name must not take the assistant down."""
    block = retrieval.schema_catalog_for("What is the denial rate?", con, strategy="nonsense")
    assert "### Domain:" in block
