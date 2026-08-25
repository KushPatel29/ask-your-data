"""
Assistant tests.

The offline test proves the prompt is wired correctly (the schema is embedded,
the tools are well-formed) without touching the API. The live test runs a couple
of golden questions through the real model and is skipped automatically when no
API key is configured, so CI stays green without a key.
"""

import os
from pathlib import Path

import pytest
import yaml

from data_manifest import MANIFEST
from engine import retrieval
from engine.assistant import TOOLS, Assistant, _format_result
from engine.query import run_query
from engine.warehouse import schema_catalog

ROOT = Path(__file__).resolve().parent.parent
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def full_catalog(_question, con, **_kwargs):
    return schema_catalog(con)


def test_system_prompt_embeds_the_schema(con, monkeypatch):
    # a dummy client is never called — we only inspect the built prompt
    assistant = Assistant(con, client=object(), catalog_builder=full_catalog)
    prompt = "\n".join(block["text"] for block in assistant._system_for("claims", []))
    assert "healthcare_fact_claims" in prompt
    assert "hr_fact_employees" in prompt
    assert "SELECT statements only" in prompt

    corpus = retrieval.build_corpus(con)
    assert len(corpus) == len(MANIFEST)
    assert len({row["id"] for row in corpus}) == len(corpus)
    claims = next(row for row in corpus if row["id"] == "healthcare_fact_claims")
    assert "claim_id" in claims["document"]

    class FakeCollection:
        def count(self):
            return 2

        def query(self, *, query_texts, n_results):
            assert query_texts == ["claim denial rate"] and n_results == 2
            return {
                "ids": [["healthcare_fact_claims", "healthcare_dim_denial_reason"]],
                "metadatas": [[
                    {"domain": "healthcare", "description": "claims"},
                    {"domain": "healthcare", "description": "denial reasons"},
                ]],
                "distances": [[0.1, 0.25]],
            }

    monkeypatch.setattr(retrieval, "build_index", lambda con: FakeCollection())
    hits = retrieval.retrieve("claim denial rate", k=99, con=con)
    assert [hit.score for hit in hits] == [0.9, 0.75]
    assert retrieval.retrieve("   ", con=con) == []

    monkeypatch.setattr(
        retrieval,
        "retrieve",
        lambda _question, *, k, con: [
            retrieval.RetrievedTable(
                "healthcare_fact_claims",
                "healthcare",
                "claim transactions",
                0.9,
            )
        ],
    )
    narrowed = retrieval.schema_catalog_for(
        "and by payer?",
        con,
        include_tables=("healthcare_dim_payer",),
    )
    assert "healthcare_fact_claims" in narrowed
    assert "healthcare_dim_payer" in narrowed
    assert "hr_fact_employees" not in narrowed


def test_tools_are_well_formed():
    names = {t["name"] for t in TOOLS}
    assert names == {"answer_with_sql", "cannot_answer"}
    sql_tool = next(t for t in TOOLS if t["name"] == "answer_with_sql")
    assert "sql" in sql_tool["input_schema"]["required"]


def test_format_result_is_readable(con):
    res = run_query(
        con, "SELECT payer_type, COUNT(*) AS n FROM healthcare_dim_payer GROUP BY 1 ORDER BY 1")
    text = _format_result(res)
    assert "payer_type | n" in text


@pytest.mark.skipif(not HAS_KEY, reason="no ANTHROPIC_API_KEY — live model test skipped")
def test_live_assistant_answers_golden_sample(con):
    golden = yaml.safe_load((ROOT / "evals" / "golden_questions.yaml").read_text(encoding="utf-8"))
    sample = [c for c in golden if c["id"] in ("total_claims", "active_employees", "top_customer")]
    assistant = Assistant(con)
    for case in sample:
        res = assistant.ask(case["question"])
        assert res.ok, f"{case['id']}: {res.reason or res.result.error}"
        got = res.result.rows[0][0]
        expect = case["expect"]
        if isinstance(expect, float):
            assert abs(float(got) - expect) < 0.05, f"{case['id']}: {got} != {expect}"
        else:
            assert str(got).strip() == str(expect), f"{case['id']}: {got!r} != {expect!r}"
