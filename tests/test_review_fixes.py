"""Exercise the actual app boundaries behind the September 8 review findings."""

import ast
import hashlib
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import streamlit

from engine import deadline, metrics, narrate, planner, retrieval
from engine.access import AccessScope
from engine.query import run_query
from engine.semantics import Layer
from engine.sql_guard import validate_sql
from engine.verify import ERROR, WARN, Finding, Verifier

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def app_functions(namespace, *names):
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def layer(con):
    return Layer(con)


@pytest.fixture
def turn(con, layer):
    records, executed = [], []

    def execute(*args, **kwargs):
        executed.append(args[1])
        return run_query(*args, **kwargs)

    namespace = app_functions({
        "con": con, "layer": layer, "planner": planner, "deadline": deadline,
        "time": time, "pd": pd, "ACCESS": AccessScope.demo("review-fixes"),
        "verifier": Verifier(con), "validate_sql": validate_sql, "run_query": execute,
        "_retrieval_bundle": lambda _question: None,
        "_plan_headline": narrate.answer_explanation,
        "_audit_turn": lambda entry, **kwargs: records.append(dict(entry)),
        "GUARD_BLOCK_PREFIX": "Blocked by SQL guard",
    }, "_plan_turn", "_refuse_unverified_plan", "_verify_now", "_plan_trace_payload",
       "_timeout_turn")
    return namespace, records, executed


def test_wrong_governed_breakdown_is_blocked_before_execution(turn):
    namespace, records, executed = turn
    entry = namespace["_plan_turn"]("What is the claim denial rate by payer type?")
    assert entry["refused"] and entry["verification_failed"]
    assert "governed_metric_filter" in [f.check for f in entry["findings"]]
    assert "Pending" in entry["reason"]
    assert not executed and not entry["ran"]
    assert entry["rows"] is None and entry["answer"] == ""
    assert len(records) == 1 and records[0]["refused"]


def test_successful_keyless_answer_still_runs_and_is_audited(turn):
    namespace, records, executed = turn
    entry = namespace["_plan_turn"]("How many denied claims are there?")
    assert not entry["refused"] and entry["ran"]
    assert entry["rows"].iloc[0, 0] == 876
    assert "876" in entry["answer"]
    assert len(executed) == len(records) == 1


@pytest.mark.parametrize("severity, rejected", [(ERROR, True), (WARN, False)])
def test_result_checks_withhold_blocking_results_but_preserve_advisories(turn, severity, rejected):
    namespace, records, _ = turn
    namespace["verifier"].check_result = lambda *_args: [Finding("result_probe", severity, "Probe")]
    entry = namespace["_plan_turn"]("How many denied claims are there?")
    assert entry["ran"] and entry["refused"] is rejected
    assert (entry["rows"] is None) is rejected
    assert bool(entry["answer"]) is not rejected
    assert len(records) == 1


def test_rejected_old_transcript_cannot_render_or_speak_its_answer():
    calls = []

    def unexpected(*_args, **_kwargs):
        pytest.fail("Rejected answer reached prose, audio, or result renderer")

    surface = SimpleNamespace(
        chat_message=lambda _role: nullcontext() if _role == "assistant"
        else SimpleNamespace(write=lambda *_args: None), caption=lambda *_args: None)
    namespace = app_functions({
        "st": surface,
        "ui": SimpleNamespace(pipeline=lambda **kwargs: calls.append(kwargs),
                              refusal=lambda *_args, **_kwargs: None, answer=unexpected),
        "_render_answer_audio": unexpected, "_result_block": unexpected,
        "_verification_readout": lambda *_args, **_kwargs: None,
    }, "render_plan_entry")
    namespace["render_plan_entry"]({
        "question": "denial rate by payer", "refused": False, "ran": True,
        "guard_ok": True, "guard_ms": 1, "verify_ms": 1,
        "answer": "Commercial: 7.8%", "rows": [("Commercial", 7.8)],
        "findings": [Finding("governed_metric_filter", ERROR, "Wrong denominator")],
    }, 0)
    assert calls[0]["verified"] == "fail"
    assert calls[0]["executed"] is True


@pytest.mark.parametrize("window", [
    "today", "yesterday", "tomorrow", "this week", "last month", "past 30 days",
    "previous quarter", "seven days ago", "on 2026-09-08", "on 09/08/2026",
    "in September", "since 2024", "between 2024 and 2025", "year-to-date",
])
def test_unimplemented_date_windows_never_become_all_time_counts(layer, window):
    result = planner.plan_question(f"How many denied claims were there {window}?", layer)
    assert result.refused and result.kind == "unsupported date filter"
    assert not result.sql


def test_date_refusal_reaches_app_without_querying(turn):
    namespace, records, executed = turn
    entry = namespace["_plan_turn"]("How many denied claims were there today?")
    assert entry["refused"] and not executed and not entry["answer"]
    assert len(records) == 1


def test_existing_explicit_year_filter_is_preserved(layer, con):
    result = planner.plan_question("How many claims were submitted in 2024?", layer)
    assert not result.refused, result.reason
    assert 'EXTRACT(YEAR FROM "healthcare_fact_claims"."submitted_date") = 2024' in result.sql
    assert run_query(con, result.sql).ok


def test_plural_claims_still_selects_the_governed_rate(con):
    metric = metrics.match_metric("What is our claims denial rate?", metrics.load_metrics())
    assert metric is not None and metric.name == "denial_rate"
    assert metrics.answer(con, metric).value == 8.2
    assert metrics.match_metric("What is our claims denial rate today?", (metric,)) is None


@pytest.fixture
def rag():
    state = {"failed": True, "calls": 0, "now": 100.0}

    def hybrid(*_args, **_kwargs):
        state["calls"] += 1
        if state["failed"]:
            raise RuntimeError("Sensitive question and provider credential")
        return [retrieval.RetrievedTable("visible", "demo", "", .9)]

    surface = SimpleNamespace(session_state={}, cache_data=streamlit.cache_data)
    namespace = app_functions({
        "__name__": __name__, "st": surface, "logging": logging, "hashlib": hashlib,
        "time": SimpleNamespace(perf_counter=time.perf_counter, monotonic=lambda: state["now"]),
        "con": None, "POOL": 20,
        "ACCESS": SimpleNamespace(fingerprint=("scope-a",), allowed_tables={"visible"},
                                  denied_by_table={}),
        "retrieval": SimpleNamespace(
            corpus_revision=lambda _con: "schema-v1", retrieve_hybrid=hybrid,
            scoped_hits=lambda hits, **_kwargs: hits,
            retrieve=lambda *_args, **_kwargs: [], retrieve_keyword=lambda *_args, **_kwargs: [],
            schema_catalog_for=lambda *_args, **_kwargs: "catalog"),
    }, "_retrieval_bundle_cached", "_retrieval_bundle")
    namespace["_retrieval_bundle_cached"].clear()
    yield namespace, state, surface
    namespace["_retrieval_bundle_cached"].clear()


def test_same_question_recovers_after_transient_rag_failure(rag, caplog):
    namespace, state, _ = rag
    bundle = namespace["_retrieval_bundle"]
    assert bundle("same question") is None
    state["failed"] = False
    assert bundle("same question") is None  # bounded cooldown, not a retry loop
    assert state["calls"] == 1
    state["now"] += 6
    assert bundle("same question")["hits"][0].table == "visible"
    assert bundle("same question") is not None
    assert state["calls"] == 2  # successful retry is cached
    assert "RuntimeError" in caplog.text and "Sensitive" not in caplog.text


def test_retry_cooldown_is_scope_isolated_and_bounded(rag):
    namespace, state, surface = rag
    bundle = namespace["_retrieval_bundle"]
    for number in range(35):
        assert bundle(f"question {number}") is None
    assert len(surface.session_state["_retrieval_retry_after"]) == 32
    state["failed"] = False
    namespace["ACCESS"].fingerprint = ("scope-b",)
    assert bundle("question 34") is not None


def test_schema_probe_failure_also_recovers(rag):
    namespace, state, _ = rag
    probe = namespace["retrieval"].corpus_revision
    namespace["retrieval"].corpus_revision = lambda _con: (_ for _ in ()).throw(ValueError())
    assert namespace["_retrieval_bundle"]("same question") is None
    assert state["calls"] == 0
    namespace["retrieval"].corpus_revision = probe
    state.update(failed=False, now=106)
    assert namespace["_retrieval_bundle"]("same question") is not None
