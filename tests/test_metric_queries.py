"""Scoped policy definitions against independent SQL, access and app boundaries."""

import ast
import time
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from engine import deadline, metric_queries, metrics
from engine.access import AccessScope, Principal
from engine.limits import MAX_QUESTION_CHARS
from engine.query import run_query
from engine.sql_guard import validate_sql
from engine.verify import ERROR, Finding, Verifier

REGISTRY = metrics.load_metrics()
CASES = [
    ("by payer type", "p.payer_type", "", ""),
    ("by payer", "p.payer_name", "", ""),
    ("by provider specialty", "d.specialty", "", ""),
    ("for Medicare", "", "p.payer_type", "Medicare"),
    ("for specialty Cardiology", "", "d.specialty", "Cardiology"),
    ("by payer type for Cardiology", "p.payer_type", "d.specialty", "Cardiology"),
    ("for payer Blue Cross Blue Shield by specialty", "d.specialty", "p.payer_name",
     "Blue Cross Blue Shield"),
]


@pytest.mark.parametrize("phrase, formula, population", [
    ("claim denial rate", "ROUND(100.0 * SUM(CASE WHEN c.status='Denied' THEN 1 ELSE 0 END)"
     " / NULLIF(SUM(CASE WHEN c.status IN ('Paid','Denied') THEN 1 ELSE 0 END),0),1)", "1=1"),
    ("net collection rate", "ROUND(100.0 * SUM(c.paid_amount)"
     " / NULLIF(SUM(c.allowed_amount),0),1)", "c.status='Paid'"),
])
@pytest.mark.parametrize("suffix, group, filtered, value", CASES)
def test_scopes_match_independently_aggregated_claims(con, phrase, formula, population,
                                                    suffix, group, filtered, value):
    question = f"What is our {phrase} {suffix}?"
    metric = metric_queries.match_scoped_metric(question, REGISTRY, con)
    assert metric is not None and metric.scope
    result = metrics.answer(con, metric)
    assert result.ok, result.result.error
    selection = f"{group}, " if group else ""
    reference = f"""SELECT {selection}{formula} FROM healthcare_fact_claims c
        LEFT JOIN healthcare_dim_payer p ON c.payer_id=p.payer_id
        LEFT JOIN healthcare_dim_provider d ON c.provider_id=d.provider_id
        WHERE {population}"""
    if filtered:
        reference += f" AND {filtered} = '{value}'"
    if group:
        reference += f" GROUP BY {group} ORDER BY {group} NULLS LAST"
    assert result.result.rows == con.execute(reference).fetchall()
    assert not result.matches_contract, "a scoped result cannot certify the overall benchmark"
    assert not [f for f in Verifier(con).check_sql(metric.sql, question) if f.blocking]
    assert metric.scope in result.sentence


@pytest.mark.parametrize("question", [
    "claim denial rate today", "claim denial rate by month", "claim denial rate in 2024",
    "claim denial rate for Medicare today", "claim denial rate for Medicare and Medicaid",
    "claim denial rate by payer type by specialty", "claim denial rate for unknown payer",
    "top payer by claim denial rate", "claim denial rate and net collection rate",
    "claim denial rate for Medicare; DROP TABLE healthcare_fact_claims",
    "claim denial rate by payer type excluding pending", "claim denial rate for Cardiology in 2024",
])
def test_unsupported_qualifiers_are_refused_in_full(con, question):
    with pytest.raises(metric_queries.MetricScopeError):
        metric_queries.match_scoped_metric(question, REGISTRY, con)


def test_catalog_filter_respects_column_authorization(con):
    scope = AccessScope(Principal.demo("restricted"), frozenset({"healthcare_dim_payer"}),
                        denied_columns=(("healthcare_dim_payer", ("payer_type", "payer_name")),))
    with pytest.raises(metric_queries.MetricScopeError, match="authorized catalog"):
        metric_queries.match_scoped_metric("claim denial rate for Medicare", REGISTRY,
                                           con, access=scope)


def test_a_scoped_definition_cannot_read_around_fact_authorization(con):
    metric = metric_queries.match_scoped_metric("claim denial rate by payer type", REGISTRY, con)
    scope = AccessScope(Principal.demo("restricted"), frozenset({"healthcare_dim_payer"}))
    denied = metrics.answer(con, metric, access=scope)
    assert denied.result.policy_denied and not denied.ok
    assert denied.sentence == "—"
    aggregate_scope = AccessScope(
        Principal.demo("aggregate-only"),
        frozenset({"healthcare_fact_claims", "healthcare_dim_payer"}),
        denied_columns=(("healthcare_fact_claims", ("claim_id", "patient_id")),))
    assert metrics.answer(con, metric, access=aggregate_scope).ok


def test_duplicate_dimension_keys_cannot_multiply_claims():
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE healthcare_dim_payer AS SELECT * FROM "
                    "(VALUES (1,'Medicare'),(1,'Medicare')) t(payer_id,payer_type)")
        with pytest.raises(metric_queries.MetricScopeError, match="not verifiably unique"):
            metric_queries.match_scoped_metric("claim denial rate by payer type", REGISTRY, con)
    finally:
        con.close()


@pytest.fixture
def app_turn(con):
    records, executed = [], []

    def execute(*args, **kwargs):
        executed.append(args[1])
        return run_query(*args, **kwargs)

    namespace = {
        "con": con, "metric_registry": REGISTRY, "metric_layer": metrics,
        "metric_queries": metric_queries, "deadline": deadline, "time": time,
        "pd": pd, "ACCESS": AccessScope.demo("scoped-test"), "verifier": Verifier(con),
        "MAX_QUESTION_CHARS": MAX_QUESTION_CHARS, "validate_sql": validate_sql,
        "run_query": execute, "GUARD_BLOCK_PREFIX": "Blocked by SQL guard",
        "_audit_turn": lambda entry, **_: records.append(dict(entry)),
        "_plan_turn": lambda *_: pytest.fail("a governed question reached the raw compiler"),
    }
    path = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    names = {"_keyless_turn", "_metric_turn", "_verify_now", "_timeout_turn"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, records, executed


def test_app_routes_and_audits_the_scoped_answer(app_turn):
    namespace, records, executed = app_turn
    entry = namespace["_keyless_turn"](
        "What is the claim denial rate by payer type for Cardiology?")
    assert entry["engine"] == "metric" and entry["ran"] and not entry["refused"]
    assert len(entry["rows"]) == 5 and entry["rows"]["payer_type"].nunique() == 5
    assert "by payer type for Cardiology" in entry["answer"]
    assert not entry["contract_match"] and len(records) == len(executed) == 1


def test_app_unsupported_scope_never_executes_or_returns_an_answer(app_turn):
    namespace, records, executed = app_turn
    entry = namespace["_keyless_turn"]("claim denial rate by payer type today")
    assert entry["refused"] and not entry["ran"] and not entry["answer"]
    assert entry["rows"] is None and not executed and len(records) == 1


@pytest.mark.parametrize("after_execution", [False, True])
def test_certified_path_withholds_blocking_verification_findings(app_turn, after_execution):
    namespace, records, executed = app_turn

    def verify(_sql, result, _question):
        findings = [Finding("probe", ERROR, "Reject the result")]
        return (findings if (result is not None) == after_execution else []), 0

    namespace["_verify_now"] = verify
    entry = namespace["_keyless_turn"]("What is the claim denial rate?")
    assert entry["refused"] and entry["verification_failed"] and not entry["answer"]
    assert entry["rows"] is None and len(executed) == int(after_execution)
    assert len(records) == 1


def test_overall_app_contract_still_returns_8_point_2(app_turn):
    namespace, _, _ = app_turn
    entry = namespace["_keyless_turn"]("What is the overall claim denial rate?")
    assert entry["contract_match"] and entry["answer"] == "The claim denial rate is 8.2%."
