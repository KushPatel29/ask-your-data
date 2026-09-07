"""Permissions apply before metadata reaches a prompt or a question picker."""

from __future__ import annotations

import pytest

from engine import demo_mode, exemplars, planner, retrieval
from engine.access import AccessScope, Principal, authorize_sql
from engine.semantics import Layer, get_layer


def scope_for(tables, *, denied=()):
    return AccessScope(
        principal=Principal("restricted", authenticated=True),
        allowed_tables=frozenset(tables),
        denied_columns=denied,
        policy_version="metadata-test",
    )


@pytest.fixture(scope="module")
def full_layer(con):
    return Layer(con)


def test_scoped_layer_excludes_table_words_values_and_join_paths(full_layer):
    scope = scope_for({"hr_fact_employees"})
    layer = full_layer.scoped(scope)

    assert set(layer.tables) == {"hr_fact_employees"}
    assert not layer.tables_with_word("claims")
    assert not layer.values_for_phrase("denied")
    assert not layer.edges
    assert layer.join_path("healthcare_fact_claims", "healthcare_fact_claims") is None
    assert layer.summary()["tables"] == 1
    assert layer.summary()["domains"] == 1


def test_scoped_layer_hides_masked_values_keys_and_join_edges(full_layer):
    scope = scope_for(
        {"healthcare_fact_claims", "healthcare_dim_payer"},
        denied=(("healthcare_fact_claims", ("status", "claim_id", "payer_id")),),
    )
    layer = full_layer.scoped(scope)
    table = layer.tables["healthcare_fact_claims"]

    assert table.column("status") is None
    assert table.column("claim_id") is None
    assert not table.grain
    assert table.description == ""
    assert not layer.values_for_phrase("denied")
    assert "healthcare_fact_claims" not in layer.tables_with_word("status")
    assert layer.join_path("healthcare_fact_claims", "healthcare_dim_payer") is None
    assert layer.values_for_phrase("self pay"), "allowed dimension values stay usable"


def test_scope_filtering_never_mutates_another_sessions_cached_layer(full_layer):
    before = full_layer.summary()
    scope = scope_for({"healthcare_fact_claims"})
    first = full_layer.scoped(scope)
    second = full_layer.scoped(scope)

    first.tables["healthcare_fact_claims"].columns.clear()
    first._value_index.clear()
    first._word_index.clear()

    assert second.tables["healthcare_fact_claims"].column("status") is not None
    assert second.values_for_phrase("denied")
    assert second.tables_with_word("claims")
    assert full_layer.summary() == before
    assert full_layer.tables["healthcare_fact_claims"].column("status") is not None


def test_no_grants_means_no_semantic_metadata(full_layer):
    layer = full_layer.scoped(scope_for(set()))
    assert not layer.tables
    assert not layer.value_phrases
    assert not layer.tables_with_word("employees")
    assert all(count == 0 for count in layer.summary().values())


def test_cached_layer_api_honors_each_callers_scope(con, full_layer, monkeypatch):
    monkeypatch.setattr("engine.semantics._LAYER", full_layer)
    monkeypatch.setattr("engine.semantics._LAYER_CON", con)

    assert get_layer(con) is full_layer
    assert set(get_layer(con, access=scope_for({"hr_fact_employees"})).tables) == {
        "hr_fact_employees",
    }
    assert "healthcare_fact_claims" in get_layer(con).tables


def test_planner_cannot_generate_hidden_table_or_salary_sql(con, full_layer):
    scope = scope_for(
        {"hr_fact_employees"},
        denied=(("hr_fact_employees", ("base_salary",)),),
    )
    layer = full_layer.scoped(scope)
    for question in ("how many denied claims are there?", "average base salary"):
        plan = planner.plan_question(question, layer)
        assert not plan.sql, plan.sql
    allowed = planner.plan_question("how many active employees do we have?", layer)
    assert allowed.sql
    assert authorize_sql(con, allowed.sql, scope).allowed


def test_contract_picker_filters_both_relations_and_masked_columns(con):
    scope = scope_for(
        {"hr_fact_employees"},
        denied=(("hr_fact_employees", ("base_salary",)),),
    )
    cases = demo_mode.visible_cases(con, access=scope)

    assert cases
    assert all(authorize_sql(con, case["sql"], scope).allowed for case in cases)
    assert all(case["domain"] == "hr" for case in cases)
    assert not any("base_salary" in case["sql"] for case in cases)
    assert demo_mode.visible_cases(con, access=scope_for(set())) == []
    assert demo_mode.visible_cases(con, []) == []
    assert len(demo_mode.visible_cases(con)) == len(demo_mode.load_golden_questions())


@pytest.fixture()
def exemplar_corpus(monkeypatch):
    cases = [
        {"id": "secret_table", "domain": "healthcare", "question": "Denied claims?",
         "sql": "SELECT COUNT(*) FROM healthcare_fact_claims", "key": "denied claims"},
        {"id": "secret_column", "domain": "hr", "question": "Average salary?",
         "sql": "SELECT AVG(base_salary) FROM hr_fact_employees", "key": "average salary"},
        {"id": "allowed", "domain": "hr", "question": "Employee count?",
         "sql": "SELECT COUNT(*) FROM hr_fact_employees", "key": "employee count"},
    ]

    class Index:
        @staticmethod
        def query(**kwargs):
            count = kwargs["n_results"]
            return {"ids": [[case["id"] for case in cases[:count]]],
                    "distances": [[0.1] * min(count, len(cases))]}

    monkeypatch.setattr(exemplars, "load_cases", lambda: cases)
    monkeypatch.setattr(exemplars, "build_index", lambda **kwargs: Index())
    return cases


@pytest.mark.parametrize("retrieved_tables", [None, {"hr_fact_employees"}])
def test_exemplars_never_send_hidden_table_or_masked_column_to_model(
    con, exemplar_corpus, retrieved_tables,
):
    scope = scope_for(
        {"hr_fact_employees"},
        denied=(("hr_fact_employees", ("base_salary",)),),
    )
    picks = exemplars.select_exemplars(
        "How large is our workforce?", k=1, retrieved_tables=retrieved_tables,
        access=scope, con=con,
    )
    assert [pick.id for pick in picks] == ["allowed"]
    block = exemplars.exemplar_block("Workforce size?", access=scope, con=con)
    assert "COUNT(*) FROM hr_fact_employees" in block
    assert "base_salary" not in block
    assert "healthcare" not in block


def test_scoped_exemplars_fail_closed_without_parser_connection(exemplar_corpus):
    scope = scope_for({"hr_fact_employees"})
    assert exemplars.select_exemplars("Workforce size?", access=scope) == []
    assert exemplars.exemplar_block("Workforce size?", access=scope) == ""


def test_exemplar_leave_one_out_still_applies_after_authorization(con, exemplar_corpus):
    scope = scope_for(
        {"hr_fact_employees"},
        denied=(("hr_fact_employees", ("base_salary",)),),
    )
    assert exemplars.select_exemplars("Employee count?", access=scope, con=con) == []
    assert exemplars.select_exemplars(
        "Workforce size?", access=scope, con=con, exclude_ids={"allowed"},
    ) == []


def test_retrieval_display_descriptions_do_not_reintroduce_masked_values():
    original = retrieval.RetrievedTable(
        "hr_fact_employees", "hr", "base_salary for Confidential Person", 0.9,
    )
    denied = retrieval.RetrievedTable("healthcare_fact_claims", "healthcare", "Claims", 0.8)
    hits = retrieval.scoped_hits(
        [original, denied], allowed_tables={"hr_fact_employees"},
        denied_columns={"hr_fact_employees": frozenset({"base_salary"})},
    )
    assert len(hits) == 1
    assert hits[0].description == ""
    assert original.description == "base_salary for Confidential Person"


@pytest.mark.parametrize("retrieval_available", [True, False])
def test_prompt_descriptions_obey_scope_even_when_retrieval_falls_back(
    con, monkeypatch, retrieval_available,
):
    hit = retrieval.RetrievedTable(
        "hr_fact_employees", "hr", "base_salary for Confidential Person", 0.9,
    )

    def picker(*args, **kwargs):
        if not retrieval_available:
            raise RuntimeError("offline index unavailable")
        return [hit]

    monkeypatch.setattr(retrieval, "retrieve_hybrid", picker)
    monkeypatch.setattr(retrieval, "MANIFEST", [
        ("hr", "fact_employees", "unused.csv", hit.description),
        ("hr", "flight_risk_scores", "unused.csv", "Confidential risk scores"),
    ])
    monkeypatch.setattr(retrieval, "DOMAINS", {"hr": "Confidential Person risk and salary"})
    block = retrieval.schema_catalog_for(
        "Workforce size?", con, allowed_tables={"hr_fact_employees"},
        denied_columns={"hr_fact_employees": frozenset({"base_salary"})},
    )
    assert "hr_fact_employees" in block
    assert "employee_id" in block
    assert "base_salary" not in block
    assert "Confidential" not in block
    assert "flight_risk_scores" not in block
