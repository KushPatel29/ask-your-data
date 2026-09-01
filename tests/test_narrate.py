"""Natural-language answers remain grounded in the compiled plan and rows.

The keyless compiler is the public app's default engine. Its scalar fallback
used to return only a formatted cell (``876``), so correctness tests for the
SQL all passed while the product still sounded like a database console. These
tests pin the answer shapes a person actually sees without involving a model.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from engine import narrate, planner
from engine.query import QueryResult, run_query
from engine.semantics import DIMENSION, MEASURE, Column, JoinEdge, Layer


@pytest.fixture(scope="module")
def layer(con):
    return Layer(con)


def answer(question: str, con, layer: Layer) -> str:
    planned = planner.plan_question(question, layer)
    assert planned.ok, planned.reason
    ran = run_query(con, planned.sql)
    assert ran.ok, ran.error
    return narrate.answer_sentence(planned.plan, ran)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("how many denied claims are there?",
         "There are 876 denied claims."),
        ("how many claims are not denied?",
         "There are 11,124 claims that are not denied."),
        ("how many distinct customers are there?",
         "There are 120 distinct customers."),
        ("what percentage of claims are denied?",
         "7.3% of claims are denied."),
        ("what is the total paid amount for claims?",
         "The total paid amount across all claims is 12,816,316.45."),
    ],
)
def test_scalar_answers_are_complete_sentences(question, expected, con, layer):
    assert answer(question, con, layer) == expected


def test_a_top_n_never_claims_the_limit_is_the_population(con, layer):
    spoken = answer("top 5 departments by headcount", con, layer)
    assert spoken == (
        "Of the 5 departments shown, Human Resources has the most employees "
        "(139) and Field Sales the fewest (134)."
    )
    assert "all 5" not in spoken and "of 5 departments" not in spoken


def test_an_unordered_breakdown_names_its_range_and_pluralises_status(con, layer):
    spoken = answer("how many claims by status?", con, layer)
    assert spoken == (
        "The number of claims across all 3 statuses ranges from Denied at 876 "
        "to Paid at 9,746."
    )


def test_a_null_group_does_not_erase_the_useful_groups(con, layer):
    spoken = answer("average allowed amount by claim status", con, layer)
    assert "Denied at 0.00 to Paid at 1,557.28" in spoken
    assert spoken.endswith("Pending has no value.")


def test_a_stored_ratio_is_not_mislabeled_as_a_percentage(con, layer):
    """0.21 is a ratio. Saying 0.2% would be a hundred-fold display error."""
    spoken = answer("Which payer type has the lowest net collection rate?", con, layer)
    assert spoken == "Self-Pay has the lowest average net collection rate, 0.21."
    assert "%" not in spoken


def test_an_explicit_pct_column_does_carry_its_unit():
    plan = planner.Plan(
        base="wholesale_fact_orders",
        aggregate=planner.AVG,
        measure=Column("wholesale_fact_orders", "fill_rate_pct", "DOUBLE", MEASURE),
    )
    ran = QueryResult(sql="", rows=[(91.25,)], row_count=1)
    assert narrate.answer_sentence(plan, ran) == (
        "The average fill rate across all orders is 91.2%."
    )


def test_a_compound_dimension_pluralises_its_head_noun():
    plan = planner.Plan(
        base="aml_fact_transactions",
        aggregate=planner.COUNT,
        group_by=Column("aml_fact_transactions", "hour_of_day", "INTEGER", DIMENSION),
    )
    ran = QueryResult(sql="", rows=[(0, 8), (1, 12)], row_count=2)
    spoken = narrate.answer_sentence(plan, ran)
    assert "2 hours of day" in spoken
    assert "hour of days" not in spoken


def test_one_valued_group_explains_the_missing_values():
    plan = planner.Plan(
        base="hr_fact_employees",
        aggregate=planner.AVG,
        measure=Column("hr_fact_employees", "base_salary", "DOUBLE", MEASURE),
        group_by=Column("hr_fact_employees", "department", "VARCHAR", DIMENSION),
    )
    ran = QueryResult(sql="", rows=[("Finance", None), ("Sales", 42.0)], row_count=2)
    assert narrate.answer_sentence(plan, ran) == (
        "Only Sales has a value for average base salary: 42.00. "
        "The other 1 department has no value."
    )


def test_decimal_results_are_formatted_as_numbers_not_opaque_strings():
    plan = planner.Plan(
        base="retail_fact_orders",
        aggregate=planner.SUM,
        measure=Column("retail_fact_orders", "revenue", "DECIMAL", MEASURE),
    )
    ran = QueryResult(sql="", rows=[(Decimal("12345.60"),)], row_count=1)
    assert narrate.answer_sentence(plan, ran) == (
        "The total revenue across all orders is 12,345.60."
    )


def test_a_null_scalar_distinguishes_missing_values_from_zero_rows():
    plan = planner.Plan(
        base="healthcare_fact_claims",
        aggregate=planner.AVG,
        measure=Column("healthcare_fact_claims", "allowed_amount", "DOUBLE", MEASURE),
    )
    ran = QueryResult(sql="", rows=[(None,)], row_count=1)
    assert narrate.answer_sentence(plan, ran) == (
        "The average allowed amount across all claims is unavailable — no "
        "matching row has a value for allowed amount."
    )


def test_a_fanout_count_calls_its_units_rows_not_entities():
    plan = planner.Plan(
        base="hr_dim_employee",
        aggregate=planner.COUNT,
        joins=[JoinEdge("hr_dim_employee", "hr_fact_events", "employee_id", "left")],
    )
    ran = QueryResult(sql="", rows=[(42,)], row_count=1)
    assert narrate.answer_sentence(plan, ran) == "There are 42 rows."


def test_an_empty_breakdown_says_what_could_not_be_broken_down():
    plan = planner.Plan(
        base="healthcare_fact_claims",
        aggregate=planner.COUNT,
        group_by=Column("healthcare_fact_claims", "status", "VARCHAR", DIMENSION),
    )
    ran = QueryResult(sql="", rows=[], row_count=0)
    assert narrate.answer_sentence(plan, ran) == (
        "No claims matched, so there is nothing to break down by status."
    )


def test_every_supported_contract_question_produces_a_sentence(con, layer):
    """Exercise narration over every plan shape the keyless contract accepts.

    Exact wording belongs to the focused tests above. This broader pass guards
    the more important product promise: no accepted question can fall back to
    a bare cell or crash while turning an otherwise valid result into prose.
    """
    contract = yaml.safe_load(
        (Path(__file__).parent.parent / "evals" / "planner_questions.yaml")
        .read_text(encoding="utf-8")
    )
    for case in contract:
        planned = planner.plan_question(case["question"], layer)
        if not planned.ok:
            continue
        ran = run_query(con, planned.sql)
        spoken = narrate.answer_sentence(planned.plan, ran)
        assert spoken and spoken[-1] in ".!?", case["id"]
        assert not re.fullmatch(r"[\d,.%\s-]+[.!]?", spoken), case["id"]
