"""Only returned facts can enter an answer, regardless of model response."""

from decimal import Decimal

import pytest

from engine import grounding
from engine.query import QueryResult


def result(rows, columns=("n",), truncated=False, sql="SELECT 1"):
    return QueryResult(sql=sql, columns=list(columns), rows=rows,
                       row_count=len(rows), truncated=truncated)


@pytest.mark.parametrize("text", [
    "There are 999 claims.",
    "There are 12,000 employees because productivity improved.",
    "There is 1 claim.",
    "There are twelve thousand employees.",
    "", "   ", "null", "[]", '{"rows": []}',
    '{"rows": [0], "answer": "Sales made 12000"}',
    '{"rows": [true]}', '{"rows": [0.0]}', '{"rows": ["0"]}',
    '{"rows": [-1]}', '{"rows": [1]}', '{"rows": [0, 0]}',
    '{"rows": [0]} trailing prose',
])
def test_model_claims_and_invalid_selections_are_never_accepted(text):
    ran = result([(12000,)])
    assert grounding.parse_selection(text, ran) is None
    answer = grounding.compose_answer(ran)
    assert "12,000" in answer
    assert "999" not in answer and "employees" not in answer


def test_only_existing_visible_rows_can_be_selected_and_order_is_preserved():
    ran = result([(index,) for index in range(40)])
    assert grounding.parse_selection('{"rows": [2, 0]}', ran) == [0, 2]
    assert grounding.parse_selection('{"rows": [29]}', ran) == [29]
    assert grounding.parse_selection('{"rows": [30]}', ran) is None
    assert grounding.parse_selection('{"rows": [0, 1, 2, 3]}', ran) is None
    assert grounding.parse_selection('{"rows": []}', result([])) == []


def test_count_subject_comes_from_the_query_not_a_model_alias():
    ran = result([(12000,)], ("employees",),
                 sql="SELECT COUNT(*) AS employees FROM healthcare_fact_claims")
    assert grounding.compose_answer(ran) == "There are 12,000 claims in the loaded dataset."


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM healthcare_fact_claims WHERE status = 'Denied'",
    "SELECT COUNT(*) FROM healthcare_fact_claims JOIN hr_fact_employees ON true",
    "SELECT COUNT(*) FROM (SELECT * FROM healthcare_fact_claims LIMIT 2)",
])
def test_filtered_or_joined_count_does_not_claim_to_count_every_entity(sql):
    answer = grounding.compose_answer(result([(7,)], sql=sql))
    assert "7" in answer and "loaded dataset" not in answer
    assert "claims" not in answer


def test_named_values_are_useful_without_inventing_a_unit_or_scale():
    answer = grounding.compose_answer(result([(Decimal("0.21"),)], ("net_collection_rate",)))
    assert "net collection rate" in answer and "0.21" in answer
    assert "%" not in answer and "$" not in answer


def test_null_does_not_become_zero_or_no_matching_rows():
    answer = grounding.compose_answer(result([(None,)], ("average_allowed_amount",)))
    assert "did not return a value" in answer
    assert "different from zero" in answer
    assert "No rows matched" not in answer


def test_large_numbers_and_decimal_precision_survive_without_float_conversion():
    assert "9,007,199,254,740,993" in grounding.compose_answer(result([(9007199254740993,)]))
    assert "9,007,199,254,740,993.25" in grounding.compose_answer(
        result([(Decimal("9007199254740993.25"),)]))
    assert "0.0000001" in grounding.compose_answer(result([(0.0000001,)]))


def test_boolean_and_text_identifiers_are_not_reinterpreted_as_numeric_claims():
    assert "true" in grounding.compose_answer(result([(True,)]))
    assert "SITE-104" in grounding.compose_answer(result([("SITE-104",)], ("site_id",)))


def test_selected_rows_keep_their_own_labels_and_values():
    ran = result([("Sales", 12000), ("Finance", 600), ("Legal", 900), ("HR", 100)],
                 ("department", "revenue"))
    answer = grounding.compose_answer(ran, [1, 2])
    assert "4 rows" in answer and "2 selected rows" in answer
    assert "100 for HR to 12,000 for Sales" in answer
    assert "department: Finance; revenue: 600" in answer
    assert "department: Legal; revenue: 900" in answer
    assert "Finance; revenue: 12,000" not in answer


def test_equal_values_and_missing_values_have_honest_context():
    ran = result([("A", 5), ("B", 5), ("C", None)], ("team", "count"))
    answer = grounding.compose_answer(ran)
    assert "same throughout at 5" in answer
    assert "1 returned row has no usable numeric value" in answer
    assert "highest" not in answer


def test_duplicate_labels_do_not_create_an_ambiguous_comparison():
    ran = result([("Sales", 12000), ("Sales", 600)], ("department", "revenue"))
    answer = grounding.compose_answer(ran)
    assert "ranges from" not in answer
    assert "12,000" in answer and "600" in answer


def test_preview_cannot_be_presented_as_the_entire_population():
    ran = result([("A", 1), ("B", 8)], ("team", "count"), truncated=True)
    answer = grounding.compose_answer(ran)
    assert "Among the returned rows" in answer
    assert "Only the first 2 rows" in answer
    assert "not the full dataset" in answer


def test_empty_or_failed_query_does_not_gain_a_successful_narrative():
    assert "No rows matched" in grounding.compose_answer(result([]))
    failed = result([(999,)])
    failed.error = "query failed"
    answer = grounding.compose_answer(failed)
    assert "no verified result" in answer and "999" not in answer


def test_result_markup_stays_inert_and_long_cells_are_explicitly_shortened():
    answer = grounding.compose_answer(
        result([("![open](https://example.org) <script>x</script>",)]))
    assert r"\!\[open\]" in answer and r"\<script\>" in answer
    assert "value shortened" in grounding.compose_answer(result([("x" * 1000,)]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), Decimal("NaN")])
def test_non_finite_values_are_reported_as_unusable(value):
    assert "not a usable numeric result" in grounding.compose_answer(result([(value,)]))
