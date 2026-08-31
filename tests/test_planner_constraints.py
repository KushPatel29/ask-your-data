"""
Round four: the CONSTRAINT, after three earlier rounds on wording and shape.

Rounds one and two varied the PHRASING of a question. Round three varied its
SHAPE — negation, follow-ups, ratios, two things at once. This round varied the
CONSTRAINT: ranges, zero, sets, two-sided negation, and shares of a measure
rather than of rows. It found twelve more silent wrong answers.

Every one of them returned a plausible business number. Not one crashed. The
largest class is also the quietest: a predicate the question clearly asked for
was dropped, and the query answered the UNCONSTRAINED question instead. There
is nothing in a result set that says a filter went missing — "how many claims
with allowed amount between 1000 and 2000" came back 12,000, which is every
claim in the warehouse, and looks exactly like an answer.

Each test is named after what the compiler did wrong.
"""

from __future__ import annotations

import pytest

from engine.planner import detect_limit, plan_question
from engine.query import run_query
from engine.semantics import Layer


@pytest.fixture(scope="module")
def layer(con):
    return Layer(con)


def ask(question, layer, **kw):
    return plan_question(question, layer, **kw)


# ---------------------------------------------------------------------------
# Ranges and numerals
# ---------------------------------------------------------------------------

def test_a_range_is_read_rather_than_dropped(layer, con):
    """COMPARE_RE had no BETWEEN alternative, so no predicate was emitted and
    the count came back 12,000 — every claim — where the answer is 1,697."""
    result = ask("How many claims with allowed amount between 1000 and 2000?",
                 layer, retrieved=["healthcare_fact_claims"])
    assert not result.refused, result.reason
    assert ">=" in result.sql and "<=" in result.sql
    assert run_query(con, result.sql).rows[0][0] == 1697


def test_the_same_defect_on_a_second_table(layer, con):
    """One instance of a dropped filter reads as a one-off. Two read as the
    class of bug it is: this one answered 1,900, the entire headcount."""
    result = ask("How many employees with base salary between 50000 and 60000?",
                 layer, retrieved=["hr_fact_employees"])
    assert not result.refused, result.reason
    assert run_query(con, result.sql).rows[0][0] == 247


def test_a_top_n_numeral_is_not_mistaken_for_a_dropped_filter(layer, con):
    """The guard on the numeral backstop, and it caught a real regression.

    `content_words` drops any one-character token, so reading consumption off
    `explained` made "top 3 departments" look like a dropped constraint and
    refused a question the plan handles perfectly. Consumption is read off the
    filter EVIDENCE instead.
    """
    result = ask("Top 3 departments by headcount", layer,
                 retrieved=["hr_flight_risk_scores"])
    assert not result.refused, result.reason
    assert len(run_query(con, result.sql).rows) == 3


def test_a_number_written_as_a_word_is_still_a_limit(layer, con):
    """"top five" parsed as no limit and fell through to the "which X has the
    most Y" default of LIMIT 1 — one row for a question asking for five."""
    assert detect_limit("top five departments by revenue") == 5
    result = ask("top five departments by revenue", layer,
                 retrieved=["wholesale_fact_department_month"])
    assert not result.refused, result.reason
    assert len(run_query(con, result.sql).rows) == 5


# ---------------------------------------------------------------------------
# Zero, and the word that was really a column name
# ---------------------------------------------------------------------------

def test_zero_is_a_comparison_this_grammar_can_spell(layer, con):
    """"claims with paid amount of zero" answered 9,746: it dropped the zero
    and bound `status = 'Paid'` off the word `paid` instead."""
    result = ask("How many claims with paid amount of zero?", layer,
                 retrieved=["healthcare_fact_claims"])
    assert not result.refused, result.reason
    assert run_query(con, result.sql).rows[0][0] == 876


def test_a_word_the_question_spent_on_a_column_is_not_a_value(layer):
    """`paid` in "paid amount" names half of `paid_amount`. Read one word at a
    time it is also a value of `status`, and the plan used it as both."""
    result = ask("How many claims with paid amount of zero?", layer,
                 retrieved=["healthcare_fact_claims"])
    assert "paid_amount" in result.sql
    assert "status" not in result.sql


def test_the_column_fragment_rule_is_scoped_to_the_tables_in_play(layer):
    """Why that rule is scoped, rather than checked warehouse-wide.

    Checked against every table, "the overall claim DENIAL RATE" matched a
    `denial_rate` column somewhere else entirely, excused both words, and the
    share filter on `status = 'Denied'` never bound — turning a question the
    compiler answers into a refusal by way of a column the query was never
    going to touch.
    """
    result = ask("What is the overall claim denial rate?", layer,
                 retrieved=["healthcare_fact_claims"])
    assert not result.refused, result.reason
    assert "'Denied'" in result.sql


def test_a_word_that_names_the_table_is_not_a_value_in_it(layer, con):
    """`clinical_query_log` carries "query" in its own name AND as a severity
    value, so "how many queries are open?" added severity = 'query' on top of
    the status filter and answered 9 where the truth is 16."""
    result = ask("How many queries are open?", layer,
                 retrieved=["clinical_query_log"])
    assert not result.refused, result.reason
    assert "severity" not in result.sql
    assert run_query(con, result.sql).rows[0][0] == 16


def test_a_value_that_is_not_in_the_warehouse_is_refused(layer):
    """There is no Legal department here. The plan bound the word `department`
    as a VALUE of `hr_flight_risk_scores.top_reason` — which really does hold
    the literal string "department" — and answered 56, a real count of an
    unrelated population, for a department that does not exist."""
    result = ask("How many employees in the Legal department?", layer,
                 retrieved=["hr_fact_employees", "hr_flight_risk_scores"])
    assert result.refused


# ---------------------------------------------------------------------------
# Sets, shares and negation
# ---------------------------------------------------------------------------

def test_a_second_value_on_one_column_is_a_set_not_a_duplicate(layer, con):
    """"revenue for Electronics and Grocery" bound Electronics, skipped Grocery
    as an already-filtered column, and answered with Electronics alone —
    41,280,624 of 65,113,503. Half a question, presented as all of it."""
    result = ask("total revenue for Electronics and Grocery", layer,
                 retrieved=["wholesale_fact_department_month"])
    assert not result.refused, result.reason
    assert " IN (" in result.sql
    assert run_query(con, result.sql).rows[0][0] == pytest.approx(65113503.5, rel=1e-6)


def test_a_share_of_a_measure_is_not_a_share_of_rows(layer, con):
    """10.0% is one department out of ten in a month-by-department fact.
    Electronics is 23.4% of the money, which is what was asked."""
    result = ask("What percent of revenue is Electronics?", layer,
                 retrieved=["wholesale_fact_department_month"])
    assert not result.refused, result.reason
    assert "SUM(" in result.sql and "FILTER" in result.sql
    assert run_query(con, result.sql).rows[0][0] == pytest.approx(23.4)


def test_a_share_of_the_rows_stays_a_share_of_the_rows(layer):
    """The guard on the rule above, and it caught a real regression.

    "What percentage of transactions are cash?" also has a measure in scope —
    the synonym map binds `transactions` to `txn_count_7d`, a rolling-window
    feature — so letting a scored measure decide turned a row share into a
    share of a windowed count and moved a contract answer to 7.7. The
    numerator is chosen by what the question names after "of", and a subject
    that is one of the table's own name-words is the population.
    """
    result = ask("What percentage of transactions are cash?", layer,
                 retrieved=["aml_fact_transactions"])
    assert not result.refused, result.reason
    assert "COUNT(*)" in result.sql
    assert "SUM(" not in result.sql


def test_a_two_sided_negation_is_refused(layer):
    """"neither denied nor paid" read only its first half and answered 876 —
    the count of denied claims — for a question that excludes them."""
    result = ask("How many claims are neither denied nor paid?", layer,
                 retrieved=["healthcare_fact_claims"])
    assert result.refused
    assert result.kind == "outside the grammar"


# ---------------------------------------------------------------------------
# Grain: the axis that shared the noun but not the level
# ---------------------------------------------------------------------------

def test_an_axis_must_be_the_level_that_was_asked_for(layer):
    """"top 5 stores by revenue" grouped by `store_format` and returned three
    warehouse CATEGORIES for a question about stores."""
    result = ask("top 5 stores by revenue", layer,
                 retrieved=["wholesale_fact_segment_month"])
    assert result.refused or "store_format" not in result.sql


def test_the_level_rule_does_not_refuse_an_axis_the_question_named(layer):
    """Its guard: a question that says the attribute word wants the attribute,
    and the rule must not take that away."""
    result = ask("Which store format has the highest revenue?", layer,
                 retrieved=["wholesale_fact_segment_month"])
    assert not result.refused, result.reason
    assert "store_format" in result.sql


def test_a_boolean_is_not_an_axis_just_because_it_shares_a_noun(layer):
    """`is_physical_store` shares the word `store`, so "which store has the
    highest revenue?" grouped by whether a store is physical — two groups —
    and answered `True`. A flag may be the axis only when the question said the
    word that DISTINGUISHES it."""
    result = ask("which store has the highest revenue", layer,
                 retrieved=["wholesale_fact_segment_month", "wholesale_dim_store"])
    assert "is_physical_store" not in result.sql


# ---------------------------------------------------------------------------
# Aggregate vocabulary
# ---------------------------------------------------------------------------

def test_a_multi_word_aggregate_phrase_does_not_excuse_its_words_alone(layer):
    """`overall value` is a way of saying SUM, so `value` joined the aggregate
    vocabulary and excused ITSELF — and "what is the average order value?"
    bound AVG(orders) on the word `order` and reported 110, the mean of a
    daily order COUNT, as an average order value."""
    result = ask("what is the average order value?", layer,
                 retrieved=["dbt_kpi_daily"])
    assert result.refused


def test_a_scope_word_is_still_free(layer):
    """The guard on that fix. "Overall" says do not group — an instruction the
    plan follows by not grouping — so holding the plan accountable for it is
    charging it for obeying."""
    result = ask("What is the overall claim denial rate?", layer,
                 retrieved=["healthcare_fact_claims"])
    assert not result.refused, result.reason
