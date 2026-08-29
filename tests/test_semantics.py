"""
The semantic layer is inferred, so what it infers has to be pinned.

Every fact `engine/semantics.py` produces is probed from DuckDB rather than
written down, which is the point of it -- and also the risk. A hand-authored
mapping is wrong loudly, on the line where someone typed it. An inferred one is
wrong quietly, three modules downstream, when a planner sums a column that
turned out not to be a measure.

So these tests assert the inferences themselves, on the real warehouse, with the
specific columns whose classification the planner depends on.
"""

from __future__ import annotations

import pytest

from engine.semantics import (
    DATE,
    DIMENSION,
    FLAG,
    KEY,
    MAX_DIMENSION_CARDINALITY,
    MEASURE,
    Layer,
    normalise_phrase,
    split_identifier,
)


@pytest.fixture(scope="module")
def layer(con):
    """One layer for the whole module.

    Profiling probes every column's cardinality and every table's grain -- ~2.4 s
    over these 71 tables -- so building it per test would put a minute of pure
    setup into a suite that currently runs in thirty seconds.
    """
    return Layer(con)


# ---------------------------------------------------------------------------
# Identifier splitting -- the fix that moved retrieval recall to 100% and that
# the planner's whole word-matching layer sits on.
# ---------------------------------------------------------------------------

def test_split_identifier_breaks_snake_case():
    assert split_identifier("qty_shipped") == ["qty", "shipped"]
    assert split_identifier("net_collection_rate") == ["net", "collection", "rate"]


def test_split_identifier_breaks_camel_case_and_digits():
    assert split_identifier("R1_STRUCTURING") == ["r", "1", "structuring"]
    assert split_identifier("txnDateTime") == ["txn", "date", "time"]


def test_normalise_phrase_strips_punctuation_and_case():
    assert normalise_phrase("Self-Pay") == "self pay"
    assert normalise_phrase("  60+ days ") == "60 days"


# ---------------------------------------------------------------------------
# Role inference. These four assertions are the ones a regression would hurt.
# ---------------------------------------------------------------------------

def test_money_columns_are_measures_not_keys(layer):
    """The bug that made this file necessary.

    `approx_count_distinct` is a HyperLogLog estimate and it OVERSHOOTS: it
    reported 12,801 distinct `paid_amount` over 12,000 claims, so uniqueness
    inference promoted the single most important measure in the warehouse to a
    key and the planner could no longer answer "total paid amount" at all.
    """
    claims = layer.tables["healthcare_fact_claims"]
    for column in ("submitted_amount", "allowed_amount", "paid_amount"):
        assert claims.column(column).role == MEASURE, f"{column} must stay a measure"


def test_identifiers_are_keys_even_when_numeric(layer):
    claims = layer.tables["healthcare_fact_claims"]
    assert claims.column("claim_id").role == KEY
    assert claims.column("payer_id").role == KEY      # BIGINT, only 9 distinct


def test_low_cardinality_text_is_a_dimension(layer):
    claims = layer.tables["healthcare_fact_claims"]
    status = claims.column("status")
    assert status.role == DIMENSION
    assert set(status.values) == {"Paid", "Denied", "Pending"}


def test_dates_and_flags_are_recognised(layer):
    assert layer.tables["healthcare_fact_claims"].column("service_date").role == DATE
    # BIGINT 0/1 named is_* -- a flag by name, since the type cannot say so.
    assert layer.tables["hr_fact_employees"].column("is_active").role == FLAG
    # A real BOOLEAN, for the other half of the rule.
    assert layer.tables["aml_cases"].column("used_mule_pool").role == FLAG


def test_ratio_columns_are_measures_but_not_additive(layer):
    """Summing a rate is nonsense; averaging one is not. The planner needs both."""
    ar = layer.tables["healthcare_ar_yield_predictions"]
    ncr = ar.column("net_collection_rate")
    assert ncr.role == MEASURE
    assert not ncr.additive
    assert ar.column("billed_amount").additive


def test_no_dimension_exceeds_the_cardinality_cap(layer):
    """Whatever is indexed as a dimension must be nameable in a question."""
    for table in layer.tables.values():
        for column in table.columns:
            if column.values:
                assert column.distinct <= MAX_DIMENSION_CARDINALITY, (
                    f"{table.name}.{column.name} has {column.distinct} values")


# ---------------------------------------------------------------------------
# Grain and joins
# ---------------------------------------------------------------------------

def test_grain_is_the_primary_key_where_one_exists(layer):
    assert layer.tables["healthcare_fact_claims"].grain == ("claim_id",)


def test_join_edges_never_cross_a_domain(layer):
    """engine/verify.py calls a cross-domain join an ERROR.

    A planner able to build one would be planning a query the verifier exists to
    block, so the edge simply must not be in the graph.
    """
    for edge in layer.edges:
        assert layer.tables[edge.left].domain == layer.tables[edge.right].domain, edge


def test_join_path_finds_the_fact_to_dimension_hop(layer):
    path = layer.join_path("healthcare_fact_claims", "healthcare_dim_payer")
    assert path is not None
    assert len(path) == 1
    assert path[0].column == "payer_id"


def test_join_path_refuses_to_cross_domains(layer):
    assert layer.join_path("healthcare_fact_claims", "hr_fact_employees") is None


# ---------------------------------------------------------------------------
# The value lexicon -- what lets "denied claims" become WHERE status = 'Denied'
# ---------------------------------------------------------------------------

def test_values_are_indexed_by_normalised_phrase(layer):
    bindings = layer.values_for_phrase("denied")
    assert bindings, "the word 'denied' must reach healthcare_fact_claims.status"
    assert any(b.table == "healthcare_fact_claims" and b.column == "status"
               and b.value == "Denied" for b in bindings)


def test_multiword_values_are_indexed_whole(layer):
    bindings = layer.values_for_phrase("self pay")
    assert any(b.column == "payer_type" and b.value == "Self-Pay" for b in bindings)


def test_high_cardinality_identifiers_are_not_in_the_lexicon(layer):
    """A hundred thousand transaction ids in a phrase table would be a bug."""
    transactions = layer.tables["aml_fact_transactions"]
    assert transactions.column("transaction_id").values == ()


def test_literal_escaping_survives_a_quote(layer):
    """No value in this warehouse has an apostrophe today. The escape still runs."""
    from engine.semantics import ValueBinding

    binding = ValueBinding(phrase="x", table="t", column="c",
                           value="O'Brien", type="VARCHAR")
    assert binding.literal == "'O''Brien'"


def test_summary_reports_the_shape_of_the_warehouse(layer):
    summary = layer.summary()
    assert summary["tables"] == len(layer.tables)
    assert summary["domains"] == 11
    assert summary["joins"] > 0
    assert summary["value_phrases"] > 100
