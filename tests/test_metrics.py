"""Certified metrics: policy-owned definitions, exact matching, live contracts."""

from pathlib import Path

import pytest

from engine import metrics

REGISTRY = metrics.load_metrics()


def test_registry_is_nonempty_unique_and_every_statement_is_read_only():
    assert len(REGISTRY) >= 5
    assert len({m.name for m in REGISTRY}) == len(REGISTRY)
    assert all(m.sql.lower().startswith(("select", "with")) for m in REGISTRY)
    assert all(m.owner and m.definition and m.derived_why for m in REGISTRY)


@pytest.mark.parametrize("metric", REGISTRY, ids=lambda m: m.name)
def test_every_committed_definition_matches_the_live_warehouse(con, metric):
    result = metrics.answer(con, metric)
    assert result.ok, result.result.error
    assert result.matches_contract, (
        f"{metric.name}: got {result.value!r}, expected {metric.expect!r}"
    )


def test_longest_exact_phrase_wins_with_conversational_wrappers():
    got = metrics.match_metric("Please tell me our overall claim denial rate", REGISTRY)
    assert got is not None and got.name == "denial_rate"

    got = metrics.match_metric("what is our voluntary attrition rate?", REGISTRY)
    assert got is not None and got.name == "voluntary_attrition_rate"


@pytest.mark.parametrize("question", [
    "denial rate by payer",
    "denial rate for Medicare",
    "monthly collection rate",
    "gross margin rate in Grocery",
    "top site by stale query share",
])
def test_a_fixed_overall_definition_never_swallows_a_qualifier(question):
    assert metrics.match_metric(question, REGISTRY) is None


def test_matching_is_not_fuzzy_and_does_not_guess_a_metric():
    assert metrics.match_metric("what is the denial percentage?", REGISTRY) is None
    assert metrics.match_metric("how many denied claims?", REGISTRY) is None


def test_contract_drift_is_detected_not_hidden(con):
    metric = REGISTRY[0]
    changed = metrics.Metric(**{**metric.__dict__, "expect": -999})
    assert not metrics.answer(con, changed).matches_contract


def test_invalid_registry_sql_is_rejected_at_load_time(tmp_path: Path):
    path = tmp_path / "metrics.yaml"
    path.write_text(
        """
- name: unsafe
  label: Unsafe
  domain: test
  owner: Nobody
  unit: rows
  definition: Not a real definition.
  phrases: [unsafe metric]
  sql: DELETE FROM healthcare_fact_claims
  expect: 0
  derived: {value: null, why: The compiler refuses.}
""",
        encoding="utf-8",
    )
    with pytest.raises(metrics.MetricRegistryError, match="unsafe SQL"):
        metrics.load_metrics(path)
