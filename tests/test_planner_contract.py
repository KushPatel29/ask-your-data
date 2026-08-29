"""
`evals/planner_questions.yaml` as a contract, not a report.

`scripts/run_planner_eval.py` prints a scorecard for a human. This makes the
same numbers fail a build, which is the only version of a claim this repo
counts. Three things are asserted, and they are separable on purpose:

  * the REFERENCE SQL still returns its baked value -- a data-drift check that
    has nothing to do with the planner;
  * the planner still compiles each question to a query returning the same rows
    as the reference -- the accuracy check;
  * the planner still REFUSES the questions marked unanswerable -- the safety
    check, asserted as firmly as any answer, because a planner that starts
    guessing at those has regressed even though its match count went up.

The one number that must never move is `differs`. A refusal costs a visitor an
answer; a disagreement costs them a wrong one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from engine.planner import plan_question
from engine.query import run_query
from engine.semantics import Layer

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "evals" / "planner_questions.yaml"


def _cases():
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


CASES = _cases()
ANSWERABLE = [c for c in CASES if not c.get("expect_refusal")]
REFUSALS = [c for c in CASES if c.get("expect_refusal")]


@pytest.fixture(scope="module")
def layer(con):
    return Layer(con)


def _ids(cases):
    return [c["id"] for c in cases]


def test_contract_is_well_formed():
    seen = set()
    for case in CASES:
        missing = {"id", "domain", "question"} - set(case)
        assert not missing, f"{case.get('id', '?')} missing {sorted(missing)}"
        assert case["id"] not in seen, f"duplicate id {case['id']}"
        seen.add(case["id"])
        if case.get("expect_refusal"):
            assert not case.get("sql", "").strip(), (
                f"{case['id']} is marked unanswerable but carries reference SQL")
        else:
            assert case["sql"].strip(), f"{case['id']} has no reference SQL"


def test_the_contract_still_contains_refusals():
    """An eval set trimmed to what already passes measures nothing."""
    assert REFUSALS, "the unanswerable questions were removed from the contract"


@pytest.mark.parametrize("case", ANSWERABLE, ids=_ids(ANSWERABLE))
def test_reference_sql_still_returns_its_baked_value(case, con):
    """Data drift, isolated from planner drift.

    If this goes red and the planner tests stay green, the vendored CSVs moved.
    If the planner tests go red and this stays green, the compiler moved. Two
    failure modes that need different fixes should not share one assertion.
    """
    ran = run_query(con, case["sql"])
    assert ran.ok, f"reference SQL failed: {ran.error}"
    assert ran.rows, "reference SQL returned no rows"
    value = ran.rows[0][0]
    expected = case["expect"]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert float(value) == pytest.approx(float(expected), rel=1e-5)
    else:
        assert str(value) == str(expected)
    if case.get("expect_rows") is not None:
        assert ran.row_count == case["expect_rows"]


@pytest.mark.parametrize("case", ANSWERABLE, ids=_ids(ANSWERABLE))
def test_planner_answers_the_contract(case, con, layer):
    """The planner's own SQL must return the reference's rows.

    Compared as multisets of the reference's columns, never first-cell against
    first-cell: a GROUP BY without an ORDER BY has no first row, and comparing
    one scored twelve correct breakdowns as wrong because DuckDB returned the
    groups in a different order.

    The reference's column count is the width compared. The planner may return
    more context alongside (the rank shape returns the measure it sorted on next
    to the name); it may not return less.
    """
    result = plan_question(case["question"], layer, retrieved=_hits(case["question"]))
    assert not result.refused, f"refused a contract question: {result.reason}"

    ran = run_query(con, result.sql)
    assert ran.ok, f"planner SQL failed: {ran.error}\n{result.sql}"
    ref = run_query(con, case["sql"])
    assert ref.ok

    width = len(ref.columns)
    assert ran.row_count == ref.row_count, (
        f"{ran.row_count} rows vs the reference's {ref.row_count}\n{result.sql}")
    assert all(len(row) >= width for row in ran.rows), (
        f"planner returned fewer columns than the reference\n{result.sql}")
    assert (sorted(_key(row[:width]) for row in ran.rows)
            == sorted(_key(row[:width]) for row in ref.rows)), (
        f"same shape, different values\n{result.sql}")


@pytest.mark.parametrize("case", REFUSALS, ids=_ids(REFUSALS))
def test_planner_refuses_what_it_cannot_bind(case, layer):
    result = plan_question(case["question"], layer, retrieved=_hits(case["question"]))
    assert result.refused, (
        f"answered a question the contract marks unanswerable: {result.sql}")
    assert result.reason, "a refusal has to say why"


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_every_planned_query_is_read_only(case, con, layer):
    """The guard runs on compiled SQL exactly as it runs on model SQL."""
    from engine.sql_guard import validate_sql

    result = plan_question(case["question"], layer, retrieved=_hits(case["question"]))
    if result.refused:
        return
    ok, reason = validate_sql(result.sql)
    assert ok, f"{case['id']} produced SQL the guard rejected: {reason}"


def _hits(question: str) -> list[str]:
    """Retrieval ranking, when the index is available.

    Retrieval is a ranking SIGNAL to the planner, never a requirement, so this
    degrades to an empty list rather than failing. In an environment without the
    embedding model the contract still runs -- which is the same argument
    `engine/retrieval.py` makes for being keyless in the first place.
    """
    try:
        from engine import retrieval

        return [hit.table for hit in retrieval.retrieve_hybrid(question)]
    except Exception:
        return []


def _key(row: tuple) -> tuple:
    out = []
    for value in row:
        if isinstance(value, bool) or value is None:
            out.append(repr(value))
        elif isinstance(value, (int, float)):
            out.append(f"{float(value):.6g}")
        else:
            out.append(str(value))
    return tuple(out)
