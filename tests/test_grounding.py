"""The model's PROSE, read back against the rows.

Every other boundary in this app checks something other than the sentence. The
guard checks SQL, the verifier checks structure, the executor checks the
database — and the summary, which is the part a reader actually quotes, was
covered by a prompt instruction and nothing else.

The reproduction that motivated this: SQL returns 12,000, the model writes 999,
and the turn is reported as successful.
"""

from __future__ import annotations

import pytest

from engine import grounding
from engine.query import QueryResult


def result(rows, columns=("n",), truncated=False):
    return QueryResult(sql="SELECT 1", columns=list(columns), rows=rows,
                       row_count=len(rows), truncated=truncated)


# ---------------------------------------------------------------------------
# The case this exists for
# ---------------------------------------------------------------------------

def test_a_number_the_query_never_returned_is_caught():
    rows = result([(12000,)])
    assert grounding.ungrounded_numbers("There are 999 claims.", rows) == ["999"]
    assert not grounding.is_grounded("There are 999 claims.", rows)


def test_the_number_the_query_did_return_passes():
    rows = result([(12000,)])
    assert grounding.is_grounded("There are 12,000 claims.", rows)
    assert grounding.is_grounded("There are 12000 claims.", rows)


def test_an_empty_answer_is_not_grounded():
    """An empty summary used to pass straight through and render as a blank
    answer above a perfectly good SQL block."""
    assert not grounding.is_grounded("", result([(1,)]))
    assert not grounding.is_grounded("   ", result([(1,)]))


def test_a_sentence_with_no_numbers_at_all_is_grounded():
    """Plenty of true answers are categorical: "Self-Pay has the lowest rate."."""
    rows = result([("Self-Pay",)], columns=("payer_type",))
    assert grounding.is_grounded("Self-Pay has the lowest net collection rate.", rows)


# ---------------------------------------------------------------------------
# What must NOT fire, because a control that cries wolf gets switched off
# ---------------------------------------------------------------------------

def test_rounding_down_to_fewer_decimals_is_not_an_invention():
    """A model that reads 8.23 and writes 8.2% has added nothing."""
    rows = result([(8.23,)])
    assert grounding.is_grounded("The denial rate is 8.2%.", rows)
    assert grounding.is_grounded("The denial rate is 8%.", rows)


def test_adding_precision_the_result_never_had_is_caught():
    """The other direction is an invention: significant figures the database
    did not produce."""
    rows = result([(8.2,)])
    assert not grounding.is_grounded("The denial rate is 8.2456%.", rows)


def test_thousands_separators_are_the_same_number():
    rows = result([(1661141.0,)])
    assert grounding.is_grounded("About 1,661,141 is collectable.", rows)


def test_small_integers_in_prose_do_not_fire():
    """English contains integers that are not claims about data."""
    rows = result([("Electronics", 41280624.23), ("Grocery", 23832879.27)],
                  columns=("department", "revenue"))
    assert grounding.is_grounded(
        "The top 2 departments are Electronics at 41,280,624.23 and Grocery at "
        "23,832,879.27.", rows)


def test_a_lone_small_integer_still_has_to_be_sourced():
    """The exemption is for prose, not for the answer itself. If the sentence
    contains exactly one number, that number IS the claim."""
    rows = result([(1483,)])
    assert not grounding.is_grounded("There are 7 employees.", rows)


def test_the_row_count_is_a_legitimate_thing_to_cite():
    """"5 departments" is true of a five-row result and the sentence has no
    other way to say it."""
    rows = result([("a", 1.5), ("b", 2.5), ("c", 3.5), ("d", 4.5), ("e", 5.5)],
                  columns=("dept", "v"))
    assert grounding.is_grounded("Across all 5 departments the values vary.", rows)


def test_numbers_inside_string_values_count_as_returned():
    """`CLM-000008` and `SITE-104` are values, and a sentence naming one is
    quoting the result."""
    rows = result([("SITE-104",)], columns=("site_id",))
    assert grounding.is_grounded("SITE-104 generates the most queries.", rows)


# ---------------------------------------------------------------------------
# The replacement
# ---------------------------------------------------------------------------

def test_the_fallback_states_the_result_rather_than_retrying_fluency():
    assert grounding.fallback_answer(result([(876,)])) == "The query returned 876."
    assert grounding.fallback_answer(result([])) == "The query returned no rows."
    many = result([("a", 1), ("b", 2)], columns=("k", "v"))
    assert "2 rows" in grounding.fallback_answer(many)


def test_the_fallback_formats_a_whole_float_as_an_integer():
    assert grounding.fallback_answer(result([(1661141.0,)])) == \
        "The query returned 1,661,141."


# ---------------------------------------------------------------------------
# The wiring, asserted at source level: the check is worthless if nothing calls it
# ---------------------------------------------------------------------------

def test_the_assistant_actually_reads_its_own_summary_back():
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "engine" / "assistant.py"
              ).read_text(encoding="utf-8")
    assert "grounding.is_grounded(answer, result)" in source
    assert "grounding.fallback_answer(result)" in source
    assert "ungrounded_answer" in source, "the turn must carry a finding, not swap silently"


@pytest.mark.parametrize("text,rows,grounded", [
    ("No rows matched.", [], True),
    ("The total is 0.", [(0,)], True),
    ("The average is -4.52%.", [(-4.52,)], True),
    ("The average is -9.9%.", [(-4.52,)], False),
])
def test_edges(text, rows, grounded):
    assert grounding.is_grounded(text, result(rows)) is grounded
