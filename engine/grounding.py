"""
Does the sentence the model wrote actually come from the rows the query returned?

THE GAP THIS CLOSES
This project's one rule is "no number without a query". The compiler path keeps
it structurally: `engine/narrate.py` composes its sentence FROM the plan and the
result, so it has no way to say a number the database did not return.

The model path did not. `Assistant._summarize` asks a language model to write
one or two sentences "using ONLY the SQL result provided" — and then trusted it.
The instruction is good and models mostly follow it, but "mostly" is the whole
problem: the SQL can return 12,000, the prose can say 999, and every layer
downstream reports a successful turn. Nothing in the guard, the verifier or the
executor looks at the prose, because none of them are about the prose.

So this module reads the sentence back against the rows.

WHAT IT CAN AND CANNOT PROVE, STATED PLAINLY
It checks NUMBERS, because a number is the part of an analytics answer that is
both checkable and load-bearing. It cannot check that the sentence describes the
right subject, and it does not try — a summary that says "employees" where the
query counted claims will pass. A narrower control that holds is worth more than
a broad one that does not, and the limit belongs in the docstring rather than in
a reader's assumptions.

SMALL INTEGERS ARE DELIBERATELY EXEMPT
English prose contains integers that are not claims about data: "one or two
sentences", "the top 3", "both". Flagging those would make the check fire on
good answers, and a control that cries wolf gets turned off. Integers 0-12 are
therefore allowed through unless they are the ONLY number in the sentence — at
which point the sentence is making a numeric claim and has to source it.

That exemption is a real hole and it is bounded: a model that hallucinates "5"
where the answer was 7 is not caught. A model that hallucinates 999, 12.4% or
1,661,141 is. The large, specific, quotable numbers — the ones that end up in a
board deck — are the ones this catches.
"""

from __future__ import annotations

import re

# Matches 1,234.56 / -12 / 0.5 / 1234. Deliberately not scientific notation:
# nothing in this warehouse renders that way, and admitting it would widen the
# token space for no gain.
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Integers a sentence can contain without making a claim about the data.
SMALL_INTEGER_CEILING = 12


def _forms(value) -> set[str]:
    """Every way one number could reasonably be written into a sentence.

    A model that reads 8.23 off the result and writes "8.2%" has not invented
    anything, so rounding to fewer decimals is accepted. Rounding to MORE
    precision than the result carries is not generated here, because that would
    be the model adding significant figures the database never produced.
    """
    out: set[str] = set()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return out
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return out

    candidates = [number]
    for places in (0, 1, 2, 3):
        candidates.append(round(number, places))
    for candidate in candidates:
        if candidate == int(candidate):
            out.add(str(int(candidate)))
            out.add(f"{int(candidate):,}")
        text = repr(float(candidate))
        out.add(text)
        for places in (1, 2, 3):
            out.add(f"{candidate:.{places}f}")
            out.add(f"{candidate:,.{places}f}")
    return {t.rstrip(".") for t in out}


def _canonical(token: str) -> str:
    return token.replace(",", "")


def result_numbers(result) -> set[str]:
    """Every number the result really contains, in the forms a writer might use.

    The row count is included because "5 departments" is a true statement about
    a result with five rows, and the sentence has no other way to say it.
    """
    out: set[str] = set()
    for row in getattr(result, "rows", None) or []:
        for value in row:
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                out |= _forms(value)
            else:
                for match in NUMBER_RE.finditer(str(value)):
                    out |= _forms(_canonical(match.group()))
    out |= _forms(getattr(result, "row_count", 0) or 0)
    return {_canonical(t) for t in out}


def ungrounded_numbers(text: str, result) -> list[str]:
    """Numbers in `text` that the result cannot account for.

    An empty list means every number in the sentence traces to a returned value
    (or to the row count). It does NOT mean the sentence is correct — see the
    module docstring.
    """
    allowed = result_numbers(result)
    tokens = [m.group() for m in NUMBER_RE.finditer(str(text or ""))]
    unexplained: list[str] = []
    for token in tokens:
        canonical = _canonical(token)
        if canonical in allowed:
            continue
        # A small bare integer is prose unless it is the sentence's only number,
        # in which case the sentence is answering with it.
        try:
            as_float = float(canonical)
        except ValueError:
            continue
        if (len(tokens) > 1 and as_float == int(as_float)
                and abs(as_float) <= SMALL_INTEGER_CEILING):
            continue
        unexplained.append(token)
    return unexplained


def is_grounded(text: str, result) -> bool:
    """A non-empty sentence whose every number came from the result."""
    if not str(text or "").strip():
        return False
    return not ungrounded_numbers(text, result)


def fallback_answer(result) -> str:
    """A deterministic sentence for when the model's prose cannot be trusted.

    Deliberately plain. This runs precisely when the interesting sentence has
    been rejected, and the honest thing left to say is what came back — not a
    second attempt at fluency from the same source that just failed.
    """
    rows = getattr(result, "rows", None) or []
    if not rows:
        return "The query returned no rows."
    if len(rows) == 1 and len(rows[0]) == 1:
        value = rows[0][0]
        if isinstance(value, float) and value == int(value):
            value = int(value)
        rendered = f"{value:,}" if isinstance(value, (int, float)) else str(value)
        return f"The query returned {rendered}."
    count = getattr(result, "row_count", len(rows))
    noun = "row" if count == 1 else "rows"
    return f"The query returned {count:,} {noun}; the full result is below."
