"""Compose analytics answers from returned cells, never from model prose.

The summarizer's contract is ``{"rows": [0, 2]}``: it may choose up to three
visible result rows to highlight. It cannot supply numbers, labels, explanations,
comparisons, units, or causal claims. Those are rendered locally from the actual
result. A malformed selection falls back to the first rows without losing a
successful query. This replaces a number-membership check, which could not catch
swapped entities, a correct number attached to the wrong measure, spelled-out
numbers, or invented explanations.

This proves provenance of the displayed facts, not that generated SQL correctly
interpreted the question. SQL verification and the visible query remain separate
controls. Comparisons are always scoped to the returned rows, including LIMITs
and truncated previews; result aliases are labels, not inferred business units.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation

MAX_INPUT_ROWS = 30
MAX_HIGHLIGHT_ROWS = 3
MAX_CELL_CHARS = 240


def parse_selection(text: str, result) -> list[int] | None:
    """Validate the complete selector response; extra model claims are rejected."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"rows"}:
        return None
    indices = payload["rows"]
    count = min(len(getattr(result, "rows", ()) or ()), MAX_INPUT_ROWS)
    if not isinstance(indices, list) or len(indices) > MAX_HIGHLIGHT_ROWS:
        return None
    if count and not indices:
        return None
    if any(type(index) is not int or not 0 <= index < count for index in indices):
        return None
    if len(set(indices)) != len(indices):
        return None
    # A model may select, but it must not silently reorder the query's ranking.
    return sorted(indices)


def _numeric(value) -> Decimal | None:
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _plain(value) -> str:
    """Keep cell content as inert, bounded text in Markdown and voice output."""
    text = " ".join(str(value).split())
    if len(text) > MAX_CELL_CHARS:
        text = text[:MAX_CELL_CHARS] + "… [value shortened]"
    # A result cell can contain Markdown links or HTML. It is data, and must
    # not become a navigation link, image, heading, or hidden HTML element.
    return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", text)


def _value(value) -> str:
    if value is None:
        return "no value"
    if isinstance(value, bool):
        return "true" if value else "false"
    number = _numeric(value)
    if number is not None:
        if number == number.to_integral_value():
            return f"{int(number):,}"
        # Preserve the returned precision, including DECIMAL. Passing through
        # float would corrupt IDs and large integer totals above 2**53.
        return f"{number:,f}"
    if isinstance(value, (int, float, Decimal)):
        return "a non-finite value (not a usable numeric result)"
    return _plain(value)


def _label(value) -> str:
    return _plain(str(value).replace("_", " "))


def _count_subject(result) -> str:
    """Name records only for an exact single-table COUNT with no joins/filters.

    This intentionally is not a SQL parser. Full matching rejects subqueries,
    WHERE, joins, quoted expressions, comments, and misleading SELECT aliases.
    Everything else uses the result's explicit labels.
    """
    identifier = r"[A-Za-z][A-Za-z0-9_]*"
    match = re.fullmatch(
        rf"\s*SELECT\s+COUNT\s*\(\s*\*\s*\)"
        rf"(?:\s+(?:AS\s+)?{identifier})?\s+FROM\s+({identifier})\s*;?\s*",
        str(getattr(result, "sql", "")), re.IGNORECASE,
    )
    if not match:
        return ""
    parts = match[1].lower().split("_")
    if len(parts) > 1:
        parts = parts[1:]
    parts = [word for word in parts if word not in {"fact", "dim", "fct", "tbl"}]
    noun = " ".join(parts)
    # Never guess an entity for a table like finance_monthly or clinical_log.
    return noun if noun.endswith("s") and not noun.endswith(("ss", "us")) else "rows"


def _scalar(result, value, column: str) -> str:
    label = _label(column)
    generic = label.lower() in {"", "n", "v", "value", "count star()", "count(*)"}
    if value is None:
        subject = "a value" if generic else f"a value for {label}"
        return (f"The query did not return {subject}. "
                "A missing value is different from zero.")
    count_subject = _count_subject(result)
    if count_subject and _numeric(value) is not None:
        if value == 1:
            # Avoid a speculative singularisation of arbitrary table names.
            return f"The {count_subject} table contains 1 record."
        return f"There are {_value(value)} {count_subject} in the loaded dataset."
    if generic:
        return f"The query returned {_value(value)}."
    return f"The returned {label} is {_value(value)}."


def _comparison(rows, columns) -> str:
    """One observed comparison, with labels and numbers bound to the same row."""
    if len(columns) != 2 or any(len(row) != 2 for row in rows):
        return ""
    if any(row[0] is None or not isinstance(row[0], str) for row in rows):
        return ""
    # Duplicate labels do not identify a row unambiguously.
    if len({row[0] for row in rows}) != len(rows):
        return ""
    valued = [(row, _numeric(row[1])) for row in rows if _numeric(row[1]) is not None]
    if len(valued) < 2:
        return ""
    low = min(valued, key=lambda item: item[1])
    high = max(valued, key=lambda item: item[1])
    measure = _label(columns[1])
    if low[1] == high[1]:
        sentence = (f"Among the returned rows with a numeric value, {measure} is "
                    f"the same throughout at {_value(low[0][1])}.")
    else:
        sentence = (f"Among the returned rows, {measure} ranges from "
                    f"{_value(low[0][1])} for {_plain(low[0][0])} to "
                    f"{_value(high[0][1])} for {_plain(high[0][0])}.")
    missing = len(rows) - len(valued)
    if missing:
        sentence += (f" {missing:,} returned {'row has' if missing == 1 else 'rows have'} "
                     "no usable numeric value for this comparison.")
    return sentence


def compose_answer(result, indices: list[int] | None = None) -> str:
    """A useful answer whose facts come exclusively from the executed result."""
    if getattr(result, "error", ""):
        return "The query could not be completed, so there is no verified result to summarize."
    rows = list(getattr(result, "rows", ()) or ())
    columns = list(getattr(result, "columns", ()) or ())
    if not rows:
        return ("No rows matched this query. "
                "Try broadening the filters or checking the available values.")
    if not columns:
        columns = [f"column {index + 1}" for index in range(len(rows[0]))]
    if len(rows) == 1 and len(rows[0]) == 1:
        answer = _scalar(result, rows[0][0], columns[0])
    else:
        valid = (parse_selection(json.dumps({"rows": indices}), result)
                 if indices is not None else None)
        chosen = valid if valid is not None else list(range(min(len(rows), MAX_HIGHLIGHT_ROWS)))
        if len(rows) == 1:
            facts = "; ".join(f"{_label(col)}: {_value(value)}"
                              for col, value in zip(columns, rows[0], strict=True))
            answer = f"The query returned one record with {facts}."
        else:
            answer = f"The query returned {len(rows):,} rows."
            comparison = _comparison(rows, columns)
            if comparison:
                answer += " " + comparison
            answer += (" The returned results are:" if len(chosen) == len(rows)
                       else f" Here are {len(chosen)} selected rows from that result:")
            details = []
            for index in chosen:
                details.append("; ".join(f"{_label(col)}: {_value(value)}"
                                          for col, value in zip(columns, rows[index], strict=True)))
            answer += "\n\n" + "\n".join(f"- {detail}" for detail in details)
    if getattr(result, "truncated", False):
        answer += (f"\n\nOnly the first {len(rows):,} rows are shown; more rows matched. "
                   "Any comparisons above describe this preview, not the full dataset.")
    return answer
