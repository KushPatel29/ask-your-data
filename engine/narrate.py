"""
A human-readable answer sentence for a compiled plan.

Called by `app/streamlit_app.py::_plan_headline`, which restated grouped results
well enough and fell off a cliff for scalars: every COUNT, SUM, AVG, MIN, MAX
and SHARE answer arrived as bare digits, because the last line of that function
is `return _fmt(value)`. "876" is a correct answer to "how many denied claims
are there?" and it is not an answer a person would give.

THE RULE THIS MODULE IS BUILT AROUND

No language model is involved. So this cannot be a summary, an insight, or a
characterisation -- it is a RESTATEMENT of the compiled query in words, with the
returned numbers substituted in. Every noun in the output sentence traces to an
identifier in the plan (a table name, a column name, a literal in a WHERE
clause) or to a value in `ran.rows`. Nothing else is allowed in.

That constraint is not stylistic. The whole argument of this project is that the
number next to the SQL is auditable; a fluent sentence that asserts one thing
more than the query returned is worse than no sentence at all, because it is the
part a reader will quote and the part nobody will check against the SQL.

Four rules follow from it, and they are the reason several obvious-looking
improvements are absent:

  1. No currency symbol, ever. `revenue` is a DOUBLE named "revenue"; nothing in
     the warehouse says dollars. Unit sense controls PRECISION and PLURALITY
     (2dp for money-ish columns, integers for counts, "%" only when the SQL
     itself computed a share or the identifier explicitly says pct/percentage),
     never a symbol that would be a claim.
  2. Row counts are scoped to what came back. A `LIMIT 5` result knows it has
     five rows and does not know how many groups exist, so it says "the 5
     departments shown" and never "of 5 departments".
  3. A NULL aggregate is reported as an absence of VALUES, not an absence of
     ROWS. `AVG(allowed_amount)` over 1,378 pending claims is NULL because every
     one of those rows is NULL in that column -- saying "no claims matched"
     would be false.
  4. A COUNT over a join that can fan out counts join rows, not entities, so the
     entity noun is withheld in that case. `edge.unique_side` is on the plan;
     the check costs nothing and it is the difference between "1,483 employees"
     and "1,483 rows".

USAGE
    from engine.narrate import answer_sentence
    answer_sentence(result.plan, ran)      # -> str

`ran` needs `.rows`, `.row_count` and `.truncated` -- i.e. an
`engine.query.QueryResult`. `plan` is an `engine.planner.Plan`.
"""

from __future__ import annotations

import re
from numbers import Number

from engine import planner

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Structural tokens in a table name that say nothing about what a ROW is.
# `hr_fact_employees` has one row per employee; "fact" is warehouse plumbing.
_TABLE_NOISE = {"fact", "facts", "dim", "dims", "dimension", "fct", "tbl", "vw"}

# Dropped from a column identifier before it is spoken. Shorter than the list
# the current `_humanise` uses, and deliberately: `_humanise` drops "total",
# "sum" and "avg" because it was written to also read RESULT ALIASES like
# `avg_base_salary`. This module only ever reads `plan.measure.name` and
# `plan.group_by.name`, which are real columns -- so dropping "total" would
# turn `total_revenue`, a stored per-customer total, into "revenue" and quietly
# rename the thing being reported.
_COLUMN_NOISE = {"id", "key", "code", "num", "cnt", "pct"}
# "name" is noise on a LABEL (`customer_name` names a customer) and not on a
# measure, so it is applied only where it belongs.
_LABEL_NOISE = _COLUMN_NOISE | {"name"}

# A trailing token that names the unit of the column it sits on. Only ever read
# in FINAL position: `tenure_months` is a tenure measured in months, but
# `days_to_adjudicate` is not an "adjudicate measured in days" -- it is a
# quantity whose whole name is "days to adjudicate".
_UNIT_TOKENS = {
    "months": "months", "month": "months", "days": "days", "day": "days",
    "hours": "hours", "hour": "hours", "years": "years", "year": "years",
    "weeks": "weeks", "week": "weeks", "minutes": "minutes",
    "seconds": "seconds", "cad": "CAD", "usd": "USD", "eur": "EUR",
    "gbp": "GBP", "kg": "kg", "lbs": "lbs",
}

# Money-ish, for PRECISION ONLY. A column in this class prints with two decimal
# places because cents are meaningful in it. It never prints with a currency
# symbol -- see rule 1 in the module docstring.
_MONEY_WORDS = {"amount", "revenue", "salary", "spend", "cost", "price",
                "margin", "sales", "paid", "billed", "allowed", "charge",
                "charges", "fee", "fees", "pay", "wage", "wages", "gmv",
                "submitted", "collected", "balance", "expense", "budget"}

# Columns whose VALUE reads as an adjective in front of the entity noun:
# `status = 'Denied'` -> "denied claims". Deliberately narrow. `payer_type` is
# NOT in here: "Medicare claims" would be an editorial compression of
# `payer_type = 'Medicare'`, and "claims for Medicare" is the same fact without
# the compression.
_STATUS_WORDS = {
    "status", "state", "result", "outcome", "disposition", "stage", "band",
}

# Columns a value sits IN rather than one a row is FOR. Chooses the preposition
# and nothing else.
_IN_WORDS = {"region", "department", "country", "city", "location", "site",
             "store", "market", "segment", "category", "channel", "format",
             "bucket", "division", "area", "zone", "branch", "team", "group",
             "class", "tier", "band", "district", "province", "sector",
             "cohort", "quarter", "month", "week", "year"}

_BOOL_PREFIXES = ("is_", "has_", "was_", "did_")
_TRUE_LITERALS = {"1", "true", "TRUE"}
_FALSE_LITERALS = {"0", "false", "FALSE"}


def _tokens(identifier) -> list:
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", str(identifier or "")) if w]


def _plural(label: str) -> str:
    """"customer" -> "customers", "hour of day" -> "hours of day".

    The head-noun rule matters more than it looks. `hour_of_day` is a real
    dimension in this warehouse, and the naive rule pluralised the tail:
    "24 hour of days".
    """
    if not label:
        return label
    parts = label.split(" ")
    if len(parts) >= 3 and parts[1] in ("of", "per", "by"):
        return " ".join([_plural(parts[0])] + parts[1:])
    head = parts[-1]
    if head.endswith(("us", "ss")):
        out = head + "es"                       # status -> statuses
    elif head.endswith("s"):
        out = head
    elif head.endswith("y") and head[-2:-1] not in "aeiou":
        out = head[:-1] + "ies"
    elif head.endswith(("ch", "sh", "x", "z")):
        out = head + "es"
    else:
        out = head + "s"
    return " ".join(parts[:-1] + [out])


def _singular(label: str) -> str:
    """Only used for "1 claim" and "just one department"; conservative."""
    if label.endswith("ies"):
        return label[:-3] + "y"
    if label.endswith(("ses", "xes", "ches", "shes")):
        return label[:-2]
    if label.endswith("s") and not label.endswith("ss"):
        return label[:-1]
    return label


def _column_label(name, units: bool = True, noise=None) -> tuple:
    """`tenure_months` -> ("tenure", "months"); `base_salary` -> ("base salary", "").

    The unit is only taken from the LAST token, and only when something is left
    after removing it. Everything else about the identifier survives, because
    the identifier is the evidence.

    `units=False` for anything that is not a measured quantity. `hour_of_day` is
    a dimension, and reading its last token as a unit produced "all 24 hour ofs".
    """
    words = _tokens(name)
    unit = ""
    if units and len(words) > 1 and words[-1] in _UNIT_TOKENS:
        unit = _UNIT_TOKENS[words[-1]]
        words = words[:-1]
    drop = _COLUMN_NOISE if noise is None else noise
    kept = [w for w in words if w not in drop] or words
    return " ".join(kept), unit


def _dimension_label(name) -> str:
    """A group-by or label column, spoken. No unit reading, "name" is noise."""
    return _column_label(name, units=False, noise=_LABEL_NOISE)[0]


def _fans_out(plan) -> bool:
    """Can this plan's joins multiply the base rows? The planner's own walk."""
    current = plan.base
    for edge in getattr(plan, "joins", None) or []:
        target = edge.other(current)
        unique_on = edge.right if edge.unique_side == "right" else edge.left
        if unique_on != target:
            return True
        current = target
    return False


def _entity_noun(plan) -> str:
    """What one row of `plan.base` is, plural. A restatement of the table name.

    `hr_fact_employees` -> "employees". `wholesale_finance_monthly` ->
    "finance monthlies" -- awkward, and left awkward: guessing a nicer noun
    would be the first invented fact in a module whose whole point is not to
    invent any.

    Returns "rows" when a join on the plan can multiply the base rows, because
    then COUNT(*) is not counting entities. `_fans_out` in the planner guards
    only SUM and AVG (planner.py:1721), so a COUNT over a fan-out join is
    reachable, and this is the only place it would be misdescribed.
    """
    if _fans_out(plan):
        return "rows"
    words = _tokens(plan.base)
    if len(words) > 1:
        words = words[1:]                       # the domain prefix
    words = [w for w in words if w not in _TABLE_NOISE] or words
    return _plural(" ".join(words))


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def _measure_kind(plan) -> str:
    """"pct" | "count" | "money" | "plain" -- precision, not meaning."""
    if plan.aggregate == planner.SHARE:
        return "pct"
    if plan.aggregate in (planner.COUNT, planner.COUNT_DISTINCT):
        return "count"
    if plan.measure is None:
        return "count"
    words = set(_tokens(plan.measure.name))
    # A rate/ratio is not necessarily stored on a 0-100 scale. The warehouse's
    # `net_collection_rate`, for example, contains 0.24; appending a percent
    # sign would turn that ratio into 0.24%, a 100x error in plain sight. A
    # pct/percent/percentage identifier does declare its scale, while SHARE's
    # compiler explicitly multiplies by 100.
    if words & {"pct", "percent", "percentage"}:
        return "pct"
    if words & _MONEY_WORDS:
        return "money"
    if str(plan.measure.type).upper() in ("BIGINT", "INTEGER", "INT", "HUGEINT",
                                          "SMALLINT", "TINYINT"):
        return "count"
    return "plain"


def _num(value, kind: str = "plain") -> str:
    if value is None:
        return "no value"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, Number):
        return str(value)
    if kind == "pct":
        return f"{value:,.1f}%"
    if kind == "money":
        return f"{value:,.2f}"
    if kind == "count":
        return (f"{int(round(value)):,}" if float(value).is_integer()
                else f"{value:,.2f}")
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _amount(value, kind: str, unit: str) -> str:
    """A number plus the unit its own column name declared, if any."""
    text = _num(value, kind)
    if unit and isinstance(value, Number) and not isinstance(value, bool):
        return f"{text} {unit}"
    return text


def _label_value(value) -> str:
    """A group's label. Numbers keep separators; text is left exactly as stored."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Number):
        return _num(value, "plain")
    if value is None:
        return "(no value)"
    return str(value)


def _measure_unit(plan) -> str:
    if plan.measure is None:
        return ""
    if plan.aggregate in (planner.COUNT, planner.COUNT_DISTINCT, planner.SHARE):
        return ""
    return _column_label(plan.measure.name)[1]


# ---------------------------------------------------------------------------
# Filters, in words
# ---------------------------------------------------------------------------

_OP_WORDS = {">": "above", ">=": "at least", "<": "below", "<=": "at most",
             "=": "of", "<>": "other than", "!=": "other than"}


def _unquote(literal) -> str:
    text = str(literal)
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'")
    return text


def _in_members(filt) -> list:
    if filt.literals:
        return [_unquote(x) for x in filt.literals]
    inner = str(filt.literal).strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    return [_unquote(p.strip()) for p in inner.split(",") if p.strip()]


def _join_or(items: list) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f" or {items[-1]}"


def _preposition(column) -> str:
    return "in" if set(_tokens(column)) & _IN_WORDS else "for"


def _flag_adjective(filt) -> str:
    """`is_active = 1` -> "active"; `is_active = 0` -> "not active"."""
    lowered = str(filt.column).lower()
    stem = ""
    for prefix in _BOOL_PREFIXES:
        if lowered.startswith(prefix):
            stem = str(filt.column)[len(prefix):]
            break
    if not stem:
        return ""
    literal = str(filt.literal)
    if literal in _TRUE_LITERALS:
        truth = filt.op in ("=", "==")
    elif literal in _FALSE_LITERALS:
        truth = filt.op in ("<>", "!=")
    else:
        return ""
    label, _ = _column_label(stem)
    return label if truth else f"not {label}"


def _status_adjective(filt) -> str:
    """`status = 'Denied'` -> "denied"; `status <> 'Denied'` -> "not denied".

    Narrow by design: a status-ish column, a single-word value, an (in)equality.
    The literal is lowercased because it is being used as an English adjective,
    and its exact stored casing is one line away in the SQL panel.
    """
    if not (set(_tokens(filt.column)) & _STATUS_WORDS):
        return ""
    if filt.op not in ("=", "<>", "!="):
        return ""
    value = _unquote(filt.literal)
    if not re.fullmatch(r"[A-Za-z][A-Za-z\-]*", value):
        return ""
    word = value.lower()
    return word if filt.op == "=" else f"not {word}"


def _try_number(literal):
    text = _unquote(literal)
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _range_phrase(low, high) -> str:
    """"with allowed amount between 1,000 and 2,000".

    The bounds print as PLAIN numbers even on a money column. They are what the
    asker typed, echoed back so they can check the compiler heard them; giving
    `>= 1000` two decimal places invents a precision the question did not have.
    """
    label, unit = _column_label(low.column)
    lo = _num(_try_number(low.literal), "plain")
    hi = _num(_try_number(high.literal), "plain")
    tail = f" {unit}" if unit else ""
    return f"with {label} between {lo} and {hi}{tail}"


def _year_phrase(filt) -> str:
    label, _ = _column_label(filt.column)
    article = "an" if label[:1] in "aeiou" else "a"
    return f"with {article} {label} in {_unquote(filt.literal)}"


def _filter_words(filters) -> tuple:
    """(adjectives, prepositional phrases) for a list of Filters.

    Adjectives go in front of the entity noun, phrases behind it. Two filters on
    one column with `>=` and `<=` collapse into a single "between", because the
    planner emits a range as two rows in `plan.filters` and reading them
    separately produced "with allowed amount at least 1,000 with allowed amount
    at most 2,000".
    """
    adjectives = []
    phrases = []
    filters = list(filters or [])

    used = set()
    for i, low in enumerate(filters):
        if i in used or low.op != ">=":
            continue
        for j, high in enumerate(filters):
            if j in used or j == i or high.op != "<=":
                continue
            if (high.table, high.column, high.expr) == (low.table, low.column, low.expr):
                phrases.append(_range_phrase(low, high))
                used.update({i, j})
                break

    for i, filt in enumerate(filters):
        if i in used:
            continue
        if filt.expr and "EXTRACT(YEAR" in str(filt.expr).upper():
            phrases.append(_year_phrase(filt))
            continue
        if str(filt.op).upper() == "IN":
            phrases.append(f"{_preposition(filt.column)} "
                           f"{_join_or(_in_members(filt))}")
            continue
        if str(filt.op).upper() == "NOT IN":
            phrases.append(f"not {_preposition(filt.column)} "
                           f"{_join_or(_in_members(filt))}")
            continue
        adjective = _flag_adjective(filt) or _status_adjective(filt)
        if adjective:
            adjectives.append(adjective)
            continue
        literal = str(filt.literal)
        if literal.startswith("'"):
            prep = _preposition(filt.column)
            value = _unquote(literal)
            phrases.append(f"{prep} {value}" if filt.op == "="
                           else f"not {prep} {value}")
            continue
        label, unit = _column_label(filt.column)
        word = _OP_WORDS.get(filt.op, filt.op)
        tail = f" {unit}" if unit else ""
        phrases.append(f"with {label} {word} "
                       f"{_num(_try_number(literal), 'plain')}{tail}")
    return adjectives, phrases


def _noun_phrase(plan, count=None) -> str:
    """"denied claims", "active employees in Data & Analytics", "1 claim".

    A NEGATED adjective moves behind the noun as a relative clause. English does
    not front them: `status <> 'Denied'` is "claims that are not denied", and
    "not denied claims" -- which the first draft produced -- reads as a denial
    of the whole phrase rather than of the status.
    """
    adjectives, phrases = _filter_words(plan.filters)
    plain = [a for a in adjectives if not a.startswith("not ")]
    negated = [a[4:] for a in adjectives if a.startswith("not ")]
    noun = _entity_noun(plan)
    if count == 1:
        noun = _singular(noun)
    out = " ".join(plain + [noun])
    if negated:
        out += " that are not " + " or ".join(negated)
    return " ".join([out] + phrases)


def _scope(plan, preposition: str) -> str:
    """" for denied claims" / " in Electronics or Grocery" / "".

    When every filter is already a prepositional phrase the entity noun adds
    nothing and is dropped -- "total revenue in Electronics or Grocery" beats
    "total revenue for department months in Electronics or Grocery" and says
    exactly the same thing.
    """
    if not plan.filters:
        return ""
    adjectives, phrases = _filter_words(plan.filters)
    if not adjectives:
        return " " + " ".join(phrases)
    return f" {preposition} " + _noun_phrase(plan)


# ---------------------------------------------------------------------------
# How the measure is spoken
# ---------------------------------------------------------------------------

def _superlatives(plan) -> tuple:
    """("most", "fewest") for a count, ("highest", "lowest") for a quantity."""
    if _measure_kind(plan) == "count":
        return "most", "fewest"
    return "highest", "lowest"


def _measure_phrase(plan) -> str:
    """The quantity, named the way the SELECT list names it."""
    if plan.aggregate == planner.SHARE:
        if plan.measure is not None:
            return f"the share of {_column_label(plan.measure.name)[0]}"
        return f"the share of {_entity_noun(plan)}"
    if plan.aggregate == planner.COUNT_DISTINCT and plan.measure is not None:
        return f"the number of distinct {_plural(_column_label(plan.measure.name)[0])}"
    if plan.aggregate == planner.COUNT or plan.measure is None:
        return f"the number of {_entity_noun(plan)}"
    word = {planner.SUM: "total", planner.AVG: "average",
            planner.MIN: "lowest", planner.MAX: "highest"}.get(plan.aggregate, "")
    label = _column_label(plan.measure.name)[0]
    # `SUM(total_revenue)` is a total of a column already called "total
    # revenue". Saying so twice is not more accurate, only worse.
    if word and label.split(" ")[0] == word:
        word = ""
    # Joined rather than stripped: with `word` emptied above, an f-string
    # leaves "the  revenue" with two spaces in the middle, which .strip()
    # cannot reach.
    return " ".join(part for part in ("the", word, label) if part)


def _upper1(text: str) -> str:
    """Capitalise the first letter and touch nothing else.

    `str.capitalize()` lowercases the tail, which would rewrite any stored value
    that reached the sentence through a label.
    """
    return text[:1].upper() + text[1:]


def _bare(phrase: str) -> str:
    """Drop a leading article so the phrase can follow "has the ..."."""
    return phrase[4:] if phrase.startswith("the ") else phrase


def _share_predicate(plan) -> str:
    """What the SHARE numerator selected: "cash", "denied", "in Electronics"."""
    filt = plan.share_filter
    if filt is None:
        return "matching"
    adjective = _flag_adjective(filt) or _status_adjective(filt)
    if adjective:
        return adjective
    if str(filt.op).upper() == "IN":
        return f"{_preposition(filt.column)} {_join_or(_in_members(filt))}"
    literal = str(filt.literal)
    if literal.startswith("'"):
        return f"{_preposition(filt.column)} {_unquote(literal)}"
    label, _ = _column_label(filt.column)
    return f"with {label} {_OP_WORDS.get(filt.op, filt.op)} {_unquote(literal)}"


def _share_subject(plan) -> str:
    """The thing whose share was measured, as a sentence subject."""
    filt = plan.share_filter
    if filt is None:
        return "The matching rows"
    literal = str(filt.literal)
    if literal.startswith("'"):
        return _unquote(literal)
    adjective = _flag_adjective(filt) or _status_adjective(filt)
    if adjective:
        if adjective.startswith("not "):
            return f"Rows that are not {adjective[4:]}"
        return f"The {adjective} rows"
    return f"{filt.column} {filt.op} {literal}"


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

def _null_sentence(plan) -> str:
    """An aggregate that came back NULL.

    Two different worlds produce this -- zero matching rows, and matching rows
    that are all NULL in the measured column -- and neither the plan nor the
    result can tell them apart. So the sentence says only what is true in both:
    no matching row carried a value. "No rows matched" would have been false for
    `AVG(allowed_amount)` over the 1,378 pending claims that produced it.
    """
    label = "the measured column"
    if plan.measure is not None:
        label = _column_label(plan.measure.name)[0]
    scope = _scope(plan, "for") or f" across all {_entity_noun(plan)}"
    return (f"{_upper1(_measure_phrase(plan))}{scope} is unavailable — "
            f"no matching row has a value for {label}.")


def _scalar_sentence(plan, value) -> str:
    if value is None:
        return _null_sentence(plan)

    kind = _measure_kind(plan)
    unit = _measure_unit(plan)

    if plan.aggregate == planner.COUNT:
        if value == 0:
            return f"There are no {_noun_phrase(plan)}."
        verb = "is" if value == 1 else "are"
        return (f"There {verb} {_num(value, 'count')} "
                f"{_noun_phrase(plan, count=value)}.")

    if plan.aggregate == planner.COUNT_DISTINCT:
        label = ("value" if plan.measure is None
                 else _column_label(plan.measure.name)[0])
        tail = _scope(plan, "among")
        if value == 0:
            return f"There are no distinct {_plural(label)}{tail}."
        noun = label if value == 1 else _plural(label)
        verb = "is" if value == 1 else "are"
        return f"There {verb} {_num(value, 'count')} distinct {noun}{tail}."

    if plan.aggregate == planner.SHARE:
        share = _num(value, "pct")
        if plan.measure is not None:
            label = _column_label(plan.measure.name)[0]
            return (f"{_share_subject(plan)} accounts for {share} of {label}"
                    f"{_scope(plan, 'for')}.")
        return f"{share} of {_noun_phrase(plan)} are {_share_predicate(plan)}."

    scope = _scope(plan, "for") or f" across all {_entity_noun(plan)}"
    return (f"{_upper1(_measure_phrase(plan))}{scope} is "
            f"{_amount(value, kind, unit)}.")


# ---------------------------------------------------------------------------
# Rankings and breakdowns
# ---------------------------------------------------------------------------

def _rank_sentence(plan, ran) -> str:
    row = ran.rows[0]
    label_name = ("row" if plan.label is None
                  else _dimension_label(plan.label.name))
    measure_label, unit = _column_label(plan.measure.name)
    high, low = _superlatives(plan)
    direction = high if (plan.order or "desc") == "desc" else low
    subject = f"{label_name[:1].upper()}{label_name[1:]} {_label_value(row[0])}"
    value = _amount(row[1], _measure_kind(plan), unit)
    sentence = (f"{subject} has the {direction} {measure_label}"
                f"{_scope(plan, 'among')}, {value}.")
    if ran.row_count > 1:
        sentence += (f" The next {ran.row_count - 1:,} by {measure_label} "
                     f"are listed below.")
    return sentence


def _scope_words(plan, ran, dimension_plural: str) -> str:
    """"all 12 departments" / "the 5 departments shown".

    A LIMIT means the result knows how many ROWS it has and not how many GROUPS
    exist; truncation means the same. Neither may be spoken as a total. The
    current headline says "the highest of 5 departments" for a top-5 over
    twelve departments, which reads as a fact about the warehouse and is not
    one.
    """
    n = ran.row_count
    if getattr(ran, "truncated", False):
        return f"the first {n:,} {dimension_plural} shown"
    if plan.limit:
        return f"the {n:,} {dimension_plural} shown"
    return f"all {n:,} {dimension_plural}"


def _possessive(subject: str) -> str:
    return subject + ("'" if subject.endswith("s") else "'s")


def _grouped_phrase(plan, superlative: bool = False) -> str:
    """The measure, phrased for a breakdown sentence.

    `superlative=True` is the slot after "has the most/highest". A COUNT has to
    lose its "the number of" there -- "has the most number of employees" was the
    first draft, and the sentence a person writes is "has the most employees".
    """
    if plan.aggregate == planner.SHARE and plan.share_filter is not None:
        if plan.measure is None:
            return (f"the share of {_noun_phrase(plan)} that are "
                    f"{_share_predicate(plan)}")
        label = _column_label(plan.measure.name)[0]
        return f"{_possessive(_share_subject(plan))} share of {label}"
    if plan.aggregate == planner.COUNT:
        if superlative:
            return _noun_phrase(plan)
        return f"the number of {_noun_phrase(plan)}"
    return _measure_phrase(plan)


def _grouped_sentence(plan, ran) -> str:
    rows = ran.rows
    dimension = _dimension_label(plan.group_by.name)
    plural = _plural(dimension)
    kind = _measure_kind(plan)
    unit = _measure_unit(plan)
    phrase = _grouped_phrase(plan)
    top = _grouped_phrase(plan, superlative=True)

    valued = [r for r in rows
              if isinstance(r[1], Number) and not isinstance(r[1], bool)]
    empty = [r for r in rows if not (isinstance(r[1], Number)
                                     and not isinstance(r[1], bool))]

    if ran.row_count == 1:
        head = _label_value(rows[0][0])
        value = _amount(rows[0][1], kind, unit)
        if plan.order:
            high, low = _superlatives(plan)
            direction = high if plan.order == "desc" else low
            tail = ("" if plan.limit
                    else f" It is the only {dimension} in the result.")
            return f"{head} has the {direction} {_bare(top)}, {value}.{tail}"
        return f"Just one {dimension} matched: {head}, {_bare(phrase)} {value}."

    if not valued:
        return (f"None of the {ran.row_count:,} {plural} in the result has a "
                f"value for {_bare(phrase)}. The full breakdown is below.")
    if len(valued) == 1:
        only = valued[0]
        missing = ran.row_count - 1
        return (f"Only {_label_value(only[0])} has a value for {_bare(phrase)}: "
                f"{_amount(only[1], kind, unit)}. The other {missing:,} "
                # `dimension` is ALREADY singular - it is what _dimension_label
                # returns - so singularising it again produced "the other 1
                # statu". _singular is correct where it is applied to an
                # already-plural entity noun; it is wrong here.
                f"{dimension if missing == 1 else plural} "
                f"{'has' if missing == 1 else 'have'} no value.")

    scope = _scope_words(plan, ran, plural)
    missing = ""
    if empty:
        missing = (f" {_label_value(empty[0][0])} has no value."
                   if len(empty) == 1
                   else f" {len(empty):,} of them have no value.")

    if plan.order:
        high, low = _superlatives(plan)
        top_word, bottom_word = ((high, low) if plan.order == "desc"
                                 else (low, high))
        first, last = valued[0], valued[-1]
        return (f"Of {scope}, {_label_value(first[0])} has the {top_word} "
                f"{_bare(top)} ({_amount(first[1], kind, unit)}) and "
                f"{_label_value(last[0])} the {bottom_word} "
                f"({_amount(last[1], kind, unit)}).{missing}")

    lo = min(valued, key=lambda r: r[1])
    hi = max(valued, key=lambda r: r[1])
    prefix = ""
    lead = _upper1(phrase)
    if plan.filters and plan.aggregate not in (planner.COUNT, planner.SHARE):
        prefix = f"Among {_noun_phrase(plan)}, "
        lead = phrase
    return (f"{prefix}{lead} across {scope} ranges from {_label_value(lo[0])} "
            f"at {_amount(lo[1], kind, unit)} to {_label_value(hi[0])} at "
            f"{_amount(hi[1], kind, unit)}.{missing}")


def _empty_sentence(plan) -> str:
    """Zero rows came back. Only a grouped or ranked query can do that."""
    noun = _noun_phrase(plan)
    if plan.group_by is not None:
        dimension = _dimension_label(plan.group_by.name)
        return (f"No {noun} matched, so there is nothing to break down "
                f"by {dimension}.")
    return f"No {noun} matched."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def answer_sentence(plan, ran) -> str:
    """One sentence restating `plan`, carrying the numbers `ran` returned.

    Never characterises, never benchmarks, never says whether a number is good.
    Every noun in the output is a table name, a column name, a WHERE-clause
    literal, or a value from `ran.rows`.
    """
    if plan is None or ran is None:
        return ""
    if not ran.rows:
        return _empty_sentence(plan)
    if plan.aggregate == planner.RANK and len(ran.rows[0]) > 1:
        return _rank_sentence(plan, ran)
    if plan.group_by is not None and len(ran.rows[0]) > 1:
        return _grouped_sentence(plan, ran)
    return _scalar_sentence(plan, ran.rows[0][0])
