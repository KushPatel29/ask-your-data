"""
A deterministic natural-language-to-SQL planner. No model, no API key.

WHY THIS EXISTS

Until now this repo had two modes and only one of them was real. With a key, the
model wrote SQL for whatever you typed. Without one, the app disabled its own
chat box and offered a dropdown of 39 pre-registered questions whose SQL was
committed in `evals/golden_questions.yaml`. That fallback was honest -- every
answer was labelled "reference SQL, not written by the model" -- but it was
still a demo of a product rather than the product. The public deployment, which
is the only version most people will ever touch, could not answer a question
nobody had thought of in advance. That is precisely the failure the README
accuses dashboards of.

So: a second engine. This one compiles the question itself, from the semantic
layer in `engine/semantics.py`, using a grammar you can read and tests can pin
down. It answers questions nobody pre-registered, it runs keyless, and it costs
nothing per question.

WHAT IT IS NOT

It is not as good as the model, and the repo says so with a number rather than a
disclaimer: `scripts/run_planner_eval.py` scores it against the same 39 golden
questions the model is graded on, and commits the result. Where it fails it
refuses; it never guesses. A planner that answered everything by picking the
nearest column would be worse than the chatbot this project was built to argue
against, because it would be confidently wrong for free.

The division of labour is the interesting part, and it is the honest version of
"do you even need an LLM for this":

  * Questions that name their measure and their dimension in words the schema
    uses -- "average claim amount by payer type", "how many denied claims",
    "top 5 departments by headcount" -- are compilable. There is no ambiguity
    for a model to resolve, so paying one to resolve it is waste.
  * Questions that need a definition the schema does not contain -- "net
    collection rate", "which cohort churned", anything where the arithmetic
    lives in someone's head -- are not. The planner refuses and says which part
    it could not bind, which is also what tells you the question needs the
    model.

HOW A QUESTION BECOMES A QUERY

  1. Tokenise, and match the longest value phrases first, so "self pay" binds
     before "pay" does.
  2. Collect candidate tables from three independent sources: the hybrid
     retriever (proven in run_retrieval_eval.py), the layer's word index, and
     any table that owns a matched value.
  3. Read the aggregation verb, the group-by ("by X", "per X", "which X has
     the most Y"), the ordering, and the limit off a small phrase grammar.
  4. For every (table, measure, dimension) triple that the evidence supports,
     build a candidate plan and score it by how much of the question it
     explains. Unexplained content words cost confidence; they are the
     planner's own account of what it did not understand.
  5. Below a floor, refuse. Above it, compile -- and hand the SQL to the same
     `sql_guard`, the same executor, and the same `Verifier` the model's SQL
     goes through. The planner gets no privileges for being deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from engine.semantics import (
    DATE,
    DIMENSION,
    FLAG,
    KEY,
    MEASURE,
    Column,
    JoinEdge,
    Layer,
    normalise_phrase,
    split_identifier,
)

# ----------------------------------------------------------------------------
# The grammar. Every phrase here is one the planner will act on; anything else
# in a question is unexplained, and unexplained words cost confidence.
# ----------------------------------------------------------------------------

# "how much of" is deliberately NOT here. It reads as a count and it is not:
# "how much of the open AR do we expect to collect" wants a SUM of dollars, and
# listing it as a count phrase made that question return 1,378 -- the number of
# rows in the AR table -- with full confidence.
COUNT_PHRASES = ("how many", "number of", "count of", "how many of")
SUM_PHRASES = ("total", "sum of", "sum", "combined", "altogether", "how much",
               "overall value", "aggregate")
AVG_PHRASES = ("average", "avg", "mean", "typical", "on average", "per average")
MAX_PHRASES = ("highest", "largest", "biggest", "most", "maximum", "max", "top",
               "greatest", "best", "peak", "longest", "leading")
MIN_PHRASES = ("lowest", "smallest", "least", "minimum", "min", "bottom",
               "worst", "fewest", "shortest", "poorest")
SHARE_PHRASES = ("rate", "percentage", "percent", "share", "proportion", "ratio",
                 "what fraction", "fraction")
DISTINCT_PHRASES = ("distinct", "unique", "different")

# Aggregates a person can reasonably ask for and this grammar cannot write.
# Naming one is a refusal: "what's the median salary?" fell through to SUM and
# answered 940,000 -- the sum of a benchmark table's median column, which is not
# a median of anything. Silently answering a different statistic is worse than
# saying no, because the word the user typed is right there in the question.
UNSUPPORTED_PHRASES = ("median", "percentile", "quartile", "stdev", "std dev",
                       "standard deviation", "variance", "correlation", "mode",
                       "year over year", "yoy", "month over month", "mom",
                       "moving average", "rolling", "cumulative", "running total",
                       "trend", "forecast", "growth rate", "cagr")

COUNT = "count"
COUNT_DISTINCT = "count_distinct"
SUM = "sum"
AVG = "avg"
MIN = "min"
MAX = "max"
SHARE = "share"
# Not an aggregate at all: rank the rows of a table that is ALREADY at the grain
# the question asks about. "Who is the top wholesale customer by revenue?" reads
# `retail_customer_analytics`, which holds one row per customer with
# `total_revenue` already computed -- so grouping is not just unnecessary, it is
# wrong: GROUP BY a unique column produces one group per row and the aggregate
# is a no-op wrapped around the value. Without this shape the planner refused
# every "who/which single thing is the biggest" question in the warehouse.
RANK = "rank"

# "by region", "per store", "for each department", "broken down by channel".
#
# Bare "each" is here as well as "for each". "How many employees are in EACH
# department?" is ordinary English for a breakdown, and without it that question
# found no axis, produced one grand total, and reported 1,900 where the answer
# is twelve rows -- a wrong shape rather than a wrong number, and the harder
# kind to notice.
GROUP_MARKERS = ("by", "per", "for each", "each", "across", "grouped by",
                 "broken down by")

# The markers that unambiguously DEMAND a breakdown, as opposed to merely
# suggesting one. "per" is the odd one out and it is why this split exists:
# "revenue per store" is an axis, but "average allowed amount per claim" is a
# statement about grain -- the answer is one number, and `per claim` is telling
# you what that number is an average OF. Every other marker in the list has only
# the axis reading.
AXIS_MARKERS = ("by", "for each", "each", "across", "grouped by", "broken down by")

# "which department has the most ...", "what payer type has the lowest ..."
WHICH_RE = re.compile(r"\b(which|what|who)\b", re.I)

TOP_N_RE = re.compile(r"\b(?:top|bottom|first|last)\s+(\d{1,3})\b", re.I)
COMPARE_RE = re.compile(
    r"\b(over|above|more than|greater than|at least|under|below|less than|"
    r"fewer than|at most|exactly|equal to)\s+\$?([0-9][0-9,_]*(?:\.[0-9]+)?)\b", re.I)
YEAR_RE = re.compile(r"\b(?:in|during|for)\s+(19|20)(\d{2})\b")

COMPARE_OPS = {
    "over": ">", "above": ">", "more than": ">", "greater than": ">",
    "at least": ">=", "under": "<", "below": "<", "less than": "<",
    "fewer than": "<", "at most": "<=", "exactly": "=", "equal to": "=",
}

# Words that carry no binding signal. Kept deliberately small: a stopword list
# that swallows domain nouns is a stopword list that hides failures, because
# every word it removes is one the confidence score stops holding the planner
# accountable for.
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "did", "do",
    "does", "each", "for", "from", "get", "give", "had", "has", "have", "how",
    "i", "in", "into", "is", "it", "its", "many", "me", "much", "of", "on", "or",
    "our", "out", "show", "so", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "whose", "will", "with", "would", "you",
    "your", "s", "list", "tell", "find", "please", "just", "any", "all", "some",
    "over", "under", "per", "across", "between", "up", "down",
}

# The fraction of a question's content words that must be accounted for by an
# actual binding before the planner will compile it.
#
# This number is the whole safety story, and the first version of it was wrong
# in an instructive way. The gate used to be a BLEND of coverage and a structure
# bonus, and a blend lets a plan buy its way past the floor with shape: bind
# something as a measure, something as a dimension, and two-thirds of the
# question could go unexplained and still clear 0.34. Scored against the 39
# golden questions that produced 8 right answers and 26 WRONG ones -- confidently
# wrong, with SQL, for free, which is precisely the failure this repo was built
# to argue against.
#
# So the gate is coverage alone, and it is high. Structure still contributes to
# picking BETWEEN plans, but it can no longer rescue one.
#
# `python scripts/run_planner_eval.py --sweep` prints the trade over all 64
# questions in both contracts, and the shape of it is the argument:
#
#     gate   match   differs   refused
#     0.40      31        17        16
#     0.50      31        15        18
#     0.60      29         5        30
#     0.70      28         1        35     <- shipped
#     0.80      27         0        37
#
# Loosening to 0.40 buys three more right answers and seventeen wrong ones.
# Tightening to 0.80 does reach zero disagreements, and it costs a correct
# answer to do it -- "who is the top wholesale customer by revenue?" starts
# refusing, because `wholesale` is a modifier this warehouse also uses as a
# domain name. 0.70 keeps that answer and leaves exactly one disagreement,
# `denial_rate`, which is not an arithmetic error: the planner computes denied
# over ALL claims and the house definition excludes pending ones. No amount of
# schema introspection discovers a convention that lives in a policy document.
# That question is why a metric layer exists, and why the model is still worth
# paying for.
MIN_COVERAGE = 0.70

# How many candidate tables get a plan built for them. Binding one table is a
# few hundred microseconds of pure Python, so depth is nearly free; the reason
# not to make it 71 is that every extra table is another chance for a
# coincidence to outscore the right answer.
CANDIDATE_DEPTH = 14

# Kept as the old name for callers that pass it; it now gates coverage.
MIN_CONFIDENCE = MIN_COVERAGE

# Analytics vocabulary that maps English onto column words the schema actually
# uses. This is the ONE hand-written mapping in the layer, and it is deliberately
# tiny: it covers words a business user says that no schema ever spells
# ("revenue", "headcount"), and nothing else. Every other binding is derived.
SYNONYMS = {
    "revenue": ("revenue", "sales", "amount", "net", "value"),
    "sales": ("sales", "revenue", "amount", "net"),
    "spend": ("spend", "cost", "amount"),
    "cost": ("cost", "spend", "expense", "amount"),
    "headcount": ("employees", "headcount", "employee"),
    "staff": ("employees", "employee", "headcount"),
    "people": ("employees", "employee", "headcount", "subjects", "subject"),
    "employees": ("employees", "employee", "headcount"),
    "salary": ("salary", "compensation", "pay", "comp"),
    "pay": ("salary", "pay", "compensation"),
    "customers": ("customer", "customers", "account"),
    "orders": ("order", "orders"),
    "profit": ("profit", "margin"),
    "margin": ("margin", "profit"),
    "quantity": ("qty", "quantity", "units"),
    "units": ("qty", "units", "quantity"),
    "amount": ("amount", "value", "total"),
    "payments": ("payment", "paid", "amount", "transactions"),
    "transactions": ("transaction", "transactions", "txn"),
    "spending": ("spend", "cost", "amount"),
    "tenure": ("tenure", "years"),
    "attrition": ("attrition", "terminated", "left", "churn"),
    "churn": ("churn", "attrition", "terminated"),
    "flight": ("flight", "risk"),
    "denied": ("denied", "denial"),
    "denials": ("denial", "denied"),
}


@dataclass(frozen=True)
class Filter:
    """One WHERE clause, and the words in the question that produced it."""

    table: str
    column: str
    op: str
    literal: str
    evidence: str
    # An optional SQL expression to compare instead of the bare column. Exists
    # for exactly one case today and it was a crash: "how many claims were
    # submitted in 2024?" emitted `service_date = 2024`, and DuckDB answered
    # `Conversion Error: Unimplemented type for cast (INTEGER -> DATE)`. A year
    # is a predicate ON a date, not a value OF one.
    expr: str = ""

    def sql(self) -> str:
        left = self.expr or f'"{self.table}"."{self.column}"'
        return f"{left} {self.op} {self.literal}"


@dataclass
class Plan:
    base: str
    aggregate: str
    joins: list[JoinEdge] = field(default_factory=list)
    measure: Column | None = None
    group_by: Column | None = None
    label: Column | None = None
    filters: list[Filter] = field(default_factory=list)
    share_filter: Filter | None = None
    order: str = ""
    limit: int | None = None
    explained: set[str] = field(default_factory=set)
    unexplained: set[str] = field(default_factory=set)
    coverage: float = 0.0
    confidence: float = 0.0
    sql: str = ""

    @property
    def tables(self) -> list[str]:
        names = [self.base]
        for edge in self.joins:
            for side in (edge.left, edge.right):
                if side not in names:
                    names.append(side)
        return names

    def rationale(self) -> list[tuple[str, str]]:
        """Every binding decision, as (what, why). This is the panel's content.

        A deterministic planner has no excuse for being a black box: unlike a
        model, it can say exactly which word bought which clause.
        """
        out: list[tuple[str, str]] = []
        out.append(("table", f"{self.base} — {len(self.joins)} join"
                             f"{'' if len(self.joins) == 1 else 's'}"))
        if self.aggregate == SHARE and self.share_filter:
            out.append(("metric", f"share of rows where "
                                  f"{self.share_filter.column} = "
                                  f"{self.share_filter.literal}"))
        elif self.measure is not None:
            out.append(("metric", f"{self.aggregate.upper()}({self.measure.name})"))
        else:
            out.append(("metric", f"{self.aggregate.upper()}(*)"))
        if self.group_by is not None:
            out.append(("grouped by", self.group_by.name))
        for filt in self.filters:
            out.append(("filter", f'{filt.column} {filt.op} {filt.literal} '
                                  f'(from "{filt.evidence}")'))
        if self.limit:
            out.append(("limit", f"{self.limit} row{'s' if self.limit != 1 else ''}"
                                 f"{', ' + self.order if self.order else ''}"))
        return out


@dataclass
class PlanResult:
    """What the planner did, whether or not it produced SQL."""

    question: str
    plan: Plan | None = None
    refused: bool = False
    reason: str = ""
    candidates: list[str] = field(default_factory=list)
    unbound: set[str] = field(default_factory=set)
    considered: int = 0
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.refused

    @property
    def sql(self) -> str:
        return self.plan.sql if self.plan else ""


# ----------------------------------------------------------------------------
# Reading the question
# ----------------------------------------------------------------------------

def tokenise(question: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", question.lower()) if t]


def content_words(question: str) -> set[str]:
    """The words the planner is accountable for explaining."""
    return {t for t in tokenise(question) if t not in STOP and len(t) > 1}


_SUFFIXES = ("ation", "ically", "ical", "ingly", "ing", "ial", "ily", "ly",
             "ed", "al", "es", "s", "y", "i")


def stems(word: str) -> set[str]:
    """Every form of a word worth matching a data value against.

    A SET, not a single canonical stem, and that is the whole design. The
    question says "denial rate"; the column holds `Denied`. The question says
    "left voluntarily"; the column holds `Voluntary`. A canonical stemmer has to
    pick one output per word, and every choice that bridges one of those pairs
    breaks the other or collides something unrelated ("denial" and "dense" both
    reduce to "den" under a rule aggressive enough to reach "denied").

    Emitting the whole ladder of strippings and asking for ANY overlap keeps the
    bridge without the collision: `denial` offers {denial, deni, den}, `denied`
    offers {denied, deni, den}, and they meet at `deni` -- four characters, long
    enough to mean something. Forms shorter than four characters are dropped for
    exactly that reason.
    """
    low = word.lower()
    out = {low}
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            for form in list(out):
                if form.endswith(suffix) and len(form) - len(suffix) >= 4:
                    trimmed = form[: -len(suffix)]
                    if trimmed not in out:
                        out.add(trimmed)
                        changed = True
    return {f for f in out if len(f) >= 4} or {low}


def _expand(word: str) -> set[str]:
    out = {word}
    out.update(SYNONYMS.get(word, ()))
    # Cheap morphology, not a stemmer: the schema says `employee` and the
    # question says `employees`. A real stemmer would also fold `denied` into
    # `deni`, which loses the exact value match that makes the filter work.
    if word.endswith("ies") and len(word) > 4:
        out.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        out.add(word[:-2])
    if word.endswith("s") and len(word) > 3:
        out.add(word[:-1])
    else:
        out.add(word + "s")
    return out


def detect_aggregate(question: str) -> tuple[str, set[str]]:
    """Which aggregate the question asks for, and the words that said so."""
    low = f" {question.lower()} "
    hits: set[str] = set()

    def seen(phrases) -> bool:
        found = False
        for phrase in phrases:
            if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
                hits.update(phrase.split())
                found = True
        return found

    is_share = seen(SHARE_PHRASES)
    is_avg = seen(AVG_PHRASES)
    is_count = seen(COUNT_PHRASES)
    is_distinct = seen(DISTINCT_PHRASES)
    is_sum = seen(SUM_PHRASES)
    # MIN/MAX are read even when another aggregate wins, because "highest
    # average salary" is an AVG that is then ordered -- the superlative is about
    # the ordering, not about the aggregate.
    seen(MAX_PHRASES)
    seen(MIN_PHRASES)

    if is_share:
        return SHARE, hits
    if is_avg:
        return AVG, hits
    if is_count and is_distinct:
        return COUNT_DISTINCT, hits
    if is_count:
        return COUNT, hits
    if is_sum:
        return SUM, hits
    return "", hits


_LISTING_RE = re.compile(
    r"^\s*(?:list|show(?: me)?|display|give me|what are|which are)\b", re.I)


def is_listing_request(question: str) -> bool:
    """"list the departments" asks for rows, and this planner does not do rows.

    It used to compile that into `COUNT(*)` over a three-table join and answer
    56, which is not a shorter version of the list -- it is a different number
    about a different thing. Refusing names the gap; answering hides it.

    Only when no aggregate verb is present: "show me the total revenue by
    department" is an aggregate with a polite opening.
    """
    if not _LISTING_RE.match(question):
        return False
    if detect_aggregate(question)[0] or detect_order(question):
        return False
    # "Give me headcount by gender" opens like a listing and is not one: the
    # axis marker says a breakdown was asked for, and a breakdown is an
    # aggregate however the sentence begins. Without this the listing rule
    # refused a question the planner answers correctly.
    return not axis_and_measure_hints(question)[0]


def unsupported_aggregate(question: str) -> str:
    """The name of a statistic the question asked for that this grammar lacks."""
    low = f" {question.lower()} "
    for phrase in UNSUPPORTED_PHRASES:
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
            return phrase
    return ""


def detect_order(question: str) -> str:
    low = f" {question.lower()} "

    def seen(phrases) -> bool:
        return any(re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", low)
                   for p in phrases)

    if seen(MIN_PHRASES):
        return "asc"
    if seen(MAX_PHRASES):
        return "desc"
    return ""


def detect_limit(question: str) -> int | None:
    match = TOP_N_RE.search(question)
    if match:
        return max(1, min(int(match.group(1)), 100))
    return None


def group_hints(question: str) -> list[str]:
    """Words that ASK to be grouped on. Empty means: do not group.

    This function used to be generous, and generosity was the single worst bug
    in the planner. It read every `which`/`what`/`who` as a grouping request, so
    "What is the overall claim denial rate?" offered `overall` as a dimension
    hint, matched a rate column in an unrelated domain, and answered a
    healthcare question with a wholesale number. Worse, "How many active
    employees are there?" grouped BY `is_active` instead of filtering ON it, and
    returned the two-row breakdown with `0` on top.

    So grouping is now something a question has to ask for, in one of two ways:

      * an explicit marker -- "by department", "per store", "for each channel";
      * the superlative form -- "which payer type has the LOWEST ...", where the
        nouns between the wh-word and its verb name the dimension, and the
        superlative is what makes it a ranking rather than a lookup.

    `what is ...` yields nothing, because the first token after the wh-word is a
    verb and the noun phrase is empty. That is the whole fix.
    """
    return axis_and_measure_hints(question)[0]


def axis_and_measure_hints(question: str) -> tuple[list[str], list[str]]:
    """Split a question into the axis it wants and the measure it ranks by.

    English marks these two roles in three shapes, and reading them as one bag
    of words gets the roles backwards. "Who is the top wholesale customer BY
    REVENUE?" put `revenue` in the axis bag, where the synonym map helpfully
    matched it to `sales_rep_name`, and the top customer came back as a sales
    rep. The word `by` introduces the MEASURE in that sentence and the DIMENSION
    in "total spend by channel"; only the surrounding shape says which.

      1. "which/what/who <noun phrase> <verb> ... <superlative> <measure>"
         -> axis is the noun phrase, measure is the rest.
      2. "<superlative> <noun phrase> by <measure>"
         -> axis is the noun phrase, measure is what follows `by`.
      3. "... by/per/for each <noun phrase>"
         -> axis is the noun phrase; no separate measure marker.

    Case 1 is tried first and only counts when its noun phrase is non-empty:
    "who IS the top ..." puts a verb immediately after the wh-word, which is
    how a lookup is distinguished from a ranking.
    """
    low = question.lower()
    axis: list[str] = []
    measure: list[str] = []

    if _has_superlative(low):
        match = WHICH_RE.search(low)
        if match:
            noun = _NOUN_PHRASE_RE.match(low[match.end():])
            if noun:
                axis = [w for w in tokenise(noun.group(1))[:3] if w not in STOP]
                if axis:
                    measure = [w for w in tokenise(low[match.end() + noun.end():])
                               if w not in STOP and w not in _SUPERLATIVE_WORDS]
                    return axis, measure
        ranked = _RANKED_BY_RE.search(low)
        if ranked:
            axis = [w for w in tokenise(ranked.group(1))[:3] if w not in STOP]
            measure = [w for w in tokenise(ranked.group(2))[:3] if w not in STOP]
            if axis:
                return axis, measure

    for marker in GROUP_MARKERS:
        for hit in re.finditer(rf"(?<![a-z]){re.escape(marker)}\s+([a-z_ ]{{2,40}})", low):
            axis.extend(w for w in tokenise(hit.group(1))[:3] if w not in STOP)
    return axis, measure


# "top 3 departments by headcount", "the largest customer by revenue".
_RANKED_BY_RE = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in MAX_PHRASES + MIN_PHRASES) + r")"
    r"\s+(?:\d+\s+)?((?:[a-z]+\s+){0,3}?)by\s+([a-z_ ]{2,30})")


# Everything from the wh-word up to the verb that follows it. "which payer type
# has the lowest" -> "payer type ".
# The verb list is generous on purpose. It is what separates "which MODEL has
# the most tests" (a ranking, axis = model) from "what IS the denial rate" (a
# lookup, no axis) -- and a verb missing from it does not merely degrade the
# parse, it inverts it: "Which model took the longest execution seconds?" fell
# through as a lookup and answered SUM(execution_seconds) across every model, a
# single number where a name was asked for.
_NOUN_PHRASE_RE = re.compile(
    r"\s+((?:[a-z]+\s+){0,3}?)(?:has|have|had|is|are|was|were|do|does|did|"
    r"shows?|showed|generat\w*|produc\w*|account\w*|receiv\w*|with|contains?|"
    r"carr\w+|hold\w*|sold|sells?|earns?|makes?|ranks?|took|takes?|ran|runs?|"
    r"spent|spends?|used?|uses|got|gets?|cost|costs?|scored|scores?|hit|hits?|"
    r"reach\w*|post\w*|logg\w*|wrote|writes?|clos\w*|open\w*|drove|drives?|"
    r"deliver\w*|return\w*|bought|buys?|need\w*|saw|sees?|came|comes?)\b")


def _has_superlative(low: str) -> bool:
    return any(re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", f" {low} ")
               for p in MAX_PHRASES + MIN_PHRASES)


def demands_axis(question: str) -> bool:
    """Did the question require a breakdown, in words that admit no other read?

    Used to reject plans that quietly drop the GROUP BY. "How many employees are
    in each department?" chose `wholesale_labor_department_month` -- a table
    whose NAME contains the word department, which was enough to mark the hint
    word explained -- and returned one number. Coverage was 1.0 and the answer
    was a row count of the wrong table with no breakdown in it at all.

    Explaining a hint word is not the same as honouring it. A hint word must be
    accounted for by the AXIS COLUMN, and that is what this predicate gates.
    """
    low = f" {question.lower()} "
    for marker in AXIS_MARKERS:
        if re.search(rf"(?<![a-z]){re.escape(marker)}\s+[a-z_]{{2,}}", low):
            return True
    return False


def expects_category(question: str) -> bool:
    """Does the question want the NAME of something back, rather than a number?

    "Which payer type has the lowest net collection rate?" wants `Self-Pay`.
    "What is the denial rate?" wants `8.2`. The two are answered by differently
    shaped queries, and a planner that returns the wrong shape has not given a
    slightly worse answer -- it has answered a different question.

    This is the check that stops the most embarrassing class of failure. Asked
    "which category has the highest revenue", the planner used to bind revenue,
    fail to bind a dimension, and return `176265402.48` -- the grand total,
    presented as if it were the name of a category. Now a question that expects
    a category and produced no GROUP BY is a refusal.
    """
    low = f" {question.lower()} "
    if TOP_N_RE.search(low):
        return True
    if not _has_superlative(low):
        return False
    match = WHICH_RE.search(low)
    if not match:
        return False
    # A wh-word followed immediately by a verb is a lookup ("what is the ..."),
    # not a ranking, however many superlatives sit further along the sentence.
    tail = low[match.end():]
    return bool(_NOUN_PHRASE_RE.match(tail))


# "how many stores", "number of payers", "count of claims", "list the departments".
# The words that end a noun phrase. Without them the capture runs on into the
# rest of the sentence: "how many claims WITH ALLOWED AMOUNT above 5000" yielded
# the entity `[claims, allowed]`, and the rule that stops an entity being
# aggregated then removed `allowed_amount` -- the very column the question was
# filtering on. Stopping at the preposition keeps the entity to the thing being
# counted.
_ENTITY_STOP = (r"with|in|for|from|by|per|each|across|were|was|are|is|do|does|"
                r"did|that|which|who|whose|have|has|had|above|below|over|under|"
                r"between|during|and|or|of|to|at|on")

_ENTITY_RE = re.compile(
    r"(?:how many|number of|count of|list(?: all| the)?|show(?: me)?(?: all| the)?|"
    r"break ?down(?: of)?)"
    rf"\s+((?:(?!(?:{_ENTITY_STOP})\b)[a-z]+\s*){{1,3}})", re.I)


def entity_noun(question: str) -> list[str]:
    """The thing being counted, when the question says it is counting things.

    "How many stores do we have?" is not asking for an aggregate of a column
    called something like store -- it is asking for the number of stores, which
    is a question about a table's GRAIN. Left unmarked, that question found
    `wholesale_fact_segment_month`, filtered on the boolean `is_physical_store`
    and answered 170: the number of store-MONTHS. The right answer is 620, and
    it lives in a table this planner had no reason to prefer.

    The words are returned so the binder can (a) prefer a table whose own name
    is about them and (b) refuse to aggregate a measure that merely shares one.
    """
    match = _ENTITY_RE.search(question)
    if not match:
        return []
    # The span after the trigger is captured GREEDILY and cleaned here, rather
    # than captured minimally. "How many DISTINCT customers are there?" names
    # `customers`; a lazy quantifier stopped at `distinct`, which the cleaner
    # then dropped as an aggregate word -- leaving an empty entity, the rule
    # inert, and COUNT(DISTINCT new_customers) answering 4 where it is 120.
    words = [w for w in tokenise(match.group(1))
             if w not in STOP and w not in _AGGREGATE_WORDS
             and w not in _SUPERLATIVE_WORDS]
    return words[:2]


def entity_head(question: str) -> str:
    """The head noun of the counted entity -- the last word of the phrase.

    Reservation uses the HEAD only, and the distinction is load-bearing in both
    directions. "How many stores do we have?" has head `stores`, which must stop
    `is_physical_store` narrowing the question from 620 to 555. "How many active
    employees are there?" has head `employees` and modifier `active`, and
    reserving the modifier too would drop the `is_active = 1` filter that makes
    the answer 1,483 instead of 1,900.
    """
    words = entity_noun(question)
    return words[-1] if words else ""


def flag_filters(question: str, layer: Layer, table: str) -> list[Filter]:
    """Boolean columns the question names by their own word.

    `is_active` is a FLAG holding 0/1, so it never reaches the value lexicon --
    there is no string "active" in the data for a question to match. But the
    COLUMN says active, and "how many active employees" means `is_active = 1`.
    Binding the word to the column and the truth to the value is the only way
    that question compiles at all, and it is unambiguous: a flag has exactly two
    states and the question named the positive one.
    """
    words = content_words(question)
    out: list[Filter] = []
    for col in layer.tables[table].columns:
        if col.role != FLAG:
            continue
        # Drop the is_/has_ prefix: the question says "active", not "is active".
        stem = [w for w in col.words if w not in ("is", "has", "was", "did")]
        if not stem:
            continue
        if not any(w in words or (_expand(w) & words) for w in stem):
            continue
        base = col.type.upper().split("(")[0]
        literal = "TRUE" if base == "BOOLEAN" else "1"
        out.append(Filter(table=table, column=col.name, op="=", literal=literal,
                          evidence=" ".join(stem)))
    return out


def match_values(question: str, layer: Layer,
                 exclude: set[str] | None = None) -> list[tuple[str, list]]:
    """Value phrases the question contains, longest first.

    Longest-first matters: `self pay` and `pay` are both indexed, and binding
    the short one first would filter on the wrong column. The same rule is why
    the loop consumes matched spans out of the haystack.
    """
    # Words the question asked to GROUP BY are removed from the haystack before
    # any value matching happens. This is the single worst bug the planner had.
    # `hr_flight_risk_scores.top_reason` holds the literal string "department"
    # among its reason codes, so "how many employees are in each department"
    # matched it as a VALUE, joined to the flight-risk table, and answered 56 --
    # the number of employees whose top attrition reason happens to be their
    # department -- for a question whose real answer is a breakdown of 1,900.
    # It looked entirely plausible and it was not close to right.
    #
    # "by X" says X is an axis. An axis is never simultaneously a filter.
    text = " " + normalise_phrase(question) + " "
    for word in sorted(exclude or (), key=len, reverse=True):
        text = text.replace(f" {word} ", "  ")
    # A second haystack, stemmed word by word, so a single-word value can match
    # a differently-inflected question word. Multi-word phrases are matched
    # against the literal text only: stemming "self pay" gains nothing and
    # stemming both sides of a phrase multiplies the chance of a false hit.
    question_stems = {f for w in text.split() for f in stems(w)}
    phrases = sorted(layer.value_phrases, key=len, reverse=True)
    found: list[tuple[str, list]] = []
    for phrase in phrases:
        # Two characters is the floor rather than three, because `GO` is a real
        # verdict value in the migration domain and "how many GO verdicts" is a
        # question the planner should be able to compile. The space padding
        # already forces a whole-token match, so a short value cannot match
        # inside a longer word; STOP is what keeps a value literally spelled
        # "on" or "no" from firing on English.
        if len(phrase) < 2 or phrase in STOP:
            continue
        needle = f" {phrase} "
        if needle in text:
            found.append((phrase, layer.values_for_phrase(phrase)))
            text = text.replace(needle, "  ")
            question_stems -= stems(phrase)
            continue
        if " " not in phrase and (stems(phrase) & question_stems):
            found.append((phrase, layer.values_for_phrase(phrase)))
            question_stems -= stems(phrase)
    return found


def unbindable_words(question: str, layer: Layer) -> set[str]:
    """Question words that NOTHING in the warehouse could ever have matched.

    Coverage exists to hold the planner accountable for words it ignored. But
    "how many claims are in the dataset in total" contains `dataset`, and there
    is no table, column, value or domain blurb anywhere in these 71 tables that
    contains that word. Counting it against the planner marks it down for the
    user's vocabulary rather than for a binding failure, and it was the only
    thing standing between that question and its correct answer.

    So the accountable set is content words MINUS the ones no plan could have
    bound. Note what this is not: a stopword list. It is derived from the layer
    on every call, so a word is excused only by the actual absence of anywhere
    to put it -- add a table tomorrow whose description says "dataset" and the
    word starts counting again, automatically.

    The excused words are reported, not swallowed: `PlanResult.unbound` carries
    them to the UI, because "I have no vocabulary for `dataset`" is a true and
    useful thing for a visitor to see.
    """
    out: set[str] = set()
    for word in content_words(question):
        forms = _expand(word) | stems(word)
        if any(layer.tables_with_word(form) for form in forms):
            continue
        if any(layer.values_for_phrase(form) for form in forms):
            continue
        if word in _AGGREGATE_WORDS or word in _SUPERLATIVE_WORDS:
            continue
        out.add(word)
    return out


# ----------------------------------------------------------------------------
# Binding
# ----------------------------------------------------------------------------

def _score_column(col: Column, words: set[str], hints: list[str]) -> float:
    """How strongly the question points at this column.

    Words naming the OPERATION are removed first. "How many distinct customers
    are there?" scored `retail_customer_analytics.distinct_skus` at 0.5 on the
    word `distinct` and answered 28 -- a column matched on the name of the
    aggregate being applied to it, which is a coincidence rather than evidence.
    Only the distinct family is stripped: `rate` and `total` DO name real
    columns here (`fill_rate_pct`, `total_revenue`) and stripping them would
    break the questions that ask for those by name.
    """
    col_words = set(col.words)
    if not col_words:
        return 0.0
    words = words - _DISTINCT_WORDS
    expanded: set[str] = set()
    for word in words:
        expanded |= _expand(word)
    overlap = col_words & expanded
    if not overlap:
        return 0.0
    # Fraction of the column's own name that the question said, so `amount`
    # matching `amount` beats `amount` matching `baseline_mean_amount_cad`.
    score = len(overlap) / len(col_words)
    # An exact full-name mention is worth more than a partial one.
    if col.name.lower() in words or "_".join(col.words) in " ".join(words):
        score += 0.5
    for rank, hint in enumerate(hints[:6]):
        if hint in col_words or hint in {w for word in col_words for w in _expand(word)}:
            score += 0.6 - 0.05 * rank
    return score


def _candidate_tables(question: str, layer: Layer, retrieved: list[str],
                      value_hits: list[tuple[str, list]]) -> list[str]:
    """Tables worth planning against, ranked by four independent signals.

    Retrieval alone is not enough here. It ranks by aboutness, and a question
    like "how many transactions were flagged as structuring" is about
    `aml_fact_transactions` in a way the retriever gets right -- but "how many
    Denied claims" is pinned by the WORD `denied` living in exactly one column
    in the whole warehouse, which is a stronger signal than any embedding and a
    different KIND of evidence.

    This used to concatenate the sources in a fixed order and take the first
    ten, and the order was doing the ranking. "What is the total billed amount?"
    put `healthcare_ar_yield_predictions` -- the one table in 71 that has a
    `billed_amount` column, which the question names exactly -- at position
    ELEVEN, one past the cut, purely because it was reached by the weakest
    source. The question was refused as unanswerable while its answer sat in a
    column with the same name.

    So the sources are scored and merged instead:

      exact column name   1.6   `billed_amount` for "total billed amount": the
                                strongest signal that exists, because a column
                                whose every word the question said is not a
                                coincidence.
      retrieval rank      1.2 down to 0.2, from the hybrid retriever
      value binding       0.9   the question names a literal this table holds
      loose word overlap  0.15  each, the old fallback and still the weakest
    """
    words = content_words(question)
    expanded: set[str] = set()
    for word in words:
        expanded |= _expand(word)

    score: dict[str, float] = {}

    def bump(name: str, amount: float) -> None:
        if name in layer.tables:
            score[name] = score.get(name, 0.0) + amount

    for rank, name in enumerate(retrieved):
        bump(name, max(0.2, 1.2 - 0.1 * rank))
    for _phrase, bindings in value_hits:
        for binding in bindings:
            bump(binding.table, 0.9)
    for name, table in layer.tables.items():
        for col in table.columns:
            col_words = set(col.words)
            if col_words and col_words <= expanded:
                bump(name, 1.6)
                break
    # A counted entity points at a TABLE, and most strongly at the one whose own
    # name is that entity. `wholesale_dim_store` beats
    # `wholesale_fact_segment_month` for "how many stores" on this signal alone,
    # which is the difference between 620 stores and 170 store-months.
    entity = {f for word in entity_noun(question) for f in _expand(word)}
    if entity:
        for name, table in layer.tables.items():
            if set(table.words) & entity:
                bump(name, 2.0)
            # Stronger still: the table whose GRAIN is that entity is the one
            # that has exactly one row per one of them. "How many distinct
            # customers are there?" kept landing on
            # `wholesale_marketing_campaigns`, because its `new_customers`
            # column is unique and therefore looked like a customer key -- and
            # answered 4. `retail_dim_customer` is keyed on customer_id, 120
            # rows, one per customer. Grain is the difference between a column
            # that mentions the entity and a table that IS the entity.
            grain_words = {w for key in table.grain
                           for w in split_identifier(key)}
            if grain_words & entity:
                bump(name, 2.5)
    for word in expanded:
        for name in layer.tables_with_word(word):
            bump(name, 0.15)

    return [name for name, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def _pick_measure(layer: Layer, table: str, words: set[str], hints: list[str],
                  aggregate: str, *, required: list[str] | None = None
                  ) -> tuple[Column | None, float]:
    """The measure the question asked to aggregate.

    `required` is the measure half of "top X by Y" — when the question says
    which quantity it is ranking on, the chosen column MUST contain one of those
    words. Without that, "bottom 3 departments by average salary" bound
    `departments_supplied` on a supplier table: the column matched the AXIS word
    `departments`, scored 0.5, and answered a salary question with a count of
    supplier categories. Matching on any word in the sentence is not evidence
    that this is the measure; matching on the word that named the measure is.
    """
    required_forms: set[str] = set()
    for word in required or ():
        required_forms |= _expand(word)
    best: Column | None = None
    best_score = 0.0
    for col in layer.tables[table].columns:
        if col.role != MEASURE:
            continue
        if required_forms and not (set(col.words) & required_forms):
            continue
        if aggregate == SUM and not col.additive:
            # Summing a rate is the `share_out_of_range` finding waiting to
            # happen. Refusing to write it beats writing it and catching it.
            continue
        score = _score_column(col, words, hints)
        if score > best_score:
            best, best_score = col, score
    return best, best_score


def _pick_dimension(layer: Layer, table: str, words: set[str], hints: list[str],
                    ) -> tuple[Column | None, float, list[JoinEdge]]:
    """The best grouping column for this question, and how to reach it.

    Searches the base table first, then tables one non-fanning join away. "What
    is the average salary by department?" needs `base_salary` from
    `hr_fact_employees` and `department` from `hr_flight_risk_scores`, and a
    dimension search confined to the base table cannot see it -- it settled for
    `tenure_band` instead and grouped average salary by tenure, which is a
    coherent query and the wrong one.
    """
    best: Column | None = None
    best_score = 0.0
    best_path: list[JoinEdge] = []

    def consider(col: Column, path: list[JoinEdge], penalty: float) -> None:
        nonlocal best, best_score, best_path
        if col.role not in (DIMENSION, FLAG):
            return
        # A grouping column with a thousand values is a listing, not a summary.
        if col.distinct > 200:
            return
        score = _score_column(col, words, hints) - penalty
        if score > best_score:
            best, best_score, best_path = col, score, path

    for col in layer.tables[table].columns:
        consider(col, [], 0.0)
    for edge in layer.edges_for(table):
        other = edge.other(table)
        # Only join OUT to a table the key is unique in: joining to the many
        # side multiplies the base rows, and a grouped aggregate over a fan-out
        # is the `join_fanout` finding, not a breakdown.
        unique_on = edge.right if edge.unique_side == "right" else edge.left
        if unique_on != other:
            continue
        for col in layer.tables[other].columns:
            # A small penalty, so an equally-good column on the base table wins:
            # every join is a chance to be wrong about the relationship.
            consider(col, [edge], 0.05)
    return best, best_score, best_path


def _pick_label(layer: Layer, table: str, words: set[str],
                hints: list[str]) -> tuple[Column | None, float]:
    """A column that NAMES one row, for a table already at that grain.

    Uniqueness is the requirement, and it is what makes the rank shape safe: if
    `customer_name` is unique across `retail_customer_analytics`, then ordering
    by a measure and taking the first row is exactly "the top customer", with no
    aggregate to get wrong and no join to fan out.
    """
    best: Column | None = None
    best_score = 0.0
    for col in layer.tables[table].columns:
        if not col.unique:
            continue
        if col.type.upper().split("(")[0] != "VARCHAR":
            # A surrogate integer id is unique too, and answering "who is the
            # top customer" with `4417` is not answering it.
            continue
        score = _score_column(col, words, hints)
        if score > best_score:
            best, best_score = col, score
    return best, best_score


def _reachable(layer: Layer, base: str, table: str) -> list[JoinEdge] | None:
    if table == base:
        return []
    return layer.join_path(base, table)


def _fans_out(layer: Layer, base: str, joins: list[JoinEdge]) -> bool:
    """Would this join multiply the base table's rows?

    An edge is safe when the column is unique on the side being joined TO. If
    the planner cannot prove that, it treats the join as a fan-out and declines
    to build an additive aggregate over it -- the same defect `engine/verify.py`
    reports as `join_fanout`, refused at write time instead of caught at read
    time.
    """
    current = base
    for edge in joins:
        target = edge.other(current)
        unique_on = edge.right if edge.unique_side == "right" else edge.left
        if unique_on != target:
            return True
        current = target
    return False


def _build_filters(layer: Layer, base: str, joins: list[JoinEdge],
                   value_hits: list[tuple[str, list]],
                   question: str) -> tuple[list[Filter], list[JoinEdge], set[str]]:
    """Turn matched values and comparisons into WHERE clauses.

    Returns the filters, any extra joins they needed, and the question words
    they explain.
    """
    filters: list[Filter] = []
    extra: list[JoinEdge] = list(joins)
    explained: set[str] = set()
    in_scope = {base} | {t for edge in extra for t in (edge.left, edge.right)}

    for phrase, bindings in value_hits:
        # Prefer a binding on a table already in the query; only then pull one in.
        chosen = next((b for b in bindings if b.table in in_scope), None)
        path: list[JoinEdge] = []
        if chosen is None:
            # Smallest table first. `Medicare` is a value of payer_type in BOTH
            # healthcare_dim_payer (8 rows) and healthcare_ar_yield_predictions
            # (1,378 rows, and only the OPEN AR subset). Taking whichever came
            # first joined claims to the AR table, whose rows do not include
            # denied claims at all, and "how many denied claims for Medicare"
            # came back 0 where the answer is 152.
            #
            # Row count is the derived stand-in for "is this the attribute's
            # canonical home": a dimension table is small and complete, a
            # denormalised copy on a fact or a subset table is neither.
            for binding in sorted(bindings, key=lambda b: layer.tables[b.table].rows):
                candidate = _reachable(layer, base, binding.table)
                if candidate is not None:
                    chosen, path = binding, candidate
                    break
        if chosen is None:
            continue
        if any(f.column == chosen.column and f.table == chosen.table for f in filters):
            continue
        for edge in path:
            if edge not in extra:
                extra.append(edge)
                in_scope.update((edge.left, edge.right))
        in_scope.add(chosen.table)
        filters.append(Filter(table=chosen.table, column=chosen.column, op="=",
                              literal=chosen.literal, evidence=phrase))
        explained.update(tokenise(phrase))

    match = COMPARE_RE.search(question)
    if match:
        op = COMPARE_OPS[match.group(1).lower()]
        number = match.group(2).replace(",", "").replace("_", "")
        words = content_words(question)
        target, score = _pick_measure(layer, base, words, [], "")
        if target is not None and score > 0:
            filters.append(Filter(table=base, column=target.name, op=op,
                                  literal=number, evidence=match.group(0)))
            explained.update(tokenise(match.group(0)))

    year = YEAR_RE.search(question)
    if year:
        dates = [c for c in layer.tables[base].columns if c.role == DATE]
        if dates:
            value = year.group(1) + year.group(2)
            # The date the question named, when it named one.
            # healthcare_fact_claims carries service_date, submitted_date AND
            # adjudicated_date, and "claims SUBMITTED in 2024" is a question
            # about the second one; taking dates[0] silently answered about the
            # first.
            asked = content_words(question)
            column = next(
                (c for c in dates
                 if {w for w in asked if _expand(w) & set(c.words)}),
                dates[0])
            filters.append(Filter(
                table=base, column=column.name, op="=", literal=value,
                evidence=year.group(0),
                expr=f'EXTRACT(YEAR FROM "{base}"."{column.name}")'))
            explained.add(value)
            # The column's own words count as explained. "How many claims were
            # SUBMITTED in 2024?" chose `submitted_date` over `service_date`
            # precisely because the question said so, and then failed the
            # coverage gate for leaving `submitted` unaccounted for.
            explained.update(column.words)
    return filters, extra, explained


# ----------------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------------

def _agg_expr(plan: Plan) -> str:
    if plan.aggregate == SHARE and plan.share_filter is not None:
        clause = plan.share_filter.sql()
        return (f"ROUND(100.0 * COUNT(*) FILTER (WHERE {clause}) "
                f"/ NULLIF(COUNT(*), 0), 1)")
    if plan.aggregate == COUNT:
        return "COUNT(*)"
    if plan.aggregate == COUNT_DISTINCT and plan.measure is not None:
        return f"COUNT(DISTINCT {plan.measure.qualified})"
    if plan.measure is None:
        return "COUNT(*)"
    return f"{plan.aggregate.upper()}({plan.measure.qualified})"


def _alias(plan: Plan) -> str:
    if plan.aggregate == SHARE:
        col = plan.share_filter.column if plan.share_filter else "match"
        return f"{col}_pct"
    if plan.aggregate in (COUNT, COUNT_DISTINCT) or plan.measure is None:
        return "n"
    return f"{plan.aggregate}_{plan.measure.name}"


def compile_sql(plan: Plan) -> str:
    """Render a plan as SQL. Pure -- everything it needs is on the plan."""
    alias = _alias(plan)
    select: list[str] = []
    if plan.aggregate == RANK:
        # No aggregate and no GROUP BY: the rows already are the answer.
        select.append(f'{plan.label.qualified} AS "{plan.label.name}"')
        select.append(f'{plan.measure.qualified} AS "{plan.measure.name}"')
    else:
        if plan.group_by is not None:
            select.append(f'{plan.group_by.qualified} AS "{plan.group_by.name}"')
        select.append(f'{_agg_expr(plan)} AS "{alias}"')

    lines = [f"SELECT {', '.join(select)}", f'FROM "{plan.base}"']
    joined = {plan.base}
    for edge in plan.joins:
        target = edge.right if edge.left in joined else edge.left
        if target in joined:
            continue
        left = edge.left if edge.left in joined else edge.right
        lines.append(f'JOIN "{target}" ON "{left}"."{edge.column}" '
                     f'= "{target}"."{edge.column}"')
        joined.add(target)

    where = [f.sql() for f in plan.filters]
    if where:
        lines.append("WHERE " + " AND ".join(where))
    if plan.aggregate == RANK:
        # NULLS LAST so an unmeasured row cannot win a "highest" question by
        # sorting above every real value -- DuckDB puts NULL first on DESC.
        lines.append(f'ORDER BY {plan.measure.qualified} '
                     f'{(plan.order or "desc").upper()} NULLS LAST')
        lines.append(f"LIMIT {plan.limit or 1}")
    elif plan.group_by is not None:
        lines.append("GROUP BY 1")
        if plan.order:
            lines.append(f'ORDER BY "{alias}" {plan.order.upper()}')
        if plan.limit:
            lines.append(f"LIMIT {plan.limit}")
    elif plan.limit:
        lines.append(f"LIMIT {plan.limit}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The planner
# ----------------------------------------------------------------------------

def _bind_candidate(question: str, layer: Layer, table: str, *,
                    aggregate: str, order: str, limit: int | None,
                    hints: list[str], value_hits: list[tuple[str, list]],
                    words: set[str], accountable: set[str],
                    measure_hints: list[str] | None = None,
                    rank: int | None = None) -> Plan | None:
    """Build the one best plan for a single candidate table, or None.

    Returning None is not a failure path, it is the common case: most of the ten
    candidate tables cannot support the question, and the ones that cannot
    should drop out here rather than survive on a structure bonus and win by
    default. That is exactly what went wrong the first time this ran -- a
    wholesale rate column answered a healthcare denial-rate question, because
    nothing required the winning table to have been mentioned.
    """
    # Measure hints are REQUIRED when present, not merely preferred: the
    # question said which quantity it wanted.
    measure, m_score = _pick_measure(
        layer, table, words, measure_hints or hints,
        aggregate or (SUM if order else ""),
        required=[h for h in (measure_hints or []) if h not in _AGGREGATE_WORDS])
    filters, joins, filter_words = _build_filters(layer, table, [], value_hits, question)
    # Same rule for boolean columns: "how many stores by store format" must not
    # quietly become "how many PHYSICAL stores" because `is_physical_store`
    # shares the word `store` with the axis the question named.
    # Neither an axis word nor a counted entity may switch a boolean on. "How
    # many stores do we have?" found `is_physical_store`, filtered to TRUE, and
    # answered 555 of 620 -- a quietly narrowed question, which is the hardest
    # kind of wrong answer to notice.
    # Expanded, because `is_physical_store` offers the evidence word `store`
    # and the question said `stores`. Matched raw, the plural slipped past and
    # "how many stores do we have?" answered 555 of 620.
    _reserved: set[str] = set()
    for word in list(hints) + [entity_head(question)]:
        if word:
            _reserved |= _expand(word)
    filters.extend(f for f in flag_filters(question, layer, table)
                   if not (set(tokenise(f.evidence)) & _reserved))
    filter_words |= {w for f in filters for w in tokenise(f.evidence)}

    # Grouping only when the question asked for it. Without hints there is no
    # dimension, which is what turns "how many active employees" from a two-row
    # breakdown into the single number it asked for.
    dimension, d_score, dim_path = (None, 0.0, [])
    if hints:
        dimension, d_score, dim_path = _pick_dimension(layer, table, words, hints)
    if dimension is not None:
        # An axis and a filter on the SAME column contradict each other: group
        # by a column already pinned to one value and you get exactly one group.
        # When the question named the axis, the axis wins and the filter goes.
        #
        # "How many models by resource type?" is the case that forced this. The
        # entity noun `models` matched a VALUE of `resource_type` -- the very
        # column `by resource type` asked to group on -- so the plan filtered to
        # resource_type = 'model', dropped the axis as already-pinned, and then
        # refused for having no axis. Dropping the filter instead gives the
        # breakdown that was asked for, and `models` is still explained, because
        # it is the table's own name.
        clash = [f for f in filters
                 if f.column == dimension.name and f.table == dimension.table]
        for f in clash:
            filters.remove(f)
    if dimension is not None:
        # The axis has to be the one that was ASKED for. Without this, "average
        # risk score by channel" grouped by `risk_band` -- a real dimension of
        # the right table, scoring on the word `risk` from elsewhere in the
        # sentence, in a table that has no channel at all. Grouping by something
        # the question never named is not a partial answer, it is a different
        # question answered confidently.
        hint_forms = {f for hint in hints for f in _expand(hint)}
        if not (set(dimension.words) & hint_forms):
            dimension, d_score, dim_path = None, 0.0, []
    for edge in dim_path:
        if edge not in joins:
            joins.append(edge)

    chosen = aggregate
    share_filter: Filter | None = None
    if chosen == SHARE:
        local = [f for f in filters if f.table == table]
        if len(local) > 1:
            # A share needs exactly one thing to be a share OF, and this
            # question named several. "What percentage of the still-open data
            # queries have been outstanding for 60 days or more?" matches
            # `status = open`, `severity = query` AND `age_band = 60+ days`, and
            # the numerator is only one of them -- the other is the population.
            # Picking the first match got 44.4% where the contract says 31.3%,
            # which is a wrong answer produced by a coin toss.
            #
            # Word order would resolve it ("percentage of X ... are Y"), but
            # only for questions phrased that way, and a rule that works on the
            # phrasings I happened to test is how a planner starts guessing.
            # Ambiguity the grammar cannot resolve is a refusal.
            return None
        share_filter = local[0] if local else None
        if share_filter is not None:
            filters = [f for f in filters if f is not share_filter]
        elif measure is not None and not measure.additive and m_score > 0:
            # The schema already carries the rate as a column and the question
            # named it ("what is the fill rate" -> `fill_rate_pct`). Averaging a
            # stored rate is a different query from computing one, so this
            # branch requires the question to have actually said the column's
            # words -- m_score > 0 is what stops it grabbing any rate anywhere.
            chosen = AVG
        else:
            # A rate with nothing to be a rate OF. Refusing beats inventing a
            # numerator.
            return None
    # An entity noun is the thing being counted, so it must never be the thing
    # being summed. "list the departments" bound `departments_supplied` and
    # answered 341; "break down transactions by channel" summed `txn_count_7d`,
    # a rolling-window feature, instead of counting rows.
    head = entity_head(question)
    entity = _expand(head) if head else set()
    if entity and measure is not None and (set(measure.words) & entity):
        measure, m_score = None, 0.0
    if entity and not chosen:
        chosen = COUNT
    if (not chosen and order and dimension is None
            and measure is not None and m_score > 0):
        # "What is the lowest risk score?" names no aggregate verb, only a
        # direction. With no axis to rank on, the direction IS the aggregate --
        # and MIN/MAX were defined in this module and never once selected, so
        # the question fell through to COUNT and then failed the coverage gate.
        #
        # `dimension is None` is load-bearing. Without it "top 5 customers by
        # revenue" read the same way and became MAX(revenue) per customer --
        # the biggest single sale, $3,658, in place of the biggest customer,
        # $182,503. With an axis present the direction is the ORDER BY, and the
        # aggregate is whatever the measure supports.
        chosen = MAX if order == "desc" else MIN
    if not chosen:
        chosen = SUM if (measure is not None and measure.additive and m_score > 0) else COUNT
    if chosen in (SUM, AVG, MIN, MAX) and (measure is None or m_score <= 0):
        # No silent downgrade to COUNT. "How much of the open AR do we expect to
        # collect" asked for a sum of dollars; answering with the number of rows
        # is not a rougher answer to that question, it is a confident answer to
        # a different one. If the question named an aggregate and no measure
        # binds, that is a refusal.
        if aggregate in (SUM, AVG, MIN, MAX):
            return None
        chosen = COUNT
    if chosen == COUNT_DISTINCT:
        # "How many distinct customers are there?" must count a column that
        # means customer. Falling back to `keys[0]` counted
        # `wholesale_marketing_campaigns.new_customers` -- a campaign metric
        # that happens to be unique -- and answered 4 where the answer is 120.
        # If nothing in the table names the entity, this is not the table.
        target = measure if m_score > 0 else None
        if target is None:
            wanted: set[str] = set()
            for word in entity_noun(question) or list(words):
                wanted |= _expand(word)
            keys = [c for c in layer.tables[table].columns
                    if c.role == KEY and (set(c.words) & wanted)]
            target = keys[0] if keys else None
        measure = target
        if measure is None:
            return None

    plan_limit = limit
    plan_order = order
    if dimension is not None and order and limit is None:
        plan_limit = 1                      # "which X has the most Y" -> top 1
    if dimension is not None and plan_limit and not plan_order:
        plan_order = "desc"
    if dimension is None:
        # Ordering a single aggregate row is noise, and a LIMIT on it is worse:
        # "top 5 departments" with no dimension bound would quietly become "the
        # one grand total, capped at five rows" -- a real answer to a question
        # nobody asked. If the question wanted a ranking and no dimension could
        # be found, that is a refusal.
        plan_order, plan_limit = "", None
        if limit is not None:
            return None

    if _fans_out(layer, table, joins) and chosen in (SUM, AVG):
        return None

    # A table already at the question's grain is ranked, not grouped.
    #
    # The fan-out guard above only covers SUM and AVG, because those are the
    # aggregates it was written for. RANK needs it too and for a subtler reason:
    # there is no aggregate to inflate, but a join to the many side duplicates
    # the base rows, so the "one row per customer" property that makes ranking
    # correct at all stops holding, and the top row can be a duplicate rather
    # than the maximum.
    if (dimension is None and expects_category(question) and order
            and measure is not None and not _fans_out(layer, table, joins)):
        label, l_score = _pick_label(layer, table, words, hints)
        if label is not None and l_score > 0 and m_score > 0:
            ranked = Plan(base=table, aggregate=RANK, joins=joins, measure=measure,
                          label=label, filters=filters, order=order,
                          limit=limit or 1)
            return _finish(ranked, question=question, layer=layer, table=table,
                           words=words, accountable=accountable, hints=hints,
                           aggregate=aggregate, m_score=m_score, d_score=l_score,
                           filter_words=filter_words | set(tokenise(label.name)),
                           share_filter=None, rank=rank)

    # Shape before content: a question that asked which THING must come back
    # with a thing, and one that asked for a number must not come back with a
    # breakdown. Getting this wrong is not a worse answer, it is an answer to a
    # different question.
    if dimension is None and expects_category(question):
        return None
    if dimension is not None and not (hints or expects_category(question)):
        return None

    plan = Plan(base=table, aggregate=chosen, joins=joins, measure=measure,
                group_by=dimension, filters=filters, share_filter=share_filter,
                order=plan_order, limit=plan_limit)
    return _finish(plan, question=question, layer=layer, table=table, words=words,
                   accountable=accountable, hints=hints, aggregate=aggregate,
                   m_score=m_score, d_score=d_score, filter_words=filter_words,
                   share_filter=share_filter, rank=rank)


def _finish(plan: Plan, *, question: str, layer: Layer, table: str, words: set[str],
            accountable: set[str], hints: list[str], aggregate: str,
            m_score: float, d_score: float, filter_words: set[str],
            share_filter, rank: int | None) -> Plan | None:
    """Score a built plan, or reject it. Shared by every plan shape."""
    measure = plan.measure
    dimension = plan.group_by or plan.label
    plan_order = plan.order
    filters = plan.filters
    if share_filter is not None:
        filter_words |= set(tokenise(share_filter.evidence))

    # ---- what the question bought ------------------------------------------
    table_words = set(split_identifier(table)) | set(
        split_identifier(layer.tables[table].domain))
    explained = set(filter_words)
    explained |= {w for w in words if w in table_words or (_expand(w) & table_words)}
    if measure is not None and m_score > 0:
        explained |= {w for w in words
                      if w in set(measure.words) or (_expand(w) & set(measure.words))}
    if dimension is not None and d_score > 0:
        explained |= {w for w in words
                      if w in set(dimension.words) or (_expand(w) & set(dimension.words))}
    if plan_order:
        explained |= {w for w in words if w in _SUPERLATIVE_WORDS}
    if aggregate:
        explained |= {w for w in words if w in _AGGREGATE_WORDS}
    explained &= words
    plan.explained = explained
    plan.unexplained = words - explained

    # A named axis is an instruction, not a hint. "Average risk score by
    # channel" against a table with no channel column used to drop the GROUP BY
    # and return one number -- the average across everything -- which is a
    # confident answer to a question nobody asked.
    #
    # The rule is "at least one axis word binds", not "all of them". An axis
    # arrives as a noun PHRASE and English hangs modifiers off the head noun:
    # "the top wholesale customer" has the head `customer`, and demanding that
    # `wholesale` bind too refused the question outright. The unbound modifier
    # is not forgiven, it just is not fatal -- it still costs coverage below,
    # which is the proportionate place for it.
    binding_hints = [h for h in hints if h in accountable]
    if binding_hints and not (set(binding_hints) & explained):
        return None
    if demands_axis(question):
        axis = plan.group_by or plan.label
        hint_forms = {f for hint in hints for f in _expand(hint)}
        if axis is None or not (set(axis.words) & hint_forms):
            return None

    # ---- and whether the table itself was ever earned -----------------------
    # A plan whose table nothing in the question points at is a coincidence,
    # however good its column match looks. This is the guard that was missing
    # when a wholesale store's fill_rate_pct answered "what is the claim denial
    # rate": every other signal was weak, and weak-times-weak still won because
    # nothing was checking that the table had been asked for.
    # Expanded, not raw. "How many payers are there?" has the single content
    # word `payers`, the table is `healthcare_dim_payer`, and a raw set
    # intersection misses it on the plural -- so the one table that could answer
    # the question was rejected as unanchored while `explained` had already
    # credited the very same word.
    anchored = bool(
        {w for w in words if w in table_words or (_expand(w) & table_words)}
        or any(f.table == table for f in filters)
        or (share_filter is not None)
        or (measure is not None and m_score > 0)
        or (dimension is not None and d_score > 0)
    )
    if not anchored:
        return None

    plan.coverage = round(len(explained & accountable) / max(len(accountable), 1), 3)
    # Structure and retrieval rank pick BETWEEN plans; they never lift one over
    # the floor. Keeping them out of the gate is the difference between 26 wrong
    # answers and none -- see the note on MIN_COVERAGE.
    structure = 0.0
    if measure is not None and m_score > 0:
        structure += 0.10
    if dimension is not None and d_score > 0:
        structure += 0.10
    structure += 0.06 * min(len(filters) + (1 if share_filter else 0), 2)
    if rank is not None:
        # The hybrid retriever is the repo's measured answer to "which tables is
        # this question about" (100% recall at k=10). Its ranking is evidence,
        # so a plan built on its top hit outranks an equally-worded plan built
        # on its ninth -- which is what separates the supply-chain fill rate
        # from a wholesale supplier's stored one.
        structure += max(0.0, 0.18 - 0.02 * rank)
    plan.confidence = round(min(1.0, plan.coverage + structure), 3)
    plan.sql = compile_sql(plan)
    return plan


_AGGREGATE_WORDS = {word for phrase in
                    COUNT_PHRASES + SUM_PHRASES + AVG_PHRASES + SHARE_PHRASES
                    + DISTINCT_PHRASES
                    for word in phrase.split()}

_SUPERLATIVE_WORDS = set(MAX_PHRASES) | set(MIN_PHRASES)
_DISTINCT_WORDS = {w for phrase in DISTINCT_PHRASES for w in phrase.split()}


def plan_question(question: str, layer: Layer, *,
                  retrieved: list[str] | None = None,
                  min_confidence: float = MIN_CONFIDENCE) -> PlanResult:
    """Compile a question, or refuse and say what could not be bound."""
    import time

    started = time.perf_counter()
    words = content_words(question)
    if not words:
        return PlanResult(question=question, refused=True,
                          reason="there is nothing in that question to bind to a table.")

    if is_listing_request(question):
        return PlanResult(
            question=question, refused=True,
            reason=("I compute aggregates, not listings — every query this "
                    "grammar writes ends in a COUNT, SUM, AVG, share or "
                    "ranking. Ask for a count, a total, or a top-N and I can "
                    "compile it."))
    named = unsupported_aggregate(question)
    if named:
        return PlanResult(
            question=question, refused=True,
            reason=(f"I have no way to compute a {named} — this grammar writes "
                    "counts, sums, averages, shares and rankings, and nothing "
                    "that needs a window function. Answering with a different "
                    "statistic would be worse than saying so."))
    aggregate, _agg_words = detect_aggregate(question)
    order = detect_order(question)
    limit = detect_limit(question)
    hints, measure_hints = axis_and_measure_hints(question)
    value_hits = match_values(question, layer, exclude=set(hints))
    candidates = _candidate_tables(question, layer, retrieved or [], value_hits)
    if not candidates:
        return PlanResult(
            question=question, refused=True,
            reason="no table in the warehouse matches any word in that question.",
            elapsed_ms=1000 * (time.perf_counter() - started))

    unbound = unbindable_words(question, layer)
    accountable = words - unbound
    # Excusing unbindable words fixes the denominator, but left alone it also
    # hands a free pass to questions this warehouse simply cannot answer: "how
    # many employees left voluntarily" excused BOTH `left` and `voluntarily`,
    # leaving `employees` as the entire accountable set, which a bare COUNT(*)
    # covered perfectly -- 1,900 employees returned as the voluntary-attrition
    # figure of 304. So the excuse itself is capped. Past a third of the
    # question, the right conclusion is not "well covered", it is "this is not
    # a question about this data".
    # At least HALF the question's content words must be bindable. The first
    # version of this cap was `len(unbound) > max(1, len(words) // 3)`, which on
    # a two-word question allows one unbound word -- so "what is the total
    # flurble?" kept `total`, matched `aml_cases.total_amount_cad` on it, and
    # returned a sum of Canadian dollars for a word that does not exist. A
    # proportion needs to stay a proportion at every length.
    if len(unbound) * 2 >= len(words):
        return PlanResult(
            question=question, refused=True, unbound=unbound,
            reason=("too much of that question has no counterpart in the "
                    "warehouse (" + ", ".join(sorted(unbound)[:5]) + "). "
                    "I would be answering a simpler question than the one you "
                    "asked."))
    if not accountable:
        return PlanResult(
            question=question, refused=True, unbound=unbound,
            reason="none of the words in that question appear anywhere in this "
                   "warehouse -- not in a table name, a column, a value, or a "
                   "domain description.")
    ranks = {name: i for i, name in enumerate(retrieved or [])}
    # TWO winners are tracked, and the reason is a bug this loop had for its
    # whole first life: plans were selected by `confidence` and then gated on
    # `coverage`. Those are different orderings, so a plan with a big structure
    # bonus and mediocre coverage could beat a plan with PERFECT coverage, and
    # then fail the gate on the winner's behalf -- taking the right answer down
    # with it. "How many models by resource type?" was refused that way while
    # `dbt_models` sat in the candidate list at position zero with coverage 1.0.
    #
    # So the gate is applied per plan, before the comparison. `best` is the best
    # plan that PASSES; `nearest` is the highest-coverage plan overall and exists
    # only to explain the refusal when nothing passes.
    best: Plan | None = None
    nearest: Plan | None = None
    considered = 0
    for table in candidates[:CANDIDATE_DEPTH]:
        considered += 1
        plan = _bind_candidate(question, layer, table, aggregate=aggregate, order=order,
                               limit=limit, hints=hints, value_hits=value_hits,
                               words=words, accountable=accountable,
                               measure_hints=measure_hints, rank=ranks.get(table))
        if plan is None:
            continue
        if nearest is None or plan.coverage > nearest.coverage:
            nearest = plan
        if plan.coverage < min_confidence:
            continue
        if best is None or plan.confidence > best.confidence:
            best = plan

    elapsed = 1000 * (time.perf_counter() - started)
    if best is None and nearest is not None:
        missed = ", ".join(sorted(nearest.unexplained)[:5])
        return PlanResult(
            question=question, plan=nearest, refused=True, candidates=candidates[:6],
            considered=considered, elapsed_ms=elapsed, unbound=unbound,
            reason=(f"I can see {nearest.base}, but too much of the question is "
                    f"unaccounted for" + (f" ({missed})" if missed else "")
                    + ". Naming a column or a value the warehouse uses would let me "
                      "compile it; otherwise this is a question for the model."))
    if best is None:
        return PlanResult(
            question=question, refused=True, candidates=candidates[:6],
            considered=considered, elapsed_ms=elapsed, unbound=unbound,
            reason="I could not build a query for that from the loaded tables "
                   "without guessing at what it means.")
    return PlanResult(question=question, plan=best, candidates=candidates[:6],
                      considered=considered, elapsed_ms=elapsed, unbound=unbound)
