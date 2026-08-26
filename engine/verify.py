"""
Catch SQL that runs and is wrong.

`sql_guard` proves the model's SQL is read-only; `query.run_query` executes it and
reports the crash if it crashes. Between those two lies the failure that actually
matters for an analytics assistant: a query that is syntactically fine, binds
cleanly, executes in three milliseconds, returns rows, and is *meaningless*. The
assistant then states its number with exactly the same confidence it uses for a
correct one. A confident wrong number is worse than an error, because an error is
visibly an error.

This module is the layer that looks for those. Everything here is deterministic,
runs offline, and needs no API key — which is also why it can be measured rather
than asserted.

WHY THE OBVIOUS CHECKS ARE NOT HERE
Two candidates were tried and dropped, both because the database already does the
job:

  * *Static schema validation before execution.* DuckDB's `EXPLAIN` binds without
    executing, so a bad table or column can be caught pre-flight. Measured on the
    39 golden queries it costs 0.69 ms each (~50% of the 1.37 ms it costs to just
    run them) and catches nothing execution would not catch a moment later. The
    real prize was never the catch, it was the *message* — so it lives in
    `explain_error()`, which enriches the error DuckDB already produced, at zero
    cost on the happy path.
  * *Aggregation without grouping.* DuckDB rejects it outright:
    `SELECT payer_id, COUNT(*) FROM healthcare_fact_claims` is a Binder Error, and
    so is a GROUP BY that omits a projected column. There is nothing to add.

WHAT IS HERE, AND WHY EACH EARNED IT

  cross_domain_join   The 11 domains are independent synthetic datasets that share
                      no keys. Joining across them is meaningless — and it does
                      not announce itself. Measured: `hr_fact_employees` joined to
                      `finance_erp_gl` on employee_id = account_id returns **5,400
                      rows**. It does not error and it is not empty, so neither the
                      retry loop nor a result-sanity check can see it. Only the
                      static structure can. 0 false positives on the 39 golden
                      queries.

  join_fanout         The classic silent-wrong analytics answer: SUM over a table
                      on the *one* side of a one-to-many join, which repeats each
                      value once per match. Measured: SUM(completed) over
                      `clinical_subjects` is 96; LEFT JOIN it to
                      `clinical_query_log` and the same SUM reads 108. Nothing
                      errors. 0 false positives on the 39 golden queries — the
                      composite-key handling is what keeps
                      `injected_defect_detection_rate` (a five-column join whose
                      individual columns are all non-unique) off the list.

  empty_result        Zero rows is not automatically a bug, but it is never
                      *self-explanatory*, and the assistant currently answers
                      "nothing matched" without saying why. When a query came back
                      empty and it joins two tables on an equality, one INTERSECT
                      per join key (measured 19.8 ms on the largest pair in this
                      warehouse, 50k x 100k rows) says whether the keys overlap at
                      all. Only runs on an empty result, so the happy path pays
                      nothing.

  null_scalar         A single row, a single column, and it is NULL. Almost always
                      a division by an empty filter or an aggregate over no rows,
                      and it renders as a blank the model will narrate around.

  share_out_of_range  A value the SQL built as `100.0 * x / y` that lands outside
                      [0, 100]. Advisory only, and deliberately narrow: the naive
                      form of this check has a *measured* 2.6% false-positive rate
                      on the golden set, because `wholesale_fy2025_revenue_change`
                      correctly returns -4.52. Percentage *changes* are not
                      percentage *shares*, so change wording is excluded, which
                      takes it to 0%.

  ambiguous_entity    Four base names exist in more than one domain (dim_customer
                      in 3, dim_product in 4, dim_supplier in 2, fact_orders in 3),
                      so "the top customer" genuinely does not name a table. This
                      does NOT ask a clarifying question — see the note on
                      `severity` below. It discloses which domain the answer came
                      from, so an ambiguity the assistant silently resolved becomes
                      something the reader can see and correct.

SEVERITY IS A COST DECISION, NOT A CONFIDENCE ONE

  error   Structurally meaningless. Never show it; hand it back to the correction
          loop. Only `cross_domain_join` is an error.
  warn    Probably wrong, cheaply checked, but heuristic. Worth one correction
          attempt, not three — a false positive must cost one extra model call,
          never the whole retry budget and never a refusal.
  note    Never blocks and never retries. It annotates the answer.

That ladder is the point. A check that flags good SQL is worse than no check, so
the ones that could be wrong are wired so that being wrong is cheap.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_manifest import MANIFEST, table_name  # noqa: E402

# table -> domain, for the 71 warehouse tables.
DOMAIN_OF: dict[str, str] = {
    table_name(domain, table): domain for domain, table, _source, _desc in MANIFEST
}

ERROR, WARN, NOTE = "error", "warn", "note"

# Aggregates whose value changes when rows are duplicated. COUNT(DISTINCT ...) is
# excluded at the call site because de-duplication is exactly what makes it safe.
ADDITIVE_AGGREGATES = {"sum", "avg", "mean", "count", "count_star", "total"}

# Wording that means "difference between two periods", where a percentage is not
# bounded by [0, 100] and may legitimately be negative.
_CHANGE_WORDS = re.compile(
    r"chang|delta|differen|growth|yoy|year[ _-]?over|versus|\bvs\b|gap|spread|swing|lift",
    re.IGNORECASE,
)
_SHARE_ALIAS = re.compile(r"(?:^|_)(rate|pct|percent|percentage|share|ratio)(?:$|_)",
                          re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    """One thing that looks wrong about a query or its result."""
    check: str
    severity: str
    message: str

    @property
    def blocking(self) -> bool:
        return self.severity == ERROR

    def __str__(self) -> str:
        return f"[{self.check}] {self.message}"


def worst(findings: list[Finding]) -> str | None:
    """The highest severity present, or None."""
    for level in (ERROR, WARN, NOTE):
        if any(f.severity == level for f in findings):
            return level
    return None


def correction_message(findings: list[Finding]) -> str:
    """The text handed back to the model when verification rejects a query.

    Phrased as a description of what is wrong plus the constraint that fixes it,
    because "that is invalid" gives a self-correcting model nothing to act on.
    """
    lines = [f"- {f.message}" for f in findings if f.severity in (ERROR, WARN)]
    return (
        "That query ran, but the result cannot be trusted:\n"
        + "\n".join(lines)
        + "\nWrite a corrected single SELECT query that avoids this. If the "
          "question genuinely cannot be answered from one domain's tables, call "
          "cannot_answer instead of joining across domains."
    )


# --------------------------------------------------------------------------
# Parsing. DuckDB's own parser, not a regex.
# --------------------------------------------------------------------------

def parse_sql(con, sql: str) -> dict | None:
    """The statement's AST as plain dicts, or None if it will not parse.

    `json_serialize_sql` is DuckDB parsing its own dialect, so CTEs, UNPIVOT,
    FILTER clauses and window functions all come back structured. Writing a
    regex that pretends to understand this SQL was the alternative, and it would
    have been wrong on the golden set's own queries — five of them nest CTEs.
    """
    try:
        raw = con.execute("SELECT json_serialize_sql(?)",
                          [(sql or "").strip().rstrip(";").strip()]).fetchone()[0]
        ast = json.loads(raw)
    except Exception:
        return None
    return None if not isinstance(ast, dict) or ast.get("error") else ast


def _walk(node):
    """Every dict in the tree, in no particular order."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _cte_names(ast) -> set[str]:
    """Names bound by WITH. They appear as BASE_TABLE nodes and are not tables."""
    names = set()
    for node in _walk(ast):
        cte_map = node.get("cte_map")
        if isinstance(cte_map, dict):
            for entry in cte_map.get("map") or []:
                key = entry.get("key")
                if key:
                    names.add(str(key).lower())
    return names


def _relations(ast) -> tuple[dict[str, str], set[str]]:
    """(alias -> real table, {real tables}) for warehouse tables in the query."""
    ctes = _cte_names(ast)
    binding: dict[str, str] = {}
    tables: set[str] = set()
    for node in _walk(ast):
        if node.get("type") != "BASE_TABLE":
            continue
        name = (node.get("table_name") or "").lower()
        if not name or name in ctes or name not in DOMAIN_OF:
            continue
        alias = (node.get("alias") or "").lower() or name
        binding[alias] = name
        binding.setdefault(name, name)
        tables.add(name)
    return binding, tables


def _equalities(node) -> list[tuple[list, list]]:
    """Every `colref = colref` under `node`, as (left names, right names)."""
    out = []
    for candidate in _walk(node):
        if candidate.get("class") != "COMPARISON" or candidate.get("type") != "COMPARE_EQUAL":
            continue
        left, right = candidate.get("left"), candidate.get("right")
        if (isinstance(left, dict) and isinstance(right, dict)
                and left.get("class") == "COLUMN_REF" and right.get("class") == "COLUMN_REF"):
            out.append((left.get("column_names") or [], right.get("column_names") or []))
    return out


# --------------------------------------------------------------------------
# The verifier.
# --------------------------------------------------------------------------

class Verifier:
    """Deterministic checks over model-written SQL and its result.

    Holds two caches keyed on (table, columns): the column list of each table and
    whether a key is unique. Both describe the warehouse, which is rebuilt from
    CSV at startup and never written to, so they are valid for the process's life.
    """

    def __init__(self, con):
        self.con = con
        self._columns: dict[str, set[str]] = {}
        self._unique: dict[tuple, bool] = {}
        self._ambiguous_bases = _ambiguous_base_names()

    # ---- catalog helpers ------------------------------------------------

    def columns_of(self, table: str) -> set[str]:
        if table not in self._columns:
            try:
                rows = self.con.execute(f'DESCRIBE "{table}"').fetchall()
                self._columns[table] = {str(r[0]).lower() for r in rows}
            except Exception:
                self._columns[table] = set()
        return self._columns[table]

    def tables_with_column(self, column: str) -> list[str]:
        """Every warehouse table carrying this column name, catalogue order."""
        column = column.lower()
        return [t for t in DOMAIN_OF if column in self.columns_of(t)]

    def _is_unique(self, table: str, columns: list[str]) -> bool:
        """Is this (possibly composite) key unique on this table?

        NULLs are not excluded: COUNT(DISTINCT) already ignores them, so a key
        with NULLs reads as non-unique, which is the conservative direction for a
        fan-out check.
        """
        key = (table, tuple(sorted(columns)))
        if key not in self._unique:
            try:
                selected = ", ".join(f'"{c}"' for c in columns)
                total, distinct = self.con.execute(
                    f'SELECT COUNT(*), COUNT(DISTINCT ({selected})) FROM "{table}"'
                ).fetchone()
                self._unique[key] = (total == distinct)
            except Exception:
                self._unique[key] = True   # unknown: do not invent a finding
        return self._unique[key]

    def _resolve(self, names, binding, tables) -> tuple[str | None, str]:
        """['c', 'payer_id'] -> ('healthcare_fact_claims', 'payer_id').

        An unqualified column resolves only when exactly one table in the query
        owns that name; ambiguity resolves to None rather than to a guess.
        """
        column = str(names[-1]).lower() if names else ""
        if len(names) >= 2:
            return binding.get(str(names[-2]).lower()), column
        owners = [t for t in tables if column in self.columns_of(t)]
        return (owners[0] if len(owners) == 1 else None), column

    # ---- pre-execution --------------------------------------------------

    def check_sql(self, sql: str) -> list[Finding]:
        """Structural checks. Parse plus, for fan-out only, cached key probes.

        Measured on the 39 golden queries: 0 findings, ~0.5 ms each.
        """
        ast = parse_sql(self.con, sql)
        if ast is None:
            # Unparseable SQL is the executor's problem, not this module's.
            return []
        binding, tables = _relations(ast)
        if len(tables) < 2:
            return []
        findings = self._cross_domain(ast, binding, tables)
        findings += self._fanout(ast, binding, tables)
        return findings

    def _cross_domain(self, ast, binding, tables) -> list[Finding]:
        findings, seen = [], set()
        for left_names, right_names in _equalities(ast):
            left, left_col = self._resolve(left_names, binding, tables)
            right, right_col = self._resolve(right_names, binding, tables)
            if not left or not right or left == right:
                continue
            if DOMAIN_OF[left] == DOMAIN_OF[right]:
                continue
            pair = tuple(sorted((left, right)))
            if pair in seen:
                continue
            seen.add(pair)
            findings.append(Finding(
                "cross_domain_join", ERROR,
                f"{left} (domain: {DOMAIN_OF[left]}) is joined to {right} "
                f"(domain: {DOMAIN_OF[right]}) on {left_col} = {right_col}. These are "
                f"independent datasets that share no identifiers, so the matched rows "
                f"are coincidental and any number computed from them is meaningless. "
                f"Answer from one domain."))
        if findings:
            return findings
        domains = sorted({DOMAIN_OF[t] for t in tables})
        if len(domains) > 1:
            findings.append(Finding(
                "cross_domain_reference", NOTE,
                f"this query reads {len(domains)} unrelated domains ({', '.join(domains)}) "
                f"without joining them; make sure the comparison is meant to be "
                f"side-by-side rather than combined."))
        return findings

    def _aggregate_sources(self, ast, binding, tables) -> dict[str, set[str]]:
        """{table -> {'sum(paid_amount)', ...}} for row-count-sensitive aggregates."""
        sources: dict[str, set[str]] = {}
        for node in _walk(ast):
            if node.get("class") != "FUNCTION":
                continue
            name = (node.get("function_name") or "").lower()
            if name not in ADDITIVE_AGGREGATES or node.get("distinct"):
                continue
            for child in _walk(node.get("children") or []):
                if child.get("class") != "COLUMN_REF":
                    continue
                table, column = self._resolve(child.get("column_names") or [], binding, tables)
                if table:
                    sources.setdefault(table, set()).add(f"{name}({column})")
        return sources

    def _fanout(self, ast, binding, tables) -> list[Finding]:
        sources = self._aggregate_sources(ast, binding, tables)
        if not sources:
            return []
        findings, seen = [], set()
        for node in _walk(ast):
            if node.get("type") != "JOIN" or not isinstance(node.get("condition"), dict):
                continue
            # One join can equate several column pairs. They are a COMPOSITE key
            # and must be tested together: clinical_injected_defects joins
            # clinical_query_log on five columns, none of which is unique alone
            # while the tuple is. Testing them one at a time reports a fan-out
            # that does not exist.
            composite: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for left_names, right_names in _equalities(node["condition"]):
                left, left_col = self._resolve(left_names, binding, tables)
                right, right_col = self._resolve(right_names, binding, tables)
                if not left or not right or left == right:
                    continue
                composite.setdefault((left, right), []).append((left_col, right_col))
            for (left, right), pairs in composite.items():
                for source, other, other_key in (
                    (left, right, [p[1] for p in pairs]),
                    (right, left, [p[0] for p in pairs]),
                ):
                    if source not in sources or (source, other) in seen:
                        continue
                    if self._is_unique(other, other_key):
                        continue
                    seen.add((source, other))
                    aggregates = ", ".join(sorted(sources[source]))
                    findings.append(Finding(
                        "join_fanout", WARN,
                        f"{aggregates} reads {source}, but the join key "
                        f"({', '.join(other_key)}) is not unique on {other}, so every "
                        f"{source} row is repeated once per matching {other} row and the "
                        f"aggregate is inflated. Aggregate {source} in a subquery before "
                        f"joining, or aggregate the column that lives on {other}."))
        return findings

    # ---- post-execution -------------------------------------------------

    def check_result(self, sql: str, result, question: str = "") -> list[Finding]:
        """Sanity checks on what came back. Cheap; the expensive one is gated on
        an empty result, which the happy path never hits.

        `question` is optional and only narrows `share_out_of_range`. It is worth
        passing: without it that check reduces to the structural rule alone.
        """
        if result is None or not getattr(result, "ok", False):
            return []
        findings: list[Finding] = []
        if not result.rows:
            findings.append(self._diagnose_empty(sql))
        elif len(result.rows) == 1 and len(result.columns) == 1 and result.rows[0][0] is None:
            findings.append(Finding(
                "null_scalar", WARN,
                "the query returned exactly one value and it is NULL, which usually "
                "means a division by an empty filter or an aggregate over zero rows. "
                "Check the WHERE clause matches any rows before dividing."))
        findings += self._share_range(sql, result, question)
        return findings

    def _diagnose_empty(self, sql: str) -> Finding:
        """Say *why* it is empty when the reason is a join whose keys never meet."""
        ast = parse_sql(self.con, sql)
        if ast is not None:
            binding, tables = _relations(ast)
            for left_names, right_names in _equalities(ast):
                left, left_col = self._resolve(left_names, binding, tables)
                right, right_col = self._resolve(right_names, binding, tables)
                if not left or not right or left == right:
                    continue
                try:
                    shared = self.con.execute(
                        f'SELECT COUNT(*) FROM (SELECT "{left_col}" FROM "{left}" '
                        f'INTERSECT SELECT "{right_col}" FROM "{right}")').fetchone()[0]
                except Exception:
                    continue
                if shared == 0:
                    return Finding(
                        "empty_result", WARN,
                        f"no rows: {left}.{left_col} and {right}.{right_col} have no "
                        f"values in common at all, so this join can never match. These "
                        f"columns are not a key pair.")
        return Finding(
            "empty_result", NOTE,
            "the query ran but matched no rows. Say so plainly rather than "
            "reporting zero as a measured value.")

    def _share_range(self, sql: str, result, question: str = "") -> list[Finding]:
        """A `100.0 * x / y` share that lands outside [0, 100].

        Narrow on purpose, and narrowed by measurement rather than by taste. The
        naive form of this check — "anything the SQL multiplies by 100 belongs in
        [0, 100]" — flags `wholesale_fy2025_revenue_change`, whose correct answer
        is -4.52 percent, for a false-positive rate of 1/39 on the golden set.

        Two guards take it to zero, and the structural one is the load-bearing
        half. A percentage *change* subtracts before it divides, and DuckDB's
        parser exposes that subtraction as a function node named `-`; a percentage
        *share* has no subtraction at all. So the presence of a subtraction is
        enough to know this is a difference and not a share, without reading the
        question. The word test on the question is the belt to that braces: it
        catches a phrasing whose SQL happens to compute the difference some other
        way, and it is why `check_result` accepts the question at all.
        """
        text = (sql or "")
        if not re.search(r"100(?:\.0+)?\s*\*", text):
            return []
        if _CHANGE_WORDS.search(question or "") or _CHANGE_WORDS.search(text):
            return []
        ast = parse_sql(self.con, sql)
        if ast is not None and any(
            node.get("class") == "FUNCTION" and node.get("function_name") == "-"
            for node in _walk(ast)
        ):
            return []
        findings = []
        for index, column in enumerate(result.columns):
            name = str(column)
            if not (_SHARE_ALIAS.search(name) or "100.0 *" in name or "100 *" in name):
                continue
            if _CHANGE_WORDS.search(name):
                continue
            for row in result.rows[:50]:
                value = row[index]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if not 0 <= value <= 100:
                    findings.append(Finding(
                        "share_out_of_range", NOTE,
                        f"{name} is built as a percentage but returned {value}, which is "
                        f"outside 0-100. If it is a share, the denominator is probably "
                        f"wrong or the join duplicated rows."))
                    break
        return findings

    # ---- ambiguity ------------------------------------------------------

    def ambiguity_note(self, question: str, sql: str) -> Finding | None:
        """Disclose which domain answered an entity-ambiguous question.

        This deliberately does NOT ask a clarifying question. Being too eager to
        ask is its own failure, and the retrieval signal that would have to decide
        when to ask does not separate the two cases: measured across the 39 golden
        questions (all unambiguous by construction), the share of the top-6
        retrieved tables belonging to the top domain ranges from 0.17 to 1.00 with
        a median of 0.83, while six genuinely ambiguous probes ("who is the top
        customer?", "what is our biggest cost?") range from 0.17 to 0.83. The
        distributions overlap, so any threshold that catches the ambiguous ones
        also stops roughly one golden question in six to ask a question that has
        an answer. Asking is therefore left to the model, which can see the
        retrieved schema; this only makes the resolution visible.
        """
        words = [w for w in self._ambiguous_bases
                 if re.search(rf"\b{w}s?\b", (question or "").lower())]
        if not words:
            return None
        ast = parse_sql(self.con, sql)
        if ast is None:
            return None
        _binding, tables = _relations(ast)
        domains = sorted({DOMAIN_OF[t] for t in tables})
        if len(domains) != 1:
            return None
        counts = ", ".join(f"{w} ({len(self._ambiguous_bases[w])} domains)" for w in sorted(words))
        return Finding(
            "ambiguous_entity", NOTE,
            f"answered from the {domains[0]} domain. More than one domain has "
            f"{counts}, so this reading was chosen, not given.")

    # ---- error enrichment -----------------------------------------------

    def explain_error(self, sql: str, error: str) -> str:
        """Turn a DuckDB error into one a self-correcting model can act on.

        Two things are added. First, DuckDB's own follow-up lines: it already
        computes `Candidate bindings:` for an unknown column and `Did you mean` for
        an unknown table, and `run_query` currently keeps only the first line of
        the exception, so those are being thrown away before the model ever sees
        them (see the wiring note in the spec). Second, the part DuckDB cannot
        know: which OTHER warehouse table carries the column that is missing here,
        which is the single most common real correction — the model reaches for
        `payer_type` on the fact table when it lives on the dimension.
        """
        error = (error or "").strip()
        if not error:
            return error
        column = _missing_column(error)
        if not column:
            return error
        used = set()
        ast = parse_sql(self.con, sql)
        if ast is not None:
            _binding, used = _relations(ast)
        elsewhere = [t for t in self.tables_with_column(column) if t not in used]
        if not elsewhere:
            return error
        # `department` sits on four tables in two domains. The one worth naming is
        # the one in the domain the query is already reading, so candidates are
        # ordered by that first — a suggestion that sends the model across domains
        # would hand it the very join `cross_domain_join` exists to reject.
        near = {DOMAIN_OF[t] for t in used if t in DOMAIN_OF}
        elsewhere.sort(key=lambda t: DOMAIN_OF.get(t) not in near)
        shown = ", ".join(elsewhere[:4])
        more = f" (and {len(elsewhere) - 4} more)" if len(elsewhere) > 4 else ""
        return (f"{error}\nColumn \"{column}\" does exist in this warehouse, on: "
                f"{shown}{more}. Join to the right table, or use a column that is "
                f"actually on {', '.join(sorted(used)) or 'the table you selected from'}.")


_MISSING_COLUMN_PATTERNS = (
    re.compile(r'Referenced column "([^"]+)" not found', re.IGNORECASE),
    re.compile(r'does not have a column named "([^"]+)"', re.IGNORECASE),
    re.compile(r'column "([^"]+)" not found', re.IGNORECASE),
)


def _missing_column(error: str) -> str | None:
    for pattern in _MISSING_COLUMN_PATTERNS:
        match = pattern.search(error)
        if match:
            return match.group(1)
    return None


def _ambiguous_base_names() -> dict[str, list[str]]:
    """{'customer': ['retail', 'supplychain', 'wholesale'], ...}

    Derived from the manifest rather than listed by hand, so a twelfth domain
    that also has a dim_customer is covered the day it lands.
    """
    by_base: dict[str, list[str]] = {}
    for domain, table, _source, _desc in MANIFEST:
        by_base.setdefault(table, []).append(domain)
    words: dict[str, list[str]] = {}
    for base, domains in by_base.items():
        if len(domains) < 2:
            continue
        word = re.sub(r"^(dim|fact)_", "", base).rstrip("s")
        words[word] = sorted(set(domains))
    return words
