"""
A semantic layer over the warehouse, derived rather than authored.

`engine/retrieval.py` answers "which tables is this question about". That is
enough to build a *prompt*. It is nowhere near enough to build a *query*: to
compile "average claim amount by payer type" into SQL you need to know that
`amount` is a measure and `payer_type` is a dimension, that the two live in
different tables, and which column joins them.

This module works that out from the warehouse itself. Every fact here is probed
from DuckDB or read off `data_manifest.py`; none of it is a hand-written mapping
file that can drift away from the data it describes. That is the same argument
`run_retrieval_eval.py` makes about ground truth -- labels derived from the
artefact beat labels typed next to it.

WHAT IS INFERRED, AND FROM WHAT

  role        A column is a KEY if its name ends in `_id`/`_key`/`_num` or it is
              unique across the table; a DATE if DuckDB says so; a FLAG if it is
              boolean, or an integer named `is_*`/`has_*` holding only 0/1; a
              MEASURE if it is numeric and not one of the above; a DIMENSION
              otherwise. Numeric-but-categorical columns are the trap here
              (`record_num` is not something you average), which is why the name
              test runs before the type test.

  grain       The narrowest set of key columns that is unique across the table.
              Tried at one column, then pairs. Used to decide whether a join
              fans out, which is the difference between a correct SUM and one
              that double-counts.

  join edges  Two tables are joinable on column `c` when both have it, and on at
              least one side `c` is unique. That direction matters: it is what
              makes the edge a foreign key rather than a coincidence of naming.
              Edges never cross a domain -- `engine/verify.py` already rules a
              cross-domain join an ERROR, and a planner that could build one
              would be planning a query the verifier exists to block.

  lexicon     Phrases that bind English to schema. Three sources, none typed by
              hand: column names (split on snake_case, so `qty_shipped` offers
              "qty", "shipped" and "qty shipped" -- the same fix that moved
              retrieval recall to 100%), table and domain descriptions from the
              manifest, and -- the one that matters most -- the DISTINCT VALUES
              of every low-cardinality dimension. That last source is why
              "denied claims" can become `WHERE status = 'Denied'` without
              anyone having written down that `Denied` is a status: the database
              already knows.

WHAT IS NOT HERE

No metric definitions, no `revenue = SUM(net_amount)` YAML. A metric layer is
the right answer for a warehouse with a modelling team behind it; it is the
wrong answer here, because a hand-authored metric is exactly the kind of claim
this repo refuses to make without a test. What the planner gets instead is the
raw material -- measures, dimensions, grains, joins, values -- and it has to
earn each query from evidence in the question. When it cannot, it refuses.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_manifest import DOMAINS, MANIFEST, table_name  # noqa: E402
from engine.warehouse import table_columns, table_names  # noqa: E402

# Above this many distinct values a text column stops being a thing you can name
# in a question ("which region" has six answers; "which transaction id" has a
# hundred thousand) and starts being an identifier. 40 is the elbow in this
# warehouse: every column below it is a real category, and the first ones above
# it are surrogate keys and free text.
MAX_DIMENSION_CARDINALITY = 40

# Sampling ceiling for the value lexicon. A dimension under the cardinality cap
# has all of its values indexed; this only guards against a pathological table
# appearing later.
MAX_INDEXED_VALUES = 60

KEY_SUFFIXES = ("_id", "_key", "_num", "_code", "_oid")
FLAG_PREFIXES = ("is_", "has_", "was_", "did_")

# Numeric and real, but meaningless to SUM because it is already a ratio, a
# score, or a calendar part. The planner may still average these; it may not add
# them. Detected by name because that is the only signal available -- the type
# system cannot tell a rate from a dollar.
RATIO_TOKENS = {"rate", "pct", "percent", "percentage", "ratio", "share",
                "score", "probability", "prob", "index", "zscore", "z",
                "precision", "recall", "f1", "threshold", "latitude",
                "longitude", "lat", "lon", "year", "month", "day", "hour",
                "week", "quarter", "epoch", "days", "age", "pk", "id"}

NUMERIC_TYPES = {"BIGINT", "INTEGER", "SMALLINT", "TINYINT", "HUGEINT",
                 "DOUBLE", "FLOAT", "REAL", "DECIMAL", "UBIGINT", "UINTEGER"}

TEMPORAL_TYPES = {"DATE", "TIMESTAMP", "TIMESTAMP_S", "TIMESTAMP_MS",
                  "TIMESTAMP_NS", "TIMESTAMP WITH TIME ZONE", "TIME"}

KEY = "key"
MEASURE = "measure"
DIMENSION = "dimension"
DATE = "date"
FLAG = "flag"


def split_identifier(name: str) -> list[str]:
    """`qty_shipped` -> ['qty', 'shipped']. Also splits camelCase and digits.

    The joined form is added back by callers that want it; this returns the
    parts, because the parts are what an English question contains.
    """
    out: list[str] = []
    for part in re.split(r"[^a-zA-Z0-9]+", name):
        if not part:
            continue
        for piece in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+", part):
            out.append(piece.lower())
    return out


@dataclass(frozen=True)
class Column:
    table: str
    name: str
    type: str
    role: str
    distinct: int = 0
    unique: bool = False
    values: tuple[str, ...] = ()

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(split_identifier(self.name))

    @property
    def additive(self) -> bool:
        """Can this be SUMmed and still mean something?

        A rate summed over ten rows is not a bigger rate, it is nonsense -- the
        same class of error `engine/verify.py` catches downstream as
        `share_out_of_range`. Catching it here means never writing it.
        """
        if self.role != MEASURE:
            return False
        return not (set(self.words) & RATIO_TOKENS)

    @property
    def qualified(self) -> str:
        return f'"{self.table}"."{self.name}"'


@dataclass
class Table:
    name: str
    domain: str
    description: str
    columns: list[Column] = field(default_factory=list)
    rows: int = 0
    grain: tuple[str, ...] = ()

    def column(self, name: str) -> Column | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def by_role(self, role: str) -> list[Column]:
        return [c for c in self.columns if c.role == role]

    @property
    def words(self) -> tuple[str, ...]:
        # The domain prefix is dropped: every table in the healthcare domain
        # starts with "healthcare", so it discriminates nothing between them.
        stem = self.name.split("_", 1)[1] if "_" in self.name else self.name
        return tuple(split_identifier(stem))


@dataclass(frozen=True)
class JoinEdge:
    left: str
    right: str
    column: str
    # Which side the column is unique on. "right" means left.column is a foreign
    # key into right -- the safe direction, because the join cannot fan out.
    unique_side: str

    def other(self, table: str) -> str:
        return self.right if table == self.left else self.left


@dataclass(frozen=True)
class ValueBinding:
    """A literal a question can name, and where it lives."""

    phrase: str
    table: str
    column: str
    value: str
    type: str

    @property
    def literal(self) -> str:
        if self.type == "BOOLEAN":
            return "TRUE" if str(self.value).lower() in ("true", "1") else "FALSE"
        return "'" + str(self.value).replace("'", "''") + "'"


class Layer:
    """The whole semantic layer for one warehouse connection."""

    def __init__(self, con):
        self.con = con
        self.tables: dict[str, Table] = {}
        self.edges: list[JoinEdge] = []
        self._value_index: dict[str, list[ValueBinding]] = {}
        self._word_index: dict[str, set[str]] = {}
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        described = {table_name(d, t): (d, desc) for d, t, _src, desc in MANIFEST}
        for name in table_names(self.con):
            domain, description = described.get(name, (name.split("_", 1)[0], ""))
            rows = self.con.cursor().execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            table = Table(name=name, domain=domain, description=description, rows=rows)
            table.columns = self._profile(name, rows)
            table.grain = self._grain(table)
            self.tables[name] = table
        self._index_words()
        self._index_values()
        self._build_edges()

    def _profile(self, name: str, rows: int) -> list[Column]:
        """One pass over a table's columns, assigning each a role.

        The distinct-count probe is batched into a single query per table. One
        statement per column would be 700 round trips at startup -- the
        difference between a cost you can hide behind a spinner and one you
        cannot.
        """
        cols = table_columns(self.con, name)
        if not cols:
            return []
        exprs = ", ".join(f'approx_count_distinct("{c}")' for c, _ in cols)
        counts = [int(n or 0) for n in
                  self.con.cursor().execute(f'SELECT {exprs} FROM "{name}"').fetchone()]

        # approx_count_distinct is a HyperLogLog estimate and it overshoots: on
        # healthcare_fact_claims it reported 12,912 distinct claim_id over 12,000
        # rows, and 12,801 distinct paid_amount. Trusting it made a DOUBLE money
        # column look like a unique key, which promoted the single most important
        # measure in the warehouse out of the measure set entirely -- the planner
        # then could not answer "total paid amount" at all.
        #
        # So the estimate is used only to NOMINATE candidates, and every one is
        # confirmed with an exact COUNT(DISTINCT). That is a second query per
        # near-unique column rather than per column, which on this warehouse is
        # ~90 statements and stays inside the startup budget.
        exact: dict[str, int] = {}
        candidates = [c for (c, _), n in zip(cols, counts, strict=True)
                      if rows and n >= 0.9 * rows]
        if candidates:
            probe = ", ".join(f'COUNT(DISTINCT "{c}")' for c in candidates)
            try:
                measured = self.con.cursor().execute(
                    f'SELECT {probe} FROM "{name}"').fetchone()
                exact = dict(zip(candidates, (int(n or 0) for n in measured), strict=True))
            except Exception:
                exact = {}

        out: list[Column] = []
        for (col_name, col_type), estimate in zip(cols, counts, strict=True):
            distinct = exact.get(col_name, estimate)
            unique = bool(rows) and distinct >= rows
            role = self._role(col_name, col_type, distinct, unique, rows)
            out.append(Column(table=name, name=col_name, type=col_type, role=role,
                              distinct=distinct, unique=unique))
        return out

    @staticmethod
    def _role(name: str, col_type: str, distinct: int, unique: bool, rows: int) -> str:
        base = (col_type or "").upper().split("(")[0].strip()
        lower = name.lower()
        if base in TEMPORAL_TYPES:
            return DATE
        if base == "BOOLEAN":
            return FLAG
        if lower.startswith(FLAG_PREFIXES) and base in NUMERIC_TYPES and distinct <= 3:
            return FLAG
        # Name before type, deliberately: `record_num` is a BIGINT and averaging
        # it is meaningless. An identifier declares itself in its name long
        # before its type gives it away.
        if lower.endswith(KEY_SUFFIXES):
            return KEY
        # Uniqueness alone promotes to KEY only for discrete types. A DOUBLE is
        # unique across 12,000 claims because money has cents, not because it
        # identifies anything -- and `paid_amount` is the measure the whole
        # healthcare domain is about.
        if unique and base not in ("DOUBLE", "FLOAT", "REAL", "DECIMAL"):
            return KEY
        if base in NUMERIC_TYPES:
            # A numeric column with a handful of values across thousands of rows
            # is a category wearing an integer's clothes (fiscal_year, severity).
            #
            # The cap was 12 and it was too tight by exactly the calendar:
            # `hour_of_day` has 24 values over 100,299 transactions and was
            # classified a MEASURE, so "how many transactions per hour of day?"
            # could find no axis to group on and answered with the grand total.
            # 24 covers hours; 31 would cover days of the month. The ratio guard
            # is what keeps a genuinely small table's measure out: 100k rows over
            # 24 values is a category, 32 rows over 4 values is not.
            if distinct <= 24 and rows > 100 * max(distinct, 1):
                return DIMENSION
            return MEASURE
        return DIMENSION

    def _grain(self, table: Table) -> tuple[str, ...]:
        """The narrowest unique key. Empty when nothing proves one."""
        keys = [c for c in table.columns if c.role == KEY]
        for col in keys:
            if col.unique:
                return (col.name,)
        if not table.rows:
            return ()
        # Pairs only. Beyond two columns the search is quadratic in a way that
        # buys nothing: a fact keyed on three columns is still a fact, and the
        # planner treats "no proven grain" as "assume fan-out", which is the
        # safe direction to be wrong in.
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                probe = (f'SELECT COUNT(*) FROM (SELECT DISTINCT "{left.name}", '
                         f'"{right.name}" FROM "{table.name}")')
                try:
                    n = self.con.cursor().execute(probe).fetchone()[0]
                except Exception:
                    continue
                if n >= table.rows:
                    return (left.name, right.name)
        return ()

    def _index_words(self) -> None:
        for table in self.tables.values():
            words = set(table.words)
            words.update(split_identifier(table.domain))
            for col in table.columns:
                words.update(col.words)
            for word in words:
                self._word_index.setdefault(word, set()).add(table.name)

    def _index_values(self) -> None:
        """Index the distinct values of every low-cardinality dimension.

        This is the source that lets a question say "denied" and a query say
        `status = 'Denied'`. Only DIMENSION and FLAG columns are indexed:
        indexing keys would put a hundred thousand transaction ids into a phrase
        table, and indexing measures makes no sense at all.
        """
        for table in self.tables.values():
            for position, col in enumerate(table.columns):
                if col.role not in (DIMENSION, FLAG):
                    continue
                if col.distinct > MAX_DIMENSION_CARDINALITY or col.distinct == 0:
                    continue
                if col.type.upper().split("(")[0] not in ("VARCHAR", "BOOLEAN"):
                    continue
                try:
                    rows = self.con.cursor().execute(
                        f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                        f'WHERE "{col.name}" IS NOT NULL LIMIT {MAX_INDEXED_VALUES}'
                    ).fetchall()
                except Exception:
                    continue
                values = tuple(str(r[0]) for r in rows)
                table.columns[position] = Column(
                    table=col.table, name=col.name, type=col.type, role=col.role,
                    distinct=col.distinct, unique=col.unique, values=values,
                )
                for value in values:
                    phrase = normalise_phrase(value)
                    if not phrase or phrase.isdigit():
                        continue
                    self._value_index.setdefault(phrase, []).append(
                        ValueBinding(phrase=phrase, table=table.name, column=col.name,
                                     value=str(value), type=col.type.upper())
                    )

    def _build_edges(self) -> None:
        """Foreign keys, inferred: shared column, unique on at least one side.

        Scoped inside a domain. Two domains both having `customer_id` is a
        naming coincidence, not a relationship, and joining across it is the
        exact bug `engine/verify.py` calls `cross_domain_join`.
        """
        by_domain: dict[str, list[Table]] = {}
        for table in self.tables.values():
            by_domain.setdefault(table.domain, []).append(table)
        seen: set[tuple[str, str, str]] = set()
        for tables in by_domain.values():
            for i, left in enumerate(tables):
                for right in tables[i + 1:]:
                    for lcol in left.columns:
                        if lcol.role != KEY:
                            continue
                        rcol = right.column(lcol.name)
                        if rcol is None or rcol.role != KEY:
                            continue
                        if not (lcol.unique or rcol.unique):
                            continue
                        marker = (left.name, right.name, lcol.name)
                        if marker in seen:
                            continue
                        seen.add(marker)
                        self.edges.append(
                            JoinEdge(left=left.name, right=right.name, column=lcol.name,
                                     unique_side="right" if rcol.unique else "left")
                        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def edges_for(self, table: str) -> list[JoinEdge]:
        return [e for e in self.edges if table in (e.left, e.right)]

    def join_path(self, start: str, goal: str, max_hops: int = 2) -> list[JoinEdge] | None:
        """Shortest join path between two tables, or None.

        Breadth-first and capped at two hops. A planner that will chain five
        joins to make a question fit is not grounding an answer, it is
        rationalising one -- and every extra hop is another chance to fan out.
        """
        if start == goal:
            return []
        frontier: list[tuple[str, list[JoinEdge]]] = [(start, [])]
        seen = {start}
        while frontier:
            nxt: list[tuple[str, list[JoinEdge]]] = []
            for node, path in frontier:
                if len(path) >= max_hops:
                    continue
                for edge in self.edges_for(node):
                    other = edge.other(node)
                    if other in seen:
                        continue
                    extended = path + [edge]
                    if other == goal:
                        return extended
                    seen.add(other)
                    nxt.append((other, extended))
            frontier = nxt
        return None

    def tables_with_word(self, word: str) -> set[str]:
        return set(self._word_index.get(word, ()))

    def values_for_phrase(self, phrase: str) -> list[ValueBinding]:
        return list(self._value_index.get(phrase, ()))

    @property
    def value_phrases(self) -> list[str]:
        return list(self._value_index)

    def measures(self, table: str) -> list[Column]:
        return self.tables[table].by_role(MEASURE)

    def dimensions(self, table: str) -> list[Column]:
        return [c for c in self.tables[table].columns if c.role in (DIMENSION, FLAG)]

    def summary(self) -> dict:
        """Counts, for the panel that reports what the layer actually found."""
        roles: dict[str, int] = {}
        for table in self.tables.values():
            for col in table.columns:
                roles[col.role] = roles.get(col.role, 0) + 1
        return {
            "tables": len(self.tables),
            "domains": len({t.domain for t in self.tables.values()}),
            "columns": sum(len(t.columns) for t in self.tables.values()),
            "joins": len(self.edges),
            "value_phrases": len(self._value_index),
            "grains": sum(1 for t in self.tables.values() if t.grain),
            **{f"role_{k}": v for k, v in sorted(roles.items())},
        }


def normalise_phrase(value: str) -> str:
    """Normalise a data value into something a question could plausibly contain."""
    text = str(value).strip().lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


_LAYER = None
_LAYER_CON = None


def get_layer(con) -> Layer:
    """One layer per connection, built once.

    Keyed on the connection object rather than cached globally, so a test that
    builds a second warehouse does not silently profile the first one.
    """
    global _LAYER, _LAYER_CON
    if _LAYER is None or _LAYER_CON is not con:
        _LAYER = Layer(con)
        _LAYER_CON = con
    return _LAYER


@lru_cache(maxsize=1)
def domain_words() -> dict[str, set[str]]:
    """Words that name a domain, for scoping a question before it is planned."""
    out: dict[str, set[str]] = {}
    for domain, blurb in DOMAINS.items():
        words = set(split_identifier(domain))
        words.update(w for w in split_identifier(blurb) if len(w) > 3)
        out[domain] = words
    return out


if __name__ == "__main__":
    from engine.warehouse import build_warehouse

    layer = Layer(build_warehouse())
    for key, value in layer.summary().items():
        print(f"{key:>18}: {value:,}")
