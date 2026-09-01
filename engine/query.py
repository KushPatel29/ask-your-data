"""
Execute a validated read-only query against the warehouse and return a small,
display-ready result. Rows are capped so a broad query can't dump a whole fact
table into a chat window; the cap being hit is reported, not hidden.

THE ROW CAP WAS NEVER THE EXPENSIVE PART
`MAX_ROWS` bounds what comes *back*. It says nothing about what the database
does to produce it, and the difference is the whole reason this module now owns
a clock. Four statements, every one a single SELECT with no forbidden verb in
it, pass the guard and pass the verifier:

    WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t WHERE n<5e7)
      SELECT COUNT(*) FROM t
    SELECT COUNT(*) FROM aml_fact_transactions a, aml_fact_transactions b
      WHERE a.amount_cad > b.amount_cad          -- 10,059,889,401 pairs
    SELECT COUNT(*) FROM range(5000000000) WHERE range % 7 = 0
    SELECT length(repeat('x', 2000000000))

Measured against the real warehouse: all four were still running after 20
seconds with memory climbing, and nothing in the process could stop them. That
is not a hypothetical. Streamlit serves every visitor from one process, the
warehouse is one shared `@st.cache_resource` connection, and since the SQL
editor shipped, any visitor can paste any SELECT. One of those four wedges the
app for everybody until the container is restarted.

So execution is bounded in time as well as in rows, using DuckDB's own
cooperative cancellation: `cursor.interrupt()`, which is safe to call from
another thread. Measured with the same four statements, each stopped within
milliseconds of its deadline with `INTERRUPT Error: Interrupted!`, while
`SELECT COUNT(*) FROM hr_fact_employees` still returned in 0.1s.

Cooperative means DuckDB checks the flag *between units of work*, so the
deadline is a floor and not a guarantee: `SELECT length(repeat('x', 2e9))` is
one allocation and overran a 3s deadline to 7.0s before the interrupt landed.
That is the other half's job — `memory_limit` and `threads` are set in
`engine/warehouse._seal()`, so a statement that cannot be stopped in time is
still bounded in what it can consume. Neither control is sufficient alone, and
saying so is more useful than quoting the one number that flatters the clock.

WHY A SHARED WATCHDOG AND NOT A THREAD PER QUERY
The obvious shape — run the query on a worker thread and `join(timeout)` on
the caller — costs a measured **12.3ms per call** on Windows, against 1.45ms
for the query itself. That is not a rounding error: it is a 7.7x tax on every
turn, it would have shown up in the app's own EXECUTE readout, and it took the
offline suite from 37s to 144s. `threading.Timer` is better and still costs
6.3ms, because it creates and tears down a thread per query.

So the deadline goes on a single long-lived heap tended by one daemon thread.
Registering is a lock, a heap push and a notify; measured at 1.28ms/call
against 1.45ms unguarded — i.e. free, inside the noise. A safety control that
is free is a safety control that never gets an exception carved out of it later
because someone was optimising a hot path.

WHY THE ERROR STRING IS TRUNCATED
`error` does not only reach a human. `engine/assistant.py` feeds it back into
the tool-enabled conversation as a user-role message so the model can correct
itself, and DuckDB quotes the offending DATA VALUE verbatim in conversion
errors:

    Conversion Error: Could not convert string 'IGNORE PREVIOUS INSTRUCTIONS…'
    to INT32 when casting from source column note

That is a warehouse cell reaching the prompt of the call whose output is
executable SQL, in one hop, on the ordinary failure path. Truncation narrows
that channel; it does not close it, and this comment should not pretend
otherwise. 200 characters keeps every correction signal that matters
("Binder Error: column X not found", "Catalog Error: Table with name Y does
not exist") and cuts a long payload to a fragment.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field

from engine.access import AccessScope, authorize_sql
from engine.deadline import DeadlineExpired, RequestDeadline
from engine.sql_guard import validate_sql

MAX_ROWS = 200

# Wall-clock ceiling for one statement. Chosen against measurement rather than
# taste: over the 39 golden queries the slowest is single-digit milliseconds,
# so 15s is roughly three orders of magnitude of headroom for an honest query
# and still short enough that a wedged container recovers inside one page load.
STATEMENT_TIMEOUT_S = 15.0

# The error string is fed back to the model. See the module docstring.
MAX_ERROR_CHARS = 200


@dataclass
class QueryResult:
    sql: str
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)   # list of tuples
    row_count: int = 0
    truncated: bool = False
    error: str = ""
    # Set only when the statement was cancelled by the clock above. `error` is
    # populated too, so every existing caller keeps working, but a caller that
    # wants to say "too expensive" rather than "broken" can now tell. They are
    # different facts about a turn and they lead to different next actions.
    timed_out: bool = False
    policy_denied: bool = False
    timeout_stage: str = ""
    elapsed_ms: float = 0.0

    @property
    def ok(self):
        return not self.error


class _Watchdog:
    """One daemon thread holding every in-flight statement's deadline.

    The thread is started lazily on first registration, so importing this
    module — which `app/cli.py`, the tests and every eval script do — costs
    nothing until a query actually runs.
    """

    def __init__(self) -> None:
        self._heap: list = []
        self._cv = threading.Condition()
        self._seq = 0
        self._thread: threading.Thread | None = None

    def _pump(self) -> None:
        while True:
            with self._cv:
                if not self._heap:
                    self._cv.wait()
                    continue
                due = self._heap[0][0]
                now = time.monotonic()
                if due > now:
                    self._cv.wait(due - now)
                    continue
                while self._heap and self._heap[0][0] <= time.monotonic():
                    _, _, cursor, state = heapq.heappop(self._heap)
                    if state["live"]:
                        state["fired"] = True
                        try:
                            cursor.interrupt()
                        except Exception:
                            # A cursor closed between expiry and interrupt is
                            # the ordinary race, not an error worth surfacing.
                            pass

    def arm(self, cursor, seconds: float) -> dict:
        """Register a deadline. Returns the state dict the caller disarms."""
        state = {"live": True, "fired": False}
        with self._cv:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._pump, name="ayd-query-watchdog", daemon=True)
                self._thread.start()
            self._seq += 1
            heapq.heappush(self._heap, (time.monotonic() + seconds, self._seq,
                                        cursor, state))
            self._cv.notify()
        return state

    @staticmethod
    def disarm(state: dict) -> None:
        # No lock and no heap removal: the entry is left to expire harmlessly
        # against a dead flag. Removing it would mean an O(n) scan under the
        # lock on the hot path to save a wakeup that costs nothing.
        state["live"] = False


_WATCHDOG = _Watchdog()


def _short(error: str) -> str:
    text = str(error).strip()
    first = text.splitlines()[0] if text else str(error)
    return first[:MAX_ERROR_CHARS]


def run_query(con, sql: str, max_rows: int = MAX_ROWS,
              timeout_s: float = STATEMENT_TIMEOUT_S,
              access: AccessScope | None = None,
              deadline: RequestDeadline | None = None) -> QueryResult:
    ok, reason = validate_sql(sql)
    if not ok:
        return QueryResult(sql=sql, error=f"blocked by SQL guard: {reason}")
    decision = authorize_sql(con, sql, access)
    if not decision.allowed:
        return QueryResult(
            sql=sql,
            policy_denied=True,
            error=f"blocked by data policy: {decision.reason}",
        )
    if deadline is not None:
        try:
            timeout_s = min(timeout_s, deadline.require("query execution"))
        except DeadlineExpired as exc:
            return QueryResult(
                sql=sql,
                timed_out=True,
                timeout_stage=exc.stage,
                error=str(exc),
            )
    # Each query runs on its own cursor (a duplicate connection to the same
    # in-memory database), so concurrent callers — e.g. two Streamlit sessions
    # sharing the cached warehouse — never interleave on one connection. The
    # cursor is also what `interrupt()` is called on, so a cancellation is
    # scoped to this statement and cannot disturb another session's query.
    cur = con.cursor()
    state = _WATCHDOG.arm(cur, timeout_s)
    started = time.perf_counter()
    try:
        cur.execute(sql)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(max_rows + 1)
    except Exception as e:  # DuckDB errors (bad column, interrupt) are data
        elapsed = 1000 * (time.perf_counter() - started)
        # Whether this was a timeout is read off the watchdog, not off the
        # error text. String-matching "INTERRUPT" would also catch a genuine
        # interrupt from somewhere else and would break on a DuckDB wording
        # change; the flag is set by the only code that cancels here.
        if state["fired"]:
            return QueryResult(
                sql=sql, timed_out=True, elapsed_ms=elapsed,
                timeout_stage=("request" if deadline is not None
                               and deadline.remaining_s <= 0 else "query execution"),
                error=(f"query exceeded the {timeout_s:g}s statement timeout and was "
                       "cancelled. Narrow it — add a filter, a GROUP BY, or a LIMIT."))
        return QueryResult(sql=sql, error=_short(e), elapsed_ms=elapsed)
    finally:
        _WATCHDOG.disarm(state)
        cur.close()

    elapsed = 1000 * (time.perf_counter() - started)
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return QueryResult(sql=sql, columns=columns, rows=rows,
                       row_count=len(rows), truncated=truncated, elapsed_ms=elapsed)
