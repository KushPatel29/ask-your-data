"""
The durable record of what this system was asked and what it did about it.

WHY THIS FILE EXISTS
An analyst who writes a query leaves a query behind and nothing else. An LLM —
or a compiler — writing SQL against a warehouse leaves the *entire* decision
trail: the question as asked, the tables retrieval was allowed to offer, the
statement that was actually executed, what the guard and the verifier said
about it, how many rows came back, how many attempts it took, and what it cost.
That record is strictly richer than the human equivalent, and until this module
existed it was assembled in full on `AskResult` / `PlanResult` and then dropped
on the floor when the function returned. Shipping the assistant without
persisting it forfeits the one compliance argument this architecture hands you
for free.

So this is not "logging". It is the artifact a reviewer asks for when they ask
*how did that number get into the board deck* — and the answer is one JSON line
containing the statement that produced it.

WHAT IT IS NOT
It is not authentication, and this module must not pretend otherwise. There is
no login on this app, so `actor` is a per-browser session identifier and a role
label, not an identity. A record that says `actor: session-8f2a1c` is honest;
one that says `actor: alice@corp` would be a lie the app has no way to check.
`describe_limits()` says so in the interface, because an audit trail whose
weaknesses are undocumented is worse than none — someone will rely on it.

DESIGN CHOICES WORTH DEFENDING

*JSON Lines, appended, one record per turn.* The consumer is a SIEM or a
`duckdb.read_json_auto()` query, not a person tailing a terminal. Append-only
means a writer never has to read what is already there, so two Streamlit
sessions cannot lose each other's records to a read-modify-write.

*The sink is pluggable and defaults to disabled.* `ASK_YOUR_DATA_AUDIT` names
the file. Unset, records go to a bounded in-memory ring — which is what the
public Streamlit deployment runs, because its filesystem is ephemeral and a log
nobody can retrieve is theatre. The ring is what the app's own operations panel
reads, so the feature is demonstrable on a deployment that persists nothing.

*Payloads are bounded before they are written.* Question, SQL and reason are
clipped. A caller can send an arbitrarily long hand-written query through the
editor, and an audit sink that grows without limit on user input is a second
denial-of-service dressed as compliance.

*Values are never recorded — only shapes.* `row_count` is written; rows are
not. The warehouse here is synthetic, but a real one would not be, and the
first thing an audit trail must not become is a second copy of the data with
weaker access controls than the first. Same reason the API key never appears:
`redact()` is applied to every string field on the way in, and there is a test
that a key-shaped token cannot survive a round trip.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from engine import automation

# One turn's record, clipped. These are not arbitrary: the SQL bound is above
# the longest statement the planner or the golden set produces (measured: 412
# characters), and the question bound is above the longest question in either
# eval file, so ordinary traffic is never truncated and only abuse is.
MAX_QUESTION_CHARS = 500
MAX_SQL_CHARS = 4000
MAX_REASON_CHARS = 600
MAX_CORRECTION_CHARS = 200

# The in-memory sink. Bounded, because the process is long-lived and the app is
# a public URL. 500 turns is far more than one session produces and small
# enough that the ring is a rounding error against the 290MB warehouse.
RING_SIZE = 500

ENV_SINK = "ASK_YOUR_DATA_AUDIT"

# Anything that looks like a provider key. Applied to every string field rather
# than to the one field a key is "supposed" to arrive in, because the whole
# point of a redactor is that it covers the path nobody thought of — a user
# pasting their key into the question box, for instance.
_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,}|sk-ant-[A-Za-z0-9_\-]{8,})")

_ring: deque = deque(maxlen=RING_SIZE)
_lock = threading.Lock()
_sink_error = ""


def redact(text: str | None) -> str:
    if not text:
        return ""
    return _KEY_RE.sub("[redacted-key]", str(text))


def _clip(text: str | None, limit: int) -> str:
    out = redact(text)
    return out if len(out) <= limit else out[: limit - 1] + "…"


@dataclass
class AuditRecord:
    """One answered (or refused) question.

    Field order is the order a reviewer reads them in: when, who, what was
    asked, what ran, what the safety layers said, what came back, what it cost.
    """

    ts: str
    actor: str
    role: str
    engine: str                     # plan | model | manual | contract
    question: str
    sql: str = ""
    outcome: str = ""               # answered | refused | blocked | error | timeout
    refusal_kind: str = ""
    reason: str = ""
    tables: list = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    attempts: int = 1
    corrections: list = field(default_factory=list)
    guard_ok: bool = True
    guard_reason: str = ""
    verifier: list = field(default_factory=list)   # [{"rule":…, "severity":…}]
    metric: str = ""                # a certified metric, when one supplied the SQL
    coverage: float | None = None   # the compiler's own confidence, when it ran
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_ms: float = 0.0
    timings: dict = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sink_path() -> Path | None:
    """The file records are appended to, or None when only the ring is active."""
    raw = os.environ.get(ENV_SINK, "").strip()
    return Path(raw) if raw else None


def record(
    *,
    actor: str,
    engine: str,
    question: str,
    role: str = "owner",
    sql: str = "",
    outcome: str = "answered",
    refusal_kind: str = "",
    reason: str = "",
    tables: list | None = None,
    row_count: int = 0,
    truncated: bool = False,
    attempts: int = 1,
    corrections: list | None = None,
    guard_ok: bool = True,
    guard_reason: str = "",
    verifier: list | None = None,
    metric: str = "",
    coverage: float | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    elapsed_ms: float = 0.0,
    timings: dict | None = None,
) -> AuditRecord:
    """Write one record. Never raises — an audit sink that can break a turn is
    a worse feature than no audit sink, and the caller is in the middle of
    answering somebody's question."""
    rec = AuditRecord(
        ts=_now(),
        actor=_clip(actor, 64),
        role=_clip(role, 32),
        engine=engine,
        question=_clip(question, MAX_QUESTION_CHARS),
        sql=_clip(sql, MAX_SQL_CHARS),
        outcome=outcome,
        refusal_kind=_clip(refusal_kind, 64),
        reason=_clip(reason, MAX_REASON_CHARS),
        tables=list(tables or [])[:20],
        row_count=int(row_count or 0),
        truncated=bool(truncated),
        attempts=int(attempts or 1),
        corrections=[_clip(c, MAX_CORRECTION_CHARS) for c in (corrections or [])][:5],
        guard_ok=bool(guard_ok),
        guard_reason=_clip(guard_reason, 200),
        verifier=list(verifier or [])[:20],
        metric=_clip(metric, 64),
        coverage=coverage,
        tokens_in=int(tokens_in or 0),
        tokens_out=int(tokens_out or 0),
        elapsed_ms=round(float(elapsed_ms or 0.0), 2),
        timings={k: round(float(v), 2) for k, v in (timings or {}).items()},
    )
    global _sink_error
    with _lock:
        _ring.append(rec)
        path = sink_path()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(rec.as_json() + "\n")
                _sink_error = ""
            except OSError as exc:
                # A read-only or full filesystem must not take the answer down
                # with it. The ring still holds the record, and the operations
                # panel reports the sink as unavailable rather than pretending.
                _sink_error = str(exc)[:200]
    automation.publish(rec)
    return rec


def recent(limit: int = RING_SIZE) -> list[AuditRecord]:
    with _lock:
        return list(_ring)[-limit:]


def clear() -> None:
    """Only for tests. The ring is process-global by design."""
    global _sink_error
    with _lock:
        _ring.clear()
        _sink_error = ""


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated. With the handful of turns one session
    produces, an interpolated p95 reports a latency that no request actually
    had — and this panel's entire claim is that every number on it happened.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def summarise(
    records: list[AuditRecord] | None = None,
    *,
    actor: str | None = None,
) -> dict:
    """Aggregate the ring into the numbers an operator asks for.

    This is the difference between observability *of a request* — which this
    app already had, in the pipeline strip — and observability *of a service*,
    which needs something durable to aggregate over. Every value here is
    derived from records that were really written; none is a placeholder.
    """
    rows = list(records if records is not None else recent())
    if actor is not None:
        rows = [row for row in rows if row.actor == actor]
    total = len(rows)
    answered = [r for r in rows if r.outcome == "answered"]
    refused = [r for r in rows if r.outcome == "refused"]
    blocked = [r for r in rows if r.outcome == "blocked"]
    failed = [r for r in rows if r.outcome in ("error", "timeout")]
    latencies = [r.elapsed_ms for r in rows if r.elapsed_ms]

    engines: dict[str, int] = {}
    for r in rows:
        engines[r.engine] = engines.get(r.engine, 0) + 1
    kinds: dict[str, int] = {}
    for r in refused:
        if r.refusal_kind:
            kinds[r.refusal_kind] = kinds.get(r.refusal_kind, 0) + 1

    stage: dict[str, list[float]] = {}
    for r in rows:
        for name, value in (r.timings or {}).items():
            stage.setdefault(name, []).append(float(value))

    return {
        "turns": total,
        "answered": len(answered),
        "refused": len(refused),
        "blocked": len(blocked),
        "failed": len(failed),
        # A refusal rate is the number this system should be proudest of and
        # most watched on: it is the price paid for never returning a wrong
        # number, and it is the first thing that moves when the grammar or the
        # gate changes.
        "refusal_rate": (len(refused) / total) if total else 0.0,
        "engines": engines,
        "refusal_kinds": kinds,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "max_ms": max(latencies) if latencies else 0.0,
        "stage_p50_ms": {k: _percentile(v, 50) for k, v in sorted(stage.items())},
        "tokens_in": sum(r.tokens_in for r in rows),
        "tokens_out": sum(r.tokens_out for r in rows),
        "certified": sum(1 for r in rows if r.metric),
        "sink": str(sink_path()) if sink_path() else "",
        "sink_error": _sink_error,
        "automation": automation.status(),
    }


def describe_limits() -> list[str]:
    """What this trail does NOT prove. Rendered in the app beside the panel.

    An audit trail is a control, and an undocumented control is how a reviewer
    ends up relying on something that was never load-bearing.
    """
    return [
        "actor is a per-browser session id, not an authenticated identity — "
        "this app has no login",
        "records are appended by the process that answers, so a crash mid-turn "
        "loses that turn",
        "row values are never written, only counts — the trail is not a second "
        "copy of the data",
        "with no ASK_YOUR_DATA_AUDIT path set, records live in a bounded "
        f"in-memory ring of {RING_SIZE} and die with the container",
        "the optional n8n event contains operational metadata only; question, "
        "SQL, result values, reasons, and retry feedback are excluded",
    ]
