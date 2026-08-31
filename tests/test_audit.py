"""The audit trail: what it records, what it refuses to record, and what it
promises it is not.

The interesting tests here are the negative ones. An audit trail is trivially
easy to write and easy to write badly — the ways it goes wrong are that it
becomes a second, less-guarded copy of the data, that it leaks a credential, or
that it takes down the turn it was supposed to be observing.
"""

import json

import pytest

from engine import audit


@pytest.fixture(autouse=True)
def _clean_ring():
    audit.clear()
    yield
    audit.clear()


def _one(**kw):
    base = dict(actor="session-abc", engine="plan", question="how many claims?")
    base.update(kw)
    return audit.record(**base)


def test_a_turn_produces_one_record_a_reviewer_can_read(monkeypatch, tmp_path):
    sink = tmp_path / "audit.jsonl"
    monkeypatch.setenv(audit.ENV_SINK, str(sink))

    _one(sql="SELECT COUNT(*) FROM healthcare_fact_claims", row_count=1,
         tables=["healthcare_fact_claims"], coverage=1.0,
         timings={"plan": 27.4, "execute": 3.1})

    lines = sink.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    # The whole point: the statement that produced the number is in the record.
    assert row["sql"] == "SELECT COUNT(*) FROM healthcare_fact_claims"
    assert row["engine"] == "plan"
    assert row["outcome"] == "answered"
    assert row["ts"].endswith("+00:00") or "T" in row["ts"]


def test_row_values_are_never_written_only_their_count():
    """The first thing an audit trail must not become is a second copy of the
    data with weaker access controls than the first."""
    rec = _one(sql="SELECT base_salary FROM hr_fact_employees", row_count=1900)
    blob = rec.as_json()

    assert '"row_count":1900' in blob
    assert "salary" not in blob.replace("SELECT base_salary FROM hr_fact_employees", "")


def test_a_key_shaped_token_cannot_survive_a_round_trip():
    """Redaction is applied to every string field, not to the one field a key is
    'supposed' to arrive in — the path that leaks is the one nobody modelled,
    e.g. a visitor pasting their key into the question box."""
    key = "sk-ant-api03-AAAAAAAABBBBBBBBCCCCCCCC"
    rec = _one(question=f"my key is {key} how many claims?",
               reason=f"failed with {key}", sql=f"SELECT '{key}'")

    blob = rec.as_json()
    assert key not in blob
    assert blob.count("[redacted-key]") >= 3


def test_a_broken_sink_never_takes_the_answer_down_with_it(monkeypatch, tmp_path):
    """An audit sink that can fail a turn is a worse feature than no sink."""
    unwritable = tmp_path / "audit.jsonl"
    unwritable.mkdir()  # a directory where a file is expected -> OSError on open
    monkeypatch.setenv(audit.ENV_SINK, str(unwritable))

    rec = _one()  # must not raise
    assert rec.question.startswith("how many claims")
    # and the record still reached the ring, so the panel still shows the turn
    assert len(audit.recent()) == 1
    assert audit.summarise()["sink_error"]


def test_payloads_are_bounded_because_the_editor_accepts_anything():
    huge = "SELECT " + ("x" * 50_000)
    rec = _one(sql=huge, question="q" * 5_000, reason="r" * 5_000)

    assert len(rec.sql) <= audit.MAX_SQL_CHARS
    assert len(rec.question) <= audit.MAX_QUESTION_CHARS
    assert len(rec.reason) <= audit.MAX_REASON_CHARS


def test_the_ring_is_bounded_so_a_long_lived_process_cannot_grow_on_traffic():
    for i in range(audit.RING_SIZE + 40):
        _one(question=f"question {i}")
    assert len(audit.recent()) == audit.RING_SIZE


def test_the_summary_reports_only_what_really_happened():
    _one(question="a", outcome="answered", elapsed_ms=10, timings={"plan": 4})
    _one(question="b", outcome="answered", elapsed_ms=30, timings={"plan": 6})
    _one(question="c", outcome="refused", refusal_kind="outside the grammar",
         elapsed_ms=5)
    _one(question="d", engine="manual", outcome="blocked", elapsed_ms=1)

    s = audit.summarise()
    assert s["turns"] == 4
    assert s["answered"] == 2 and s["refused"] == 1 and s["blocked"] == 1
    assert s["refusal_rate"] == pytest.approx(0.25)
    assert s["engines"] == {"plan": 3, "manual": 1}
    assert s["refusal_kinds"] == {"outside the grammar": 1}
    assert s["stage_p50_ms"]["plan"] in (4.0, 6.0)


def test_session_summary_does_not_mix_other_browser_actors():
    _one(actor="session-a", question="a")
    _one(actor="session-b", question="b")
    _one(actor="session-a", question="c")
    summary = audit.summarise(actor="session-a")
    assert summary["turns"] == 2


def test_percentiles_are_nearest_rank_so_every_number_really_happened():
    """An interpolated p95 over a handful of turns reports a latency no request
    had. This panel's claim is that every number on it happened."""
    for i, ms in enumerate([5, 10, 15, 20, 100]):
        _one(question=str(i), elapsed_ms=ms)
    s = audit.summarise()
    assert s["p50_ms"] in (10.0, 15.0)
    assert s["p95_ms"] == 100.0
    assert s["max_ms"] == 100.0


def test_an_empty_ring_summarises_to_zeroes_not_to_a_crash():
    s = audit.summarise()
    assert s["turns"] == 0 and s["refusal_rate"] == 0.0 and s["p95_ms"] == 0.0


def test_the_trail_documents_what_it_does_not_prove():
    """An undocumented control is how a reviewer ends up relying on something
    that was never load-bearing."""
    limits = audit.describe_limits()
    joined = " ".join(limits).lower()
    assert len(limits) >= 3
    assert "not an authenticated identity" in joined
    assert "row values are never written" in joined


def test_with_no_sink_configured_nothing_is_written_to_disk(monkeypatch, tmp_path):
    monkeypatch.delenv(audit.ENV_SINK, raising=False)
    _one()
    assert audit.sink_path() is None
    assert len(audit.recent()) == 1
    assert not list(tmp_path.iterdir())


def test_the_record_can_be_read_back_by_duckdb(monkeypatch, tmp_path):
    """The consumer is a query engine, not a person tailing a terminal. If the
    file cannot be read with read_json_auto the format is decorative."""
    import duckdb

    sink = tmp_path / "audit.jsonl"
    monkeypatch.setenv(audit.ENV_SINK, str(sink))
    _one(question="a", outcome="answered", row_count=10)
    _one(question="b", outcome="refused", refusal_kind="no previous turn")

    con = duckdb.connect(":memory:")
    rows = con.execute(
        "SELECT outcome, COUNT(*) FROM read_json_auto(?) GROUP BY 1 ORDER BY 1",
        [str(sink)]).fetchall()
    assert rows == [("answered", 1), ("refused", 1)]
