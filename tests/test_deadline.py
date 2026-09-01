from __future__ import annotations

import time

import pytest

from engine.assistant import Assistant
from engine.deadline import DeadlineExpired, RequestDeadline, configured_timeout_s
from engine.providers import ProviderBudgetExpired, ProviderResponse, ToolCall
from engine.query import run_query


def test_configured_deadline_is_bounded(monkeypatch):
    monkeypatch.setenv("ASK_REQUEST_TIMEOUT_S", "0.1")
    with pytest.raises(ValueError, match="between"):
        configured_timeout_s()
    monkeypatch.setenv("ASK_REQUEST_TIMEOUT_S", "301")
    with pytest.raises(ValueError, match="between"):
        configured_timeout_s()


def test_expired_deadline_names_the_stage():
    deadline = RequestDeadline(timeout_s=1, started_at=time.monotonic() - 2)
    with pytest.raises(DeadlineExpired, match="schema retrieval"):
        deadline.require("schema retrieval")


def test_executor_refuses_before_opening_a_cursor_when_request_expired(con):
    deadline = RequestDeadline(timeout_s=1, started_at=time.monotonic() - 2)
    result = run_query(con, "SELECT COUNT(*) FROM hr_fact_employees", deadline=deadline)
    assert not result.ok
    assert result.timed_out
    assert result.timeout_stage == "query execution"
    assert "end-to-end deadline" in result.error


class DeadlineAwareProvider:
    def __init__(self):
        self.timeouts = []

    def create_tool_call(self, *, timeout_s=None, **_kwargs):
        self.timeouts.append(timeout_s)
        return ProviderResponse(ToolCall(
            "answer_with_sql",
            {"sql": "SELECT COUNT(*) AS n FROM hr_fact_employees",
             "explanation": "Count employees."},
        ))

    def complete(self, *, timeout_s=None, **_kwargs):
        self.timeouts.append(timeout_s)
        return ProviderResponse(None, text="There are 1,900 employees.")


def test_assistant_passes_remaining_budget_to_every_provider_call(con):
    provider = DeadlineAwareProvider()
    assistant = Assistant(
        con,
        provider=provider,
        catalog_builder=lambda *_args, **_kwargs: "SCHEMA\n======\n",
    )
    result = assistant.ask("How many employees are there?", deadline=RequestDeadline(10))
    assert result.ok
    assert len(provider.timeouts) == 2
    assert all(timeout is not None and 0 < timeout <= 10 for timeout in provider.timeouts)


class ExhaustedProvider:
    def create_tool_call(self, **_kwargs):
        raise ProviderBudgetExpired("fallbacks used the remaining budget")


def test_provider_budget_exhaustion_is_reported_as_a_stage_timeout(con):
    assistant = Assistant(
        con,
        provider=ExhaustedProvider(),
        catalog_builder=lambda *_args, **_kwargs: "SCHEMA\n======\n",
    )
    result = assistant.ask("How many employees are there?", deadline=RequestDeadline(10))

    assert result.refused
    assert result.timed_out
    assert result.timeout_stage == "SQL generation"
    assert "end-to-end deadline" in result.reason
