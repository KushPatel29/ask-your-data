"""The n8n seam is bounded, signed, and privacy-minimized."""

from types import SimpleNamespace

from engine import automation


def _record():
    return SimpleNamespace(
        ts="2026-08-31T12:00:00+00:00",
        actor="session-private",
        role="owner",
        engine="model",
        outcome="answered",
        refusal_kind="",
        row_count=1,
        truncated=False,
        attempts=1,
        guard_ok=True,
        tokens_in=200,
        tokens_out=20,
        elapsed_ms=125.5,
        question="What is Alice's salary?",
        sql="SELECT base_salary FROM hr_fact_employees",
        reason="",
        corrections=["secret database error"],
    )


def test_operational_event_has_a_strict_non_sensitive_allow_list():
    event = automation.operational_event(_record())
    blob = str(event)
    assert event["outcome"] == "answered" and event["elapsed_ms"] == 126
    assert event["actor_hash"] != "session-private"
    for secret in ("Alice", "salary", "SELECT", "database error", "session-private"):
        assert secret not in blob


def test_signature_is_deterministic_and_secret_bound():
    event = automation.operational_event(_record())
    first = automation.signature(event, "secret-one")
    assert first == automation.signature(dict(reversed(list(event.items()))), "secret-one")
    assert first != automation.signature(event, "secret-two")
    assert first.startswith("sha256=")


def test_unconfigured_automation_is_an_immediate_no_op():
    assert automation.publish(_record(), environ={}) is False


def test_weak_webhook_secret_fails_before_starting_delivery():
    configured = {
        automation.ENV_WEBHOOK_URL: "http://n8n.local/webhook",
        automation.ENV_WEBHOOK_SECRET: "too-short",
    }
    assert automation.publish(_record(), environ=configured) is False
    assert "at least 32" in automation.status()["last_error"]
