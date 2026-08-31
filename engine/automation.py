"""Optional, non-blocking operational events for a self-hosted n8n workflow.

The analytics request path must never depend on an automation server. Events
therefore enter a bounded queue and a daemon delivers them with a short timeout.
The payload deliberately omits question text, SQL, results, reasons, and retry
feedback: n8n is an operations integration, not a second audit-data store.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import threading
import urllib.request

ENV_WEBHOOK_URL = "ASK_N8N_WEBHOOK_URL"
ENV_WEBHOOK_SECRET = "ASK_N8N_WEBHOOK_SECRET"
QUEUE_SIZE = 128

_queue: queue.Queue[dict] = queue.Queue(maxsize=QUEUE_SIZE)
_lock = threading.Lock()
_worker: threading.Thread | None = None
_status = {"queued": 0, "delivered": 0, "failed": 0, "dropped": 0, "last_error": ""}


def webhook_url(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return str(env.get(ENV_WEBHOOK_URL, "") or "").strip()


def _value(record, name: str, default=None):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def operational_event(record) -> dict:
    """Return the strict allow-list sent to automation infrastructure."""
    actor = str(_value(record, "actor", ""))
    ts = str(_value(record, "ts", ""))
    engine = str(_value(record, "engine", ""))
    event_id = hashlib.sha256(f"{ts}\0{actor}\0{engine}".encode()).hexdigest()[:24]
    return {
        "schema_version": 1,
        "event_id": event_id,
        "ts": ts,
        "actor_hash": hashlib.sha256(actor.encode()).hexdigest()[:16] if actor else "",
        "role": str(_value(record, "role", "")),
        "engine": engine,
        "outcome": str(_value(record, "outcome", "")),
        "refusal_kind": str(_value(record, "refusal_kind", "")),
        "row_count": int(_value(record, "row_count", 0) or 0),
        "truncated": bool(_value(record, "truncated", False)),
        "attempts": int(_value(record, "attempts", 1) or 1),
        "guard_ok": bool(_value(record, "guard_ok", True)),
        "tokens_in": int(_value(record, "tokens_in", 0) or 0),
        "tokens_out": int(_value(record, "tokens_out", 0) or 0),
        # Integer milliseconds keep the signed JSON canonical across Python and
        # JavaScript (`125.0` is encoded as 125.0 by Python but 125 by JS).
        "elapsed_ms": int(round(float(_value(record, "elapsed_ms", 0.0) or 0.0))),
    }


def _encoded(event: dict) -> bytes:
    return json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")


def signature(event: dict, secret: str) -> str:
    digest = hmac.new(str(secret).encode("utf-8"), _encoded(event), hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _deliver(item: dict) -> None:
    event = item["event"]
    data = _encoded(event)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ask-your-data/1",
        "X-Ask-Signature": signature(event, item["secret"]),
    }
    request = urllib.request.Request(item["url"], data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=3.0) as response:
        response.read(256)


def _run() -> None:
    while True:
        item = _queue.get()
        try:
            _deliver(item)
        except Exception as exc:  # noqa: BLE001 - automation cannot break analytics
            with _lock:
                _status["failed"] += 1
                _status["last_error"] = str(exc)[:200]
        else:
            with _lock:
                _status["delivered"] += 1
                _status["last_error"] = ""
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_run, name="ask-your-data-n8n", daemon=True,
            )
            _worker.start()


def publish(record, environ: dict[str, str] | None = None) -> bool:
    """Queue one safe event. Return immediately and never raise."""
    env = os.environ if environ is None else environ
    url = webhook_url(env)
    if not url:
        return False
    secret = str(env.get(ENV_WEBHOOK_SECRET, "") or "")
    if not secret:
        with _lock:
            _status["failed"] += 1
            _status["last_error"] = f"{ENV_WEBHOOK_SECRET} is not configured"
        return False
    _ensure_worker()
    try:
        _queue.put_nowait({"url": url, "secret": secret, "event": operational_event(record)})
    except queue.Full:
        with _lock:
            _status["dropped"] += 1
        return False
    with _lock:
        _status["queued"] += 1
    return True


def status() -> dict:
    with _lock:
        return {**_status, "pending": _queue.qsize(), "configured": bool(webhook_url())}
