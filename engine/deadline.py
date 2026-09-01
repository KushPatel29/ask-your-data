"""One monotonic deadline shared by every stage of a request."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

DEFAULT_REQUEST_TIMEOUT_S = 45.0
MIN_REQUEST_TIMEOUT_S = 1.0
MAX_REQUEST_TIMEOUT_S = 300.0


class DeadlineExpired(TimeoutError):
    def __init__(self, stage: str, timeout_s: float):
        self.stage = stage
        self.timeout_s = timeout_s
        super().__init__(
            f"request exceeded its {timeout_s:g}s end-to-end deadline during {stage}"
        )


def configured_timeout_s() -> float:
    raw = os.environ.get("ASK_REQUEST_TIMEOUT_S", str(DEFAULT_REQUEST_TIMEOUT_S))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("ASK_REQUEST_TIMEOUT_S must be a number") from exc
    if not MIN_REQUEST_TIMEOUT_S <= value <= MAX_REQUEST_TIMEOUT_S:
        raise ValueError(
            f"ASK_REQUEST_TIMEOUT_S must be between {MIN_REQUEST_TIMEOUT_S:g} "
            f"and {MAX_REQUEST_TIMEOUT_S:g} seconds"
        )
    return value


@dataclass(frozen=True)
class RequestDeadline:
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    started_at: float = field(default_factory=time.monotonic)

    @classmethod
    def configured(cls) -> "RequestDeadline":
        return cls(configured_timeout_s())

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.timeout_s - self.elapsed_s)

    def require(self, stage: str) -> float:
        remaining = self.remaining_s
        if remaining <= 0:
            raise DeadlineExpired(stage, self.timeout_s)
        return remaining
