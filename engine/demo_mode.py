"""
Key-free demo mode.

The live assistant needs an API key: the model writes the SQL, and without a key
there is no model. But the repo already carries something that answers the same
questions honestly without one -- `evals/golden_questions.yaml`, the accuracy
contract, where each question is paired with reference SQL and the answer that
SQL must produce.

Demo mode serves those. Clicking a question runs its reference SQL live against
DuckDB and shows the result and the query. Nothing is cached, nothing is
hard-coded: if the vendored data drifts, the demo changes with it and
`tests/test_golden_sql.py` goes red.

What this is NOT is the model writing SQL, and the UI says so plainly. Passing
pre-registered SQL off as a live natural-language answer would be exactly the
kind of thing this project exists to argue against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from engine.query import QueryResult, run_query

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
GOLDEN_PATH = EVALS_DIR / "golden_questions.yaml"

DEMO_NOTICE = (
    "Demo mode — no API key is configured, so the model is not running. These "
    "questions come from the project's accuracy contract "
    "(`evals/golden_questions.yaml`): each one ships with reference SQL that CI "
    "re-runs on every push. Clicking a question executes that SQL live against "
    "DuckDB. The natural-language layer — where the model writes the SQL itself "
    "— is what needs the key."
)


@dataclass
class DemoAnswer:
    """One golden question, answered by executing its reference SQL."""

    question: str
    domain: str
    sql: str
    result: QueryResult
    expected: Any

    @property
    def ok(self) -> bool:
        return bool(self.result.ok and self.result.rows)

    @property
    def headline(self) -> str:
        """The scalar answer, formatted for display."""
        if not self.ok:
            return "—"
        value = self.result.rows[0][0]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return str(value)
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,}"

    @property
    def matches_contract(self) -> bool:
        """
        Did the live query return what the contract says it should?

        Surfaced in the UI rather than hidden, because "the number you are
        looking at is the number CI asserts" is the whole point of showing it.
        """
        if not self.ok:
            return False
        value = self.result.rows[0][0]
        if isinstance(self.expected, float):
            try:
                return abs(float(value) - self.expected) < 0.05
            except (TypeError, ValueError):
                return False
        return value == self.expected


def load_golden_questions(path: Path = GOLDEN_PATH) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError(f"{path.name} is empty")
    for case in cases:
        missing = {"id", "domain", "question", "sql", "expect"} - set(case)
        if missing:
            raise ValueError(f"golden question {case.get('id', '?')} missing {sorted(missing)}")
    return cases


def questions_by_domain(cases: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for case in cases:
        grouped.setdefault(case["domain"], []).append(case)
    return grouped


def answer(con, case: dict) -> DemoAnswer:
    """Execute a golden question's reference SQL and wrap the result."""
    return DemoAnswer(
        question=case["question"],
        domain=case["domain"],
        sql=case["sql"].strip(),
        result=run_query(con, case["sql"]),
        expected=case["expect"],
    )


def has_api_key(environ: dict[str, str] | None = None) -> bool:
    """Whether a live model is reachable. The only switch between the two modes."""
    import os

    env = environ if environ is not None else os.environ
    return bool(env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"))
