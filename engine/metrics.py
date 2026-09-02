"""A small, tested registry for business definitions the schema cannot infer.

The deterministic planner is still graded with this layer switched off. A
metric is selected only when its exact governed phrase appears and every other
content word is a harmless wrapper ("what is our ..."). That conservative
match is load-bearing: a fixed overall definition must never answer "denial
rate by payer" while silently dropping the breakdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from engine.query import QueryResult, run_query
from engine.sql_guard import validate_sql

METRICS_PATH = Path(__file__).resolve().parent.parent / "metrics.yaml"
REQUIRED = {
    "name", "label", "domain", "owner", "unit", "definition", "phrases",
    "sql", "expect", "derived",
}
WRAPPER_WORDS = {
    "a", "calculate", "current", "give", "is", "me", "our", "overall",
    "please", "report", "show", "s", "tell", "the", "us", "what", "whats",
}


class MetricRegistryError(ValueError):
    """The committed registry is ambiguous or cannot prove one of its entries."""


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    domain: str
    owner: str
    unit: str
    definition: str
    phrases: tuple[str, ...]
    sql: str
    expect: Any
    derived_value: Any
    derived_why: str


@dataclass(frozen=True)
class MetricAnswer:
    metric: Metric
    result: QueryResult

    @property
    def ok(self) -> bool:
        return bool(self.result.ok and self.result.rows)

    @property
    def value(self):
        return self.result.rows[0][0] if self.ok else None

    @property
    def matches_contract(self) -> bool:
        if not self.ok:
            return False
        if isinstance(self.metric.expect, float):
            try:
                return abs(float(self.value) - self.metric.expect) < 0.05
            except (TypeError, ValueError):
                return False
        return self.value == self.metric.expect

    @property
    def headline(self) -> str:
        value = self.value
        if value is None:
            return "—"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return str(value)
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.1f}"

    @property
    def sentence(self) -> str:
        """The certified answer as a sentence, built from the registry entry.

        A governed metric arrives with more than a number: `metrics.yaml` gives
        it a label an owner wrote and a unit. The turn was rendering the bare
        scalar — "8.2" — which is the same defect the compiler path had before
        engine/narrate.py, in the one place the app is most confident.

        Nothing is invented. The label and the unit are committed fields; the
        value is whatever the query just returned. `percent` is the only unit
        rendered as a symbol, because it is the only one where the SQL itself
        did the multiplication.
        """
        if not self.ok:
            return "—"
        rendered = self.headline
        if (self.metric.unit or "").strip().lower() in ("percent", "percentage", "%"):
            rendered = f"{rendered}%"
        label = (self.metric.label or self.metric.name.replace("_", " ")).strip()
        # "The Claim denial rate is" reads as a proper noun that is not one.
        if label[:1].isupper() and not label.split(" ")[0].isupper():
            label = label[0].lower() + label[1:]
        return f"The {label} is {rendered}."


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def load_metrics(path: Path = METRICS_PATH) -> tuple[Metric, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise MetricRegistryError(f"{path.name} must contain at least one metric")
    metrics: list[Metric] = []
    names: set[str] = set()
    phrases: dict[str, str] = {}
    for index, row in enumerate(raw, 1):
        if not isinstance(row, dict):
            raise MetricRegistryError(f"metric {index} is not an object")
        missing = REQUIRED - set(row)
        if missing:
            raise MetricRegistryError(
                f"metric {row.get('name', index)!r} missing {sorted(missing)}"
            )
        name = str(row["name"]).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise MetricRegistryError(f"invalid or duplicate metric name: {name!r}")
        names.add(name)
        metric_phrases = tuple(str(p).strip().lower() for p in row["phrases"] if str(p).strip())
        if not metric_phrases:
            raise MetricRegistryError(f"metric {name!r} has no phrases")
        for phrase in metric_phrases:
            normalized = " ".join(_words(phrase))
            if normalized in phrases:
                raise MetricRegistryError(
                    f"phrase {phrase!r} belongs to both {phrases[normalized]!r} and {name!r}"
                )
            phrases[normalized] = name
        sql = " ".join(str(row["sql"]).split())
        ok, reason = validate_sql(sql)
        if not ok:
            raise MetricRegistryError(f"metric {name!r} has unsafe SQL: {reason}")
        derived = row["derived"]
        if not isinstance(derived, dict) or "value" not in derived or not derived.get("why"):
            raise MetricRegistryError(f"metric {name!r} needs derived.value and derived.why")
        metrics.append(Metric(
            name=name,
            label=str(row["label"]).strip(),
            domain=str(row["domain"]).strip(),
            owner=str(row["owner"]).strip(),
            unit=str(row["unit"]).strip(),
            definition=" ".join(str(row["definition"]).split()),
            phrases=metric_phrases,
            sql=sql,
            expect=row["expect"],
            derived_value=derived["value"],
            derived_why=" ".join(str(derived["why"]).split()),
        ))
    return tuple(metrics)


def match_metric(question: str, registry: tuple[Metric, ...]) -> Metric | None:
    """Return an exact, overall metric match; never discard a qualifier."""
    question_words = _words(question)
    padded = " " + " ".join(question_words) + " "
    candidates: list[tuple[int, Metric, tuple[str, ...]]] = []
    for metric in registry:
        for phrase in metric.phrases:
            phrase_words = _words(phrase)
            if f" {' '.join(phrase_words)} " in padded:
                candidates.append((len(phrase_words), metric, phrase_words))
    if not candidates:
        return None
    _, metric, phrase_words = max(candidates, key=lambda item: item[0])
    remaining = list(question_words)
    # Remove one contiguous occurrence of the selected phrase, then require all
    # remaining words to be conversational wrappers. "by payer" therefore
    # cannot be lost just because "denial rate" also appears.
    for start in range(len(remaining) - len(phrase_words) + 1):
        if tuple(remaining[start:start + len(phrase_words)]) == phrase_words:
            del remaining[start:start + len(phrase_words)]
            break
    if any(word not in WRAPPER_WORDS for word in remaining):
        return None
    return metric


def answer(con, metric: Metric, *, access=None) -> MetricAnswer:
    """Run a certified definition, under the same policy as everything else.

    A governed metric is exactly the kind of number a restricted principal must
    not be able to read around, so the executor's default-deny applies here as
    it does to model- and compiler-authored SQL. The definition being committed
    makes it trustworthy; it does not make it universally visible.
    """
    return MetricAnswer(metric=metric, result=run_query(con, metric.sql, access=access))
