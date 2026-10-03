"""Compose explicit scopes around certified SQL without replacing its formula.

Only two policy-owned healthcare definitions opt in. One breakdown and one
exact catalog-value filter are supported. Every other word must be accounted
for; unsupported dates, rankings and multiple metrics are rejected. The raw
schema compiler's benchmark stays independent of this policy layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from engine.metrics import WRAPPER_WORDS, Metric, _words
from engine.query import run_query


class MetricScopeError(ValueError):
    """A named definition cannot preserve every requested qualifier."""


@dataclass(frozen=True)
class Dimension:
    table: str
    key: str
    column: str
    label: str
    phrases: tuple[str, ...]


DIMENSIONS = (
    Dimension("healthcare_dim_payer", "payer_id", "payer_type", "payer type",
              ("payer type", "payer types")),
    Dimension("healthcare_dim_payer", "payer_id", "payer_name", "payer",
              ("payer", "payers", "payer name", "payer names")),
    Dimension("healthcare_dim_provider", "provider_id", "specialty", "specialty",
              ("specialty", "specialties", "provider specialty", "provider specialties")),
)
SUPPORTED = frozenset({"denial_rate", "net_collection_rate"})
FORMULA_COLUMNS = {
    "denial_rate": ("status",),
    "net_collection_rate": ("status", "paid_amount", "allowed_amount"),
}
SOURCE = "healthcare_fact_claims"


def _named(question: str, registry: tuple[Metric, ...]):
    words = _words(question)
    matches = []
    for metric in registry:
        for phrase in metric.phrases:
            tokens = _words(phrase)
            for start in range(len(words) - len(tokens) + 1):
                if words[start:start + len(tokens)] == tokens:
                    matches.append((len(tokens), start, metric, tokens))
    if not matches or not any(m[2].name in SUPPORTED for m in matches):
        return None
    if len({m[2].name for m in matches}) != 1:
        raise MetricScopeError("Ask for one certified metric at a time.")
    _, start, metric, tokens = max(matches, key=lambda m: m[0])
    if any(w not in WRAPPER_WORDS for w in words[:start]):
        raise MetricScopeError("I could not preserve the requested ranking or time window.")
    return metric, words[start + len(tokens):]


def _dimension(tokens: tuple[str, ...]):
    matches = [(len(_words(p)), d) for d in DIMENSIONS for p in d.phrases
               if tokens[:len(_words(p))] == _words(p)]
    return max(matches, key=lambda match: match[0]) if matches else (0, None)


def _catalog(con, dimension: Dimension, access, deadline):
    result = run_query(con, f'SELECT DISTINCT "{dimension.column}" FROM "{dimension.table}"',
                       access=access, deadline=deadline)
    if not result.ok or result.truncated:
        raise MetricScopeError("The authorized catalog could not verify this filter.")
    return {value[0] for value in result.rows if value[0] is not None}


def _filter(con, tokens, access, deadline):
    size, explicit = _dimension(tokens)
    value_tokens = tokens[size:] if explicit else tokens
    candidates = []
    for dim in (explicit,) if explicit else DIMENSIONS:
        for value in _catalog(con, dim, access, deadline):
            if _words(str(value)) == value_tokens:
                candidates.append((dim, value))
    if not candidates:
        raise MetricScopeError("Use an exact payer, payer type or specialty from the data catalog.")
    # Medicare/Medicaid/Self-Pay are both payer names and their own types in this
    # catalog. Prefer the declared type; a match in another domain is ambiguous.
    if len({d.table for d, _ in candidates}) > 1:
        raise MetricScopeError("Name the filter explicitly as a payer, payer type or specialty.")
    return candidates[0]


def match_scoped_metric(question: str, registry: tuple[Metric, ...], con, *, access=None,
                        deadline=None) -> Metric | None:
    named = _named(question, registry)
    if named is None:
        return None
    metric, remainder = named
    group, filtered = None, None
    while remainder:
        operator, *tail = remainder
        if operator not in {"by", "for", "in"}:
            raise MetricScopeError(
                "I could not preserve every filter or breakdown in this question.")
        boundary = next((i for i, w in enumerate(tail) if w in {"by", "for", "in"}), len(tail))
        clause, remainder = tuple(tail[:boundary]), tuple(tail[boundary:])
        if operator == "by" and group is None:
            size, group = _dimension(clause)
            if group is None or size != len(clause):
                raise MetricScopeError("Supported breakdowns are payer, payer type and specialty.")
        elif operator in {"for", "in"} and filtered is None:
            filtered = _filter(con, clause, access, deadline)
        else:
            raise MetricScopeError("Ask for one breakdown and one catalog-value filter at a time.")
    if group is None and filtered is None:
        return None

    used = tuple(dict.fromkeys(d for d in (group, filtered[0] if filtered else None) if d))
    joins = []
    columns = [f'f."{column}"' for column in FORMULA_COLUMNS[metric.name]]
    aliases = {}
    for dim in used:
        if dim.table not in aliases:
            alias = f"d{len(aliases)}"
            aliases[dim.table] = alias
            integrity = run_query(
                con, f'SELECT COUNT(*) - COUNT(DISTINCT "{dim.key}") FROM "{dim.table}"',
                access=access, deadline=deadline)
            if not integrity.ok or not integrity.rows or integrity.rows[0][0] != 0:
                raise MetricScopeError(
                    "The dimension join is not verifiably unique; no rate was calculated.")
            joins.append(f'LEFT JOIN "{dim.table}" {alias} ON f."{dim.key}" = {alias}."{dim.key}"')
    if group:
        columns.append(f'{aliases[group.table]}."{group.column}" AS __metric_group')
    if filtered:
        dim, _ = filtered
        columns.append(f'{aliases[dim.table]}."{dim.column}" AS __metric_filter')
    source = f'(SELECT {", ".join(columns)} FROM "{SOURCE}" f {" ".join(joins)}) metric_scope'
    marker = re.compile(rf"\bFROM\s+{SOURCE}\b", re.I)
    if len(marker.findall(metric.sql)) != 1:
        raise MetricScopeError("This definition does not support scoped queries.")
    sql = marker.sub(lambda _: f"FROM {source}", metric.sql)
    scope = []
    if filtered:
        dim, value = filtered
        literal = str(value).replace("'", "''")
        trailing = metric.sql[marker.search(metric.sql).end():]
        sql += (" AND " if re.search(r"\bWHERE\b", trailing, re.I) else " WHERE ")
        sql += f"__metric_filter = '{literal}'"
        scope.append(f"for {value}")
    if group:
        sql = re.sub(r"^SELECT\s+", f'SELECT __metric_group AS "{group.column}", ', sql, flags=re.I)
        sql += " GROUP BY __metric_group ORDER BY __metric_group NULLS LAST"
        scope.insert(0, f"by {group.label}")
    return replace(metric, sql=sql, scope=" ".join(scope), group_by=group.label if group else "")
