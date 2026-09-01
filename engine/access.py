"""Authenticated principals and fail-closed SQL data authorization.

The public portfolio app deliberately has no login.  In that mode the caller
gets an explicit ``demo`` principal and access to the synthetic catalogue.  A
deployment can switch to ``ASK_AUTH_MODE=oidc``; then every request must carry a
cryptographically verified bearer JWT and at least one role must be present in
the configured policy file.  Missing identity, an identity with no mapped role,
an unparseable statement, an unauthorized relation, and a sensitive-column
reference all deny access before DuckDB executes the statement.

This is an application policy-enforcement point, not a substitute for RLS or
masked views in the production database.  The same grants should be enforced by
the database credentials used by this process so a future execution path cannot
bypass them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from data_manifest import MANIFEST, table_name
from engine.verify import DOMAIN_OF, parse_sql

AUTH_DISABLED = "disabled"
AUTH_OIDC = "oidc"
SUPPORTED_AUTH_MODES = {AUTH_DISABLED, AUTH_OIDC}
ALL_TABLES = frozenset(
    table_name(domain, table) for domain, table, _source, _description in MANIFEST
)
# Table functions do not produce BASE_TABLE nodes. Keep a deliberately tiny
# allow-list of functions that only expand values already present in the query;
# dynamic SQL, catalogue readers, extension scans, and future DuckDB functions
# are denied by default at the policy boundary.
SAFE_TABLE_FUNCTIONS = frozenset({"generate_series", "range", "unnest"})


class AuthenticationError(RuntimeError):
    """The request does not carry a valid identity for the configured mode."""


class PolicyConfigurationError(RuntimeError):
    """The authorization policy is missing, malformed, or unsafe to apply."""


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str = ""
    roles: frozenset[str] = field(default_factory=frozenset)
    authenticated: bool = False
    claims: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def demo(cls, subject: str) -> "Principal":
        return cls(subject=subject, roles=frozenset({"demo"}), authenticated=False)


@dataclass(frozen=True)
class AccessScope:
    """The immutable authorization decision attached to one request/session."""

    principal: Principal
    allowed_tables: frozenset[str]
    denied_columns: tuple[tuple[str, tuple[str, ...]], ...] = ()
    policy_version: str = "demo"

    @property
    def denied_by_table(self) -> dict[str, frozenset[str]]:
        return {table: frozenset(columns) for table, columns in self.denied_columns}

    @property
    def fingerprint(self) -> tuple:
        return (
            self.principal.subject,
            self.principal.tenant_id,
            tuple(sorted(self.principal.roles)),
            tuple(sorted(self.allowed_tables)),
            self.denied_columns,
            self.policy_version,
        )

    @classmethod
    def demo(cls, subject: str) -> "AccessScope":
        return cls(Principal.demo(subject), ALL_TABLES)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str = ""
    tables: tuple[str, ...] = ()


def auth_mode() -> str:
    mode = os.environ.get("ASK_AUTH_MODE", AUTH_DISABLED).strip().lower()
    if mode not in SUPPORTED_AUTH_MODES:
        raise PolicyConfigurationError(
            f"ASK_AUTH_MODE must be one of {sorted(SUPPORTED_AUTH_MODES)}, got {mode!r}"
        )
    return mode


def _claim_values(claims: Mapping[str, Any], name: str) -> frozenset[str]:
    value = claims.get(name, [])
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise AuthenticationError(f"OIDC claim {name!r} must be a string or list")
    return frozenset(str(item).strip() for item in value if str(item).strip())


def principal_from_claims(claims: Mapping[str, Any]) -> Principal:
    subject = str(claims.get("sub", "")).strip()
    if not subject:
        raise AuthenticationError("OIDC token is missing the required subject claim")
    roles_claim = os.environ.get("ASK_OIDC_ROLES_CLAIM", "roles").strip()
    tenant_claim = os.environ.get("ASK_OIDC_TENANT_CLAIM", "tenant_id").strip()
    return Principal(
        subject=subject,
        tenant_id=str(claims.get(tenant_claim, "")).strip(),
        roles=_claim_values(claims, roles_claim),
        authenticated=True,
        claims=dict(claims),
    )


def _required_oidc_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise PolicyConfigurationError(f"{name} is required when ASK_AUTH_MODE=oidc")
    return value


@lru_cache(maxsize=4)
def _jwk_client(url: str):
    try:
        from jwt import PyJWKClient
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise PolicyConfigurationError(
            "PyJWT[crypto] is required when ASK_AUTH_MODE=oidc"
        ) from exc
    return PyJWKClient(url, cache_keys=True, lifespan=300)


def authenticate_headers(headers: Mapping[str, Any]) -> Principal:
    """Verify an OIDC bearer token and convert its claims into a principal."""
    if auth_mode() != AUTH_OIDC:
        raise AuthenticationError("OIDC authentication is not enabled")
    issuer = _required_oidc_setting("ASK_OIDC_ISSUER")
    audience = _required_oidc_setting("ASK_OIDC_AUDIENCE")
    jwks_url = _required_oidc_setting("ASK_OIDC_JWKS_URL")
    header_name = os.environ.get("ASK_AUTH_HEADER", "Authorization").strip()
    raw_value = headers.get(header_name)
    if raw_value is None:
        raw_value = next(
            (
                value
                for name, value in headers.items()
                if str(name).lower() == header_name.lower()
            ),
            "",
        )
    raw = str(raw_value or "").strip()
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()
    if not raw:
        raise AuthenticationError(f"request is missing a bearer token in {header_name}")
    try:
        import jwt

        signing_key = _jwk_client(jwks_url).get_signing_key_from_jwt(raw)
        claims = jwt.decode(
            raw,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
    except Exception as exc:
        raise AuthenticationError(f"bearer token validation failed: {exc}") from exc
    return principal_from_claims(claims)


def _as_names(value: Any, *, field_name: str) -> frozenset[str]:
    if value == "*":
        return frozenset({"*"})
    if not isinstance(value, list):
        raise PolicyConfigurationError(f"{field_name} must be a list or '*'")
    return frozenset(str(item).strip().lower() for item in value if str(item).strip())


@dataclass(frozen=True)
class PolicySet:
    version: str
    roles: Mapping[str, frozenset[str]]
    sensitive_columns: Mapping[str, Mapping[str, frozenset[str]]]

    def scope_for(self, principal: Principal) -> AccessScope:
        allowed: set[str] = set()
        known_role = False
        for role in principal.roles:
            grants = self.roles.get(role)
            if grants is None:
                continue
            known_role = True
            if "*" in grants:
                allowed.update(ALL_TABLES)
                continue
            for grant in grants:
                if grant in DOMAIN_OF:
                    allowed.add(grant)
                else:
                    allowed.update(table for table, domain in DOMAIN_OF.items() if domain == grant)
        # Default deny.  An identity with no mapped role receives no catalogue,
        # even if another unrecognized claim happens to look privileged.
        if not known_role:
            allowed.clear()

        denied: list[tuple[str, tuple[str, ...]]] = []
        for table, columns in self.sensitive_columns.items():
            blocked = []
            for column, allow_roles in columns.items():
                if not (principal.roles & allow_roles):
                    blocked.append(column)
            if blocked:
                denied.append((table, tuple(sorted(blocked))))
        return AccessScope(
            principal=principal,
            allowed_tables=frozenset(allowed),
            denied_columns=tuple(sorted(denied)),
            policy_version=self.version,
        )


def load_policy(path: str | Path) -> PolicySet:
    policy_path = Path(path)
    try:
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PolicyConfigurationError(f"cannot read policy {policy_path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") is None:
        raise PolicyConfigurationError("policy must be an object with a version")
    raw_roles = raw.get("roles")
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise PolicyConfigurationError("policy.roles must contain at least one role")
    roles: dict[str, frozenset[str]] = {}
    for role, grants in raw_roles.items():
        if not isinstance(grants, dict):
            raise PolicyConfigurationError(f"role {role!r} must be an object")
        domains = _as_names(grants.get("domains", []), field_name=f"roles.{role}.domains")
        tables = _as_names(grants.get("tables", []), field_name=f"roles.{role}.tables")
        combined = domains | tables
        invalid = {item for item in combined if item != "*" and item not in DOMAIN_OF
                   and item not in set(DOMAIN_OF.values())}
        if invalid:
            raise PolicyConfigurationError(f"role {role!r} has unknown grants: {sorted(invalid)}")
        roles[str(role)] = combined

    sensitive: dict[str, dict[str, frozenset[str]]] = {}
    raw_sensitive = raw.get("sensitive_columns", {})
    if not isinstance(raw_sensitive, dict):
        raise PolicyConfigurationError("policy.sensitive_columns must be an object")
    for table, columns in raw_sensitive.items():
        table = str(table).lower()
        if table not in ALL_TABLES or not isinstance(columns, dict):
            raise PolicyConfigurationError(f"invalid sensitive-column table: {table!r}")
        sensitive[table] = {}
        for column, rule in columns.items():
            if not isinstance(rule, dict):
                raise PolicyConfigurationError(f"sensitive rule {table}.{column} must be an object")
            allow_roles = _as_names(
                rule.get("allow_roles", []),
                field_name=f"sensitive_columns.{table}.{column}.allow_roles",
            )
            unknown = allow_roles - set(roles)
            if unknown:
                raise PolicyConfigurationError(
                    f"sensitive rule {table}.{column} has unknown roles: {sorted(unknown)}"
                )
            sensitive[table][str(column).lower()] = allow_roles
    return PolicySet(str(raw["version"]), roles, sensitive)


@lru_cache(maxsize=8)
def _load_policy_cached(path: str, modified_ns: int) -> PolicySet:
    del modified_ns  # part of the cache key, not the document
    return load_policy(path)


def access_scope(principal: Principal) -> AccessScope:
    if auth_mode() == AUTH_DISABLED:
        return AccessScope.demo(principal.subject)
    path = _required_oidc_setting("ASK_POLICY_FILE")
    policy_path = Path(path).resolve()
    try:
        modified_ns = policy_path.stat().st_mtime_ns
    except OSError as exc:
        raise PolicyConfigurationError(f"cannot stat policy {policy_path}: {exc}") from exc
    return _load_policy_cached(str(policy_path), modified_ns).scope_for(principal)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _cte_names(ast) -> set[str]:
    names: set[str] = set()
    for node in _walk(ast):
        cte_map = node.get("cte_map")
        if not isinstance(cte_map, dict):
            continue
        for entry in cte_map.get("map") or []:
            if entry.get("key"):
                names.add(str(entry["key"]).lower())
    return names


def validate_scope_schema(con, scope: AccessScope) -> None:
    """Fail startup when policy masking refers to a column that does not exist."""
    for table, denied_columns in scope.denied_columns:
        try:
            actual = {str(row[0]).lower() for row in con.cursor().execute(
                f'DESCRIBE "{table}"'
            ).fetchall()}
        except Exception as exc:
            raise PolicyConfigurationError(
                f"cannot validate policy relation {table}: {exc}"
            ) from exc
        missing = set(denied_columns) - actual
        if missing:
            raise PolicyConfigurationError(
                f"policy references unknown column(s) on {table}: {sorted(missing)}"
            )


def authorize_sql(con, sql: str, scope: AccessScope | None) -> AccessDecision:
    """Authorize all warehouse relations and sensitive-column references.

    A ``None`` scope is retained for internal library callers and existing
    offline contracts.  Network entry points pass an explicit scope.
    """
    if scope is None:
        return AccessDecision(True)
    ast = parse_sql(con, sql)
    if ast is None:
        return AccessDecision(False, "authorization could not parse the statement")

    bindings: dict[str, str] = {}
    tables: set[str] = set()
    unknown_relations: set[str] = set()
    ctes = _cte_names(ast)
    for node in _walk(ast):
        if node.get("type") == "TABLE_FUNCTION":
            function = node.get("function")
            name = (
                str(function.get("function_name") or "").lower()
                if isinstance(function, dict)
                else ""
            )
            if name not in SAFE_TABLE_FUNCTIONS:
                unknown_relations.add(f"{name or '<unknown table function>'}()")
            continue
        if node.get("type") != "BASE_TABLE":
            continue
        name = str(node.get("table_name") or "").lower()
        if name in ctes:
            continue
        if name not in ALL_TABLES:
            schema = str(node.get("schema_name") or "").lower()
            unknown_relations.add(f"{schema}.{name}" if schema else name)
            continue
        alias = str(node.get("alias") or name).lower()
        bindings[alias] = name
        bindings.setdefault(name, name)
        tables.add(name)
    if unknown_relations:
        return AccessDecision(
            False,
            "relation(s) are outside the governed catalogue: "
            + ", ".join(sorted(unknown_relations)),
            tuple(sorted(tables)),
        )
    denied_tables = sorted(tables - set(scope.allowed_tables))
    if denied_tables:
        return AccessDecision(
            False,
            "principal is not authorized for relation(s): " + ", ".join(denied_tables),
            tuple(sorted(tables)),
        )

    denied = scope.denied_by_table
    sensitive = {column for table in tables for column in denied.get(table, ())}
    for node in _walk(ast):
        names = node.get("column_names") if node.get("class") == "COLUMN_REF" else None
        if not names:
            continue
        column = str(names[-1]).lower()
        if column not in sensitive:
            continue
        if len(names) >= 2:
            table = bindings.get(str(names[-2]).lower())
            if table and column not in denied.get(table, ()):
                continue
        return AccessDecision(
            False,
            f"principal is not authorized for sensitive column: {column}",
            tuple(sorted(tables)),
        )

    # SELECT * is denied when any source contains a masked column.  This is
    # intentionally conservative: even ``* EXCLUDE`` is refused because future
    # schema changes must not silently widen an allow decision.
    if sensitive:
        for node in _walk(ast):
            select_list = node.get("select_list")
            if not isinstance(select_list, list):
                continue
            if any(isinstance(expr, dict) and expr.get("class") == "STAR"
                   for expr in select_list):
                return AccessDecision(
                    False,
                    "SELECT * is not allowed on a relation with sensitive columns",
                    tuple(sorted(tables)),
                )
    return AccessDecision(True, tables=tuple(sorted(tables)))
