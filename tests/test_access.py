from __future__ import annotations

import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from engine.access import (
    AccessScope,
    AuthenticationError,
    PolicyConfigurationError,
    Principal,
    access_scope,
    authenticate_headers,
    authorize_sql,
    load_policy,
    principal_from_claims,
    validate_scope_schema,
)
from engine.query import run_query

POLICY = """
version: test-v1
roles:
  healthcare_analyst:
    domains: [healthcare]
    tables: []
  people_analyst:
    domains: [hr]
    tables: []
  compensation_admin:
    domains: [hr]
    tables: []
sensitive_columns:
  hr_fact_employees:
    base_salary:
      allow_roles: [compensation_admin]
"""


@pytest.fixture()
def policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY, encoding="utf-8")
    return path


def test_claims_become_an_authenticated_principal(monkeypatch):
    monkeypatch.setenv("ASK_OIDC_ROLES_CLAIM", "groups")
    monkeypatch.setenv("ASK_OIDC_TENANT_CLAIM", "organization")
    principal = principal_from_claims({
        "sub": "analyst-123",
        "groups": ["healthcare_analyst", "auditor"],
        "organization": "tenant-a",
    })
    assert principal.authenticated
    assert principal.subject == "analyst-123"
    assert principal.tenant_id == "tenant-a"
    assert principal.roles == frozenset({"healthcare_analyst", "auditor"})


def test_missing_subject_is_never_accepted():
    with pytest.raises(AuthenticationError, match="subject"):
        principal_from_claims({"roles": ["platform_admin"]})


def test_unknown_policy_grant_fails_configuration(policy_file: Path):
    bad = policy_file.with_name("bad.yaml")
    bad.write_text(POLICY.replace("[healthcare]", "[not_a_domain]"), encoding="utf-8")
    with pytest.raises(PolicyConfigurationError, match="unknown grants"):
        load_policy(bad)


def test_unknown_role_is_default_deny(policy_file: Path):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"made_up_admin"}), authenticated=True)
    )
    assert scope.allowed_tables == frozenset()


def test_extra_identity_group_does_not_erase_a_mapped_role(policy_file: Path):
    scope = load_policy(policy_file).scope_for(
        Principal(
            "subject",
            roles=frozenset({"healthcare_analyst", "unrelated_identity_group"}),
            authenticated=True,
        )
    )
    assert "healthcare_fact_claims" in scope.allowed_tables
    assert not any(table.startswith("hr_") for table in scope.allowed_tables)


def test_role_matching_is_case_insensitive_and_normalized(tmp_path: Path):
    policy = tmp_path / "case-policy.yaml"
    policy.write_text(POLICY.replace("healthcare_analyst:", "Healthcare_Analyst:"),
                      encoding="utf-8")

    scope = load_policy(policy).scope_for(
        Principal("subject", roles=frozenset({"HEALTHCARE_ANALYST"}), authenticated=True)
    )

    assert "healthcare_fact_claims" in scope.allowed_tables


def test_role_limits_relations_before_execution(con, policy_file: Path):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"healthcare_analyst"}), authenticated=True)
    )
    allowed = run_query(con, "SELECT COUNT(*) FROM healthcare_fact_claims", access=scope)
    denied = run_query(con, "SELECT COUNT(*) FROM hr_fact_employees", access=scope)
    assert allowed.ok
    assert not denied.ok
    assert denied.policy_denied
    assert "hr_fact_employees" in denied.error


def test_sensitive_column_and_star_are_denied(con, policy_file: Path):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"people_analyst"}), authenticated=True)
    )
    explicit = authorize_sql(
        con, "SELECT base_salary FROM hr_fact_employees", scope,
    )
    wildcard = authorize_sql(con, "SELECT * FROM hr_fact_employees", scope)
    aggregate = authorize_sql(con, "SELECT COUNT(*) FROM hr_fact_employees", scope)
    assert not explicit.allowed and "base_salary" in explicit.reason
    assert not wildcard.allowed and "SELECT *" in wildcard.reason
    assert aggregate.allowed


def test_sensitive_role_can_read_protected_column(con, policy_file: Path):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"compensation_admin"}), authenticated=True)
    )
    result = run_query(
        con,
        "SELECT ROUND(AVG(base_salary), 2) FROM hr_fact_employees",
        access=scope,
    )
    assert result.ok


def test_information_schema_cannot_bypass_catalogue_policy(con, policy_file: Path):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"healthcare_analyst"}), authenticated=True)
    )
    result = run_query(
        con,
        "SELECT table_name FROM information_schema.tables",
        access=scope,
    )
    assert result.policy_denied
    assert "outside the governed catalogue" in result.error


@pytest.mark.parametrize(
    "relation",
    ["temp.healthcare_fact_claims", "other.main.healthcare_fact_claims"],
)
def test_allowlisted_name_in_an_unmanaged_schema_or_catalog_is_denied(
    con, policy_file: Path, relation: str,
):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"healthcare_analyst"}), authenticated=True)
    )

    decision = authorize_sql(con, f"SELECT COUNT(*) FROM {relation}", scope)

    assert not decision.allowed
    assert "outside the governed catalogue" in decision.reason


def test_policy_column_typo_fails_schema_validation(con, policy_file: Path):
    bad = policy_file.with_name("bad-column.yaml")
    bad.write_text(POLICY.replace("base_salary", "base_sallary"), encoding="utf-8")
    scope = load_policy(bad).scope_for(
        Principal("subject", roles=frozenset({"people_analyst"}), authenticated=True)
    )
    with pytest.raises(PolicyConfigurationError, match="unknown column"):
        validate_scope_schema(con, scope)


def test_oidc_mode_requires_policy_file(monkeypatch):
    monkeypatch.setenv("ASK_AUTH_MODE", "oidc")
    monkeypatch.delenv("ASK_POLICY_FILE", raising=False)
    with pytest.raises(PolicyConfigurationError, match="ASK_POLICY_FILE"):
        access_scope(Principal("subject", authenticated=True))


def test_oidc_bearer_signature_issuer_audience_and_required_claims(monkeypatch):
    monkeypatch.setenv("ASK_AUTH_MODE", "oidc")
    monkeypatch.setenv("ASK_OIDC_ISSUER", "https://id.example/")
    monkeypatch.setenv("ASK_OIDC_AUDIENCE", "ask-your-data")
    monkeypatch.setenv("ASK_OIDC_JWKS_URL", "https://id.example/jwks")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class SigningKey:
        key = private_key.public_key()

    class JWKClient:
        @staticmethod
        def get_signing_key_from_jwt(_token):
            return SigningKey()

    monkeypatch.setattr("engine.access._jwk_client", lambda _url: JWKClient())
    now = int(time.time())
    token = jwt.encode({
        "sub": "user-42",
        "roles": ["healthcare_analyst"],
        "tenant_id": "tenant-a",
        "iss": "https://id.example/",
        "aud": "ask-your-data",
        "iat": now,
        "exp": now + 300,
    }, private_key, algorithm="RS256", headers={"kid": "test-key"})
    # HTTP field names are case-insensitive; reverse proxies commonly normalize
    # this one to lowercase before the application receives it.
    principal = authenticate_headers({"authorization": f"Bearer {token}"})
    assert principal.subject == "user-42"
    assert principal.tenant_id == "tenant-a"
    assert principal.authenticated


def test_oidc_never_accepts_token_selected_hmac_algorithm(monkeypatch):
    monkeypatch.setenv("ASK_AUTH_MODE", "oidc")
    monkeypatch.setenv("ASK_OIDC_ISSUER", "https://id.example/")
    monkeypatch.setenv("ASK_OIDC_AUDIENCE", "ask-your-data")
    monkeypatch.setenv("ASK_OIDC_JWKS_URL", "https://id.example/jwks")

    secret = "shared-secret-that-is-at-least-32-bytes"

    class SigningKey:
        key = secret

    class JWKClient:
        @staticmethod
        def get_signing_key_from_jwt(_token):
            return SigningKey()

    monkeypatch.setattr("engine.access._jwk_client", lambda _url: JWKClient())
    now = int(time.time())
    token = jwt.encode({
        "sub": "attacker", "iss": "https://id.example/", "aud": "ask-your-data",
        "iat": now, "exp": now + 300,
    }, secret, algorithm="HS256")
    with pytest.raises(AuthenticationError, match="validation failed"):
        authenticate_headers({"Authorization": f"Bearer {token}"})


def test_demo_scope_is_explicit_not_authenticated():
    scope = AccessScope.demo("session-abc")
    assert not scope.principal.authenticated
    assert scope.principal.roles == frozenset({"demo"})
    assert scope.allowed_tables


@pytest.mark.parametrize(
    "sql, hidden",
    [
        ("SELECT COUNT(*) FROM query_table('hr_fact_employees')", "query_table"),
        ("SELECT * FROM duckdb_tables() LIMIT 1", "duckdb_tables"),
    ],
)
def test_table_functions_cannot_bypass_relation_policy(con, policy_file, sql, hidden):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"healthcare_analyst"}), authenticated=True)
    )
    decision = authorize_sql(con, sql, scope)
    assert not decision.allowed
    assert hidden in decision.reason


def test_cte_cannot_shadow_an_unauthorized_governed_table(con, policy_file):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"healthcare_analyst"}), authenticated=True)
    )
    sql = """
        WITH hr_fact_employees AS (
            SELECT employee_id FROM hr_fact_employees
        )
        SELECT COUNT(*) FROM hr_fact_employees
    """

    decision = authorize_sql(con, sql, scope)

    assert not decision.allowed
    assert "hr_fact_employees" in decision.reason

    qualified = authorize_sql(
        con,
        "WITH hr_fact_employees AS ("
        "SELECT employee_id FROM main.hr_fact_employees"
        ") SELECT COUNT(*) FROM hr_fact_employees",
        scope,
    )
    assert not qualified.allowed
    assert "hr_fact_employees" in qualified.reason

    recursive = authorize_sql(
        con,
        "WITH RECURSIVE hr_fact_employees(employee_id) AS ("
        "SELECT employee_id FROM hr_fact_employees "
        "UNION ALL SELECT employee_id FROM hr_fact_employees WHERE 1=0"
        ") SELECT COUNT(*) FROM hr_fact_employees",
        scope,
    )
    assert not recursive.allowed
    assert "hr_fact_employees" in recursive.reason


def test_cte_cannot_shadow_an_unknown_physical_relation(con, policy_file):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"healthcare_analyst"}), authenticated=True)
    )
    sql = "WITH outside AS (SELECT * FROM outside) SELECT * FROM outside"

    decision = authorize_sql(con, sql, scope)

    assert not decision.allowed
    assert "outside the governed catalogue" in decision.reason


def test_legitimate_and_recursive_ctes_remain_available(con, policy_file):
    scope = load_policy(policy_file).scope_for(
        Principal("subject", roles=frozenset({"healthcare_analyst"}), authenticated=True)
    )
    ordinary = run_query(
        con,
        "WITH claims AS (SELECT claim_id FROM healthcare_fact_claims) "
        "SELECT COUNT(*) FROM claims",
        access=scope,
    )
    recursive = run_query(
        con,
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 3) "
        "SELECT SUM(x) FROM n",
        access=scope,
    )

    assert ordinary.ok and ordinary.rows == [(12000,)]
    assert recursive.ok and recursive.rows == [(6,)]
