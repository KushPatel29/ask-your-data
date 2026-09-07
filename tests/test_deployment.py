"""Production can never inherit the anonymous demonstration configuration."""

import os
from pathlib import Path

import pytest

from engine.access import PolicyConfigurationError
from engine.deployment import validate_deployment


def configured(tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\nroles:\n  analyst:\n    domains: [healthcare]\n", "utf-8")
    return {
        "ASK_DEPLOYMENT_MODE": "production", "ASK_AUTH_MODE": "oidc",
        "ASK_OIDC_ISSUER": "https://identity.example.test/",
        "ASK_OIDC_AUDIENCE": "analytics",
        "ASK_OIDC_JWKS_URL": "https://identity.example.test/keys",
        "ASK_POLICY_FILE": str(policy), "ASK_YOUR_DATA_AUDIT": str(tmp_path / "audit.jsonl"),
    }


def test_demo_keeps_keyless_defaults():
    assert validate_deployment({}) == "demo"


def test_production_refuses_anonymous_defaults():
    with pytest.raises(PolicyConfigurationError, match="ASK_AUTH_MODE=oidc"):
        validate_deployment({"ASK_DEPLOYMENT_MODE": "production"})


def test_complete_config_is_validated_without_writing_an_audit(tmp_path):
    env = configured(tmp_path)
    assert validate_deployment(env) == "production"
    assert not (tmp_path / "audit.jsonl").exists()


@pytest.mark.parametrize("setting", ["ASK_OIDC_ISSUER", "ASK_OIDC_AUDIENCE",
                                    "ASK_OIDC_JWKS_URL", "ASK_POLICY_FILE", "ASK_YOUR_DATA_AUDIT"])
def test_missing_production_prerequisite_fails_closed(tmp_path, setting):
    env = configured(tmp_path)
    del env[setting]
    with pytest.raises(PolicyConfigurationError, match=setting):
        validate_deployment(env)


@pytest.mark.parametrize("url", ["http://identity.test", "https://user:secret@identity.test",
                                 "not a url", "https://identity.test/#secret"])
def test_identity_urls_fail_without_echoing_secrets(tmp_path, url):
    env = configured(tmp_path)
    env["ASK_OIDC_JWKS_URL"] = url
    with pytest.raises(PolicyConfigurationError) as error:
        validate_deployment(env)
    assert url not in str(error.value)


def test_invalid_policy_and_relative_audit_path_fail_closed(tmp_path):
    env = configured(tmp_path)
    env["ASK_YOUR_DATA_AUDIT"] = "audit.jsonl"
    with pytest.raises(PolicyConfigurationError, match="absolute"):
        validate_deployment(env)
    env = configured(tmp_path)
    (tmp_path / "policy.yaml").write_text("{unparseable", "utf-8")
    with pytest.raises(PolicyConfigurationError, match="policy file"):
        validate_deployment(env)


@pytest.mark.parametrize("name", ["NUL", "nul.jsonl", "NUL:events", "NUL::$DATA",
                                 "NUL .log", "CON", "aux.txt", "PRN", "COM1.log",
                                 "lpt9:audit", "COM¹", "LPT².txt", "CONIN$", "CONOUT$"])
def test_windows_audit_devices_are_rejected_without_opening_them(tmp_path, name):
    env = configured(tmp_path)
    env["ASK_YOUR_DATA_AUDIT"] = str(tmp_path / name)
    with pytest.raises(PolicyConfigurationError, match="absolute file"):
        validate_deployment(env)


def test_existing_audit_must_be_a_regular_file(tmp_path):
    env = configured(tmp_path)
    env["ASK_YOUR_DATA_AUDIT"] = str(Path(os.devnull).resolve())
    with pytest.raises(PolicyConfigurationError, match="absolute file"):
        validate_deployment(env)


def test_existing_audit_file_is_preserved_by_preflight(tmp_path):
    env = configured(tmp_path)
    target = Path(env["ASK_YOUR_DATA_AUDIT"])
    target.write_text('{"existing":"event"}\n', encoding="utf-8")
    before = target.read_bytes()
    assert validate_deployment(env) == "production"
    assert target.read_bytes() == before
