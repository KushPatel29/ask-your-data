"""Explicit demo/production configuration boundary; no network or secret output."""

import os
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

from engine.access import PolicyConfigurationError, load_policy


def _reserved_audit_device(path: Path) -> bool:
    """Windows devices remain devices with an extension or stream suffix.

    Check their names on every platform so a deployment preflight stays
    portable; opening NUL/a device for an audit silently discards the trail.
    """
    name = PureWindowsPath(str(path)).name
    base = name.partition(":")[0].partition(".")[0].rstrip(" ").upper()
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{digit}" for prefix in ("COM", "LPT")
                    for digit in "123456789¹²³")
    return base in reserved


def validate_deployment(environ: Mapping[str, str] | None = None) -> str:
    """Validate prerequisites, not an IdP login or production certification.

    Public synthetic demos retain their keyless defaults. Operators explicitly
    selecting production cannot accidentally inherit anonymous access or an
    ephemeral in-memory-only audit. JSONL still needs a protected durable mount
    and forwarding/retention; this check does not establish either guarantee.
    """
    env = os.environ if environ is None else environ
    mode = str(env.get("ASK_DEPLOYMENT_MODE", "demo")).strip().lower()
    if mode not in {"demo", "production"}:
        raise PolicyConfigurationError("ASK_DEPLOYMENT_MODE must be demo or production")
    if mode == "demo":
        return mode
    if str(env.get("ASK_AUTH_MODE", "")).strip().lower() != "oidc":
        raise PolicyConfigurationError("Production requires ASK_AUTH_MODE=oidc")
    required = ("ASK_OIDC_ISSUER", "ASK_OIDC_AUDIENCE", "ASK_OIDC_JWKS_URL",
                "ASK_POLICY_FILE", "ASK_YOUR_DATA_AUDIT")
    missing = [key for key in required if not str(env.get(key, "")).strip()]
    if missing:
        raise PolicyConfigurationError("Production requires " + ", ".join(missing))
    for key in ("ASK_OIDC_ISSUER", "ASK_OIDC_JWKS_URL"):
        try:
            url = urlsplit(str(env[key]).strip())
            valid = (url.scheme == "https" and bool(url.hostname)
                     and not url.username and not url.password and not url.fragment)
        except ValueError:
            valid = False
        if not valid:
            raise PolicyConfigurationError(f"{key} must be an HTTPS URL without credentials")
    policy = Path(str(env["ASK_POLICY_FILE"]).strip())
    try:
        load_policy(policy)
    except Exception as exc:
        # Policy validation can contain restricted table/column names; keep
        # deployment error text free of file contents and configured values.
        raise PolicyConfigurationError("Production policy file is missing or invalid") from exc
    audit_path = Path(str(env["ASK_YOUR_DATA_AUDIT"]).strip())
    if (not audit_path.is_absolute() or not audit_path.parent.is_dir()
            or _reserved_audit_device(audit_path)
            or (audit_path.exists() and not audit_path.is_file())):
        raise PolicyConfigurationError(
            "Production audit must name an absolute file in an existing protected directory"
        )
    return mode
