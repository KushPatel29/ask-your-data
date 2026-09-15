"""Versioned release-assurance evidence for the public decision-support app.

This module deliberately does not call a model. It makes the evidence needed
to decide whether a release is reviewable available without credentials:

* which benchmark files and release thresholds were approved together;
* a content fingerprint that changes when any governed evidence changes;
* the query-cost envelope currently enforced by the running code;
* the threat families in the live-model red-team set; and
* the grants and sensitive-column exceptions in the example role policy.

The manifest records CI observations, not production outcomes. The UI retains
that boundary instead of turning a green portfolio build into a claim about a
real organization's adoption, accuracy, capacity, or control effectiveness.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "evals" / "assurance_release.yaml"


class AssuranceConfigurationError(ValueError):
    """The release pack is incomplete, inconsistent, or unsafe to resolve."""


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssuranceConfigurationError(
            f"cannot read assurance evidence {path.name}: {exc}"
        ) from exc


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AssuranceConfigurationError(
            f"assurance evidence path escapes the repository: {relative!r}"
        ) from exc
    if not candidate.is_file():
        raise AssuranceConfigurationError(f"assurance evidence is missing: {relative}")
    return candidate


def validate_manifest(document: Any, *, root: Path = ROOT) -> dict:
    if not isinstance(document, dict):
        raise AssuranceConfigurationError("assurance manifest must be an object")
    if document.get("schema_version") != 1:
        raise AssuranceConfigurationError("assurance manifest schema_version must be 1")
    for field in ("release_id", "owner", "review_status", "decision", "production_boundary"):
        if not str(document.get(field, "")).strip():
            raise AssuranceConfigurationError(f"assurance manifest is missing {field}")

    suites = document.get("suites")
    budgets = document.get("budgets")
    gates = document.get("release_gates")
    if not isinstance(suites, list) or not suites:
        raise AssuranceConfigurationError("assurance manifest must contain suites")
    if not isinstance(budgets, list) or not budgets:
        raise AssuranceConfigurationError("assurance manifest must contain budgets")
    if not isinstance(gates, list) or not gates:
        raise AssuranceConfigurationError("assurance manifest must contain release_gates")

    collections = (("suites", suites), ("budgets", budgets), ("release_gates", gates))
    for collection_name, rows in collections:
        ids = [str(row.get("id", "")).strip() for row in rows if isinstance(row, dict)]
        if len(ids) != len(rows) or not all(ids) or len(set(ids)) != len(ids):
            raise AssuranceConfigurationError(f"{collection_name} must have unique, non-empty ids")

    allowed_execution = {"offline-ci", "live-model-required"}
    for suite in suites:
        for field in ("label", "path", "execution", "gate"):
            if not str(suite.get(field, "")).strip():
                raise AssuranceConfigurationError(f"suite {suite['id']} is missing {field}")
        if suite["execution"] not in allowed_execution:
            raise AssuranceConfigurationError(
                f"suite {suite['id']} has unsupported execution mode {suite['execution']!r}"
            )
        if not isinstance(suite.get("cases"), int) or suite["cases"] <= 0:
            raise AssuranceConfigurationError(
                f"suite {suite['id']} must declare a positive case count"
            )
        evidence = _inside(root, suite["path"])
        loaded = _load_yaml(evidence)
        if suite["id"] != "role_policy":
            if not isinstance(loaded, list) or len(loaded) != suite["cases"]:
                actual = len(loaded) if isinstance(loaded, list) else "not a list"
                raise AssuranceConfigurationError(
                    f"suite {suite['id']} declares {suite['cases']} cases but contains {actual}"
                )
        else:
            roles = loaded.get("roles", {}) if isinstance(loaded, dict) else {}
            if len(roles) != suite["cases"]:
                raise AssuranceConfigurationError(
                    f"role_policy declares {suite['cases']} roles but contains {len(roles)}"
                )

    valid_gate_status = {"PASS", "FAIL", "RUN REQUIRED"}
    for gate in gates:
        if gate.get("status") not in valid_gate_status:
            raise AssuranceConfigurationError(
                f"release gate {gate['id']} has unsupported status {gate.get('status')!r}"
            )
        if not str(gate.get("control", "")).strip() or not str(gate.get("evidence", "")).strip():
            raise AssuranceConfigurationError(f"release gate {gate['id']} is incomplete")

    return document


def load_manifest(path: Path = DEFAULT_MANIFEST, *, root: Path = ROOT) -> dict:
    return validate_manifest(_load_yaml(path), root=root)


def benchmark_fingerprint(document: dict | None = None, *, root: Path = ROOT) -> str:
    """Fingerprint the manifest and every evidence file it governs."""
    manifest = document or load_manifest(root=root)
    digest = hashlib.sha256()
    manifest_path = _inside(root, str(DEFAULT_MANIFEST.relative_to(ROOT)))
    governed = {str(manifest_path.relative_to(root.resolve())).replace("\\", "/")}
    governed.update(str(suite["path"]) for suite in manifest["suites"])
    for relative in sorted(governed):
        path = _inside(root, relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def suite_rows(document: dict | None = None) -> list[dict[str, Any]]:
    manifest = document or load_manifest()
    return [
        {
            "Suite": suite["label"],
            "Cases": suite["cases"],
            "Run mode": (
                "CI — no model key" if suite["execution"] == "offline-ci" else "Live model rerun"
            ),
            "Release gate": suite["gate"],
            "Evidence": suite["path"],
        }
        for suite in manifest["suites"]
    ]


def gate_rows(document: dict | None = None) -> list[dict[str, str]]:
    manifest = document or load_manifest()
    return [
        {
            "Control": gate["control"],
            "Status": gate["status"],
            "Evidence": gate["evidence"],
        }
        for gate in manifest["release_gates"]
    ]


def runtime_budgets() -> dict[str, float | int]:
    """Read limits from the modules that enforce them, never from the UI."""
    from engine.deadline import DEFAULT_REQUEST_TIMEOUT_S
    from engine.limits import MAX_QUESTION_CHARS
    from engine.query import MAX_ROWS, STATEMENT_TIMEOUT_S
    from engine.retrieval import DEFAULT_K

    return {
        "result_rows": MAX_ROWS,
        "statement_time": STATEMENT_TIMEOUT_S,
        "request_time": DEFAULT_REQUEST_TIMEOUT_S,
        "question_size": MAX_QUESTION_CHARS,
        "retrieval_depth": DEFAULT_K,
    }


def budget_rows(document: dict | None = None) -> list[dict[str, Any]]:
    manifest = document or load_manifest()
    running = runtime_budgets()
    return [
        {
            "Boundary": budget["label"],
            "Approved": budget["value"],
            "Running": running.get(budget["id"]),
            "Unit": budget["unit"],
            "Status": "MATCH" if running.get(budget["id"]) == budget["value"] else "DRIFT",
        }
        for budget in manifest["budgets"]
    ]


def threat_rows(path: Path | None = None) -> list[dict[str, str]]:
    cases = _load_yaml(path or (ROOT / "evals" / "adversarial_questions.yaml"))
    if not isinstance(cases, list):
        raise AssuranceConfigurationError("adversarial evidence must be a list")
    return [
        {
            "Threat": str(case.get("risk", "")).replace("_", " ").title(),
            "Prompt": str(case.get("question", "")),
            "Control": str(case.get("control", "")),
            "Expected behavior": str(case.get("expected_behavior", "")),
        }
        for case in cases
    ]


def role_rows(path: Path | None = None) -> list[dict[str, Any]]:
    policy = _load_yaml(path or (ROOT / "enterprise-policy.example.yaml"))
    roles = policy.get("roles", {}) if isinstance(policy, dict) else {}
    sensitive = policy.get("sensitive_columns", {}) if isinstance(policy, dict) else {}
    rows = []
    for role, grants in roles.items():
        domains = grants.get("domains", []) if isinstance(grants, dict) else []
        tables = grants.get("tables", []) if isinstance(grants, dict) else []
        domains_text = "all" if domains == "*" else ", ".join(domains) or "none"
        tables_text = "all" if tables == "*" else ", ".join(tables) or "none"
        exceptions = []
        for table, columns in sensitive.items():
            for column, rule in columns.items():
                if role in rule.get("allow_roles", []):
                    exceptions.append(f"{table}.{column}")
        rows.append(
            {
                "Role": role,
                "Domain grants": domains_text,
                "Table grants": tables_text,
                "Sensitive-column access": ", ".join(exceptions) or "none",
            }
        )
    return rows


def release_snapshot(document: dict | None = None) -> dict[str, Any]:
    manifest = document or load_manifest()
    threats = threat_rows()
    return {
        "release_id": manifest["release_id"],
        "fingerprint": benchmark_fingerprint(manifest),
        "offline_suites": sum(s["execution"] == "offline-ci" for s in manifest["suites"]),
        "threat_cases": len(threats),
        "threat_families": len({row["Threat"] for row in threats}),
        "passing_gates": sum(g["status"] == "PASS" for g in manifest["release_gates"]),
        "open_gates": sum(g["status"] != "PASS" for g in manifest["release_gates"]),
        "decision": manifest["decision"],
        "production_boundary": manifest["production_boundary"],
    }
