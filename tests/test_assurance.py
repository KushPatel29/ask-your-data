"""Release assurance is governed evidence, not hand-maintained UI copy."""

from pathlib import Path

import pytest
import yaml

from engine import assurance

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = assurance.load_manifest()
ATTACKS = yaml.safe_load(
    (ROOT / "evals" / "adversarial_questions.yaml").read_text(encoding="utf-8")
)


def test_release_pack_is_versioned_and_owned():
    assert MANIFEST["schema_version"] == 1
    assert MANIFEST["release_id"] == "2026.09.15"
    assert MANIFEST["owner"]
    assert MANIFEST["review_status"] == "portfolio evidence"


def test_every_declared_suite_count_matches_its_evidence():
    # load_manifest performs the actual cross-file validation. This assertion
    # keeps the reason for loading it at collection time visible in the suite.
    assert sum(row["cases"] for row in MANIFEST["suites"]) == 177


def test_every_governed_evidence_path_is_repository_relative():
    for suite in MANIFEST["suites"]:
        path = Path(suite["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert (ROOT / path).is_file()


def test_benchmark_fingerprint_is_short_stable_and_content_derived():
    first = assurance.benchmark_fingerprint(MANIFEST)
    second = assurance.benchmark_fingerprint(assurance.load_manifest())
    assert first == second
    assert len(first) == 12
    assert all(char in "0123456789abcdef" for char in first)


def test_the_release_pack_budget_matches_the_running_enforcement():
    rows = assurance.budget_rows(MANIFEST)
    assert {row["Status"] for row in rows} == {"MATCH"}
    assert {row["Boundary"] for row in rows} == {
        "Result rows",
        "Statement execution",
        "End-to-end request",
        "Question length",
        "Retrieved schema",
    }


def test_a_budget_change_cannot_hide_as_a_green_release():
    changed = {**MANIFEST, "budgets": [dict(row) for row in MANIFEST["budgets"]]}
    changed["budgets"][0]["value"] += 1
    assert assurance.budget_rows(changed)[0]["Status"] == "DRIFT"


def test_offline_and_live_model_evidence_are_not_blurred():
    modes = {row["id"]: row["execution"] for row in MANIFEST["suites"]}
    assert modes["reference_sql"] == "offline-ci"
    assert modes["adversarial_model"] == "live-model-required"
    model_gate = next(
        row for row in MANIFEST["release_gates"] if row["id"] == "live_model_red_team"
    )
    assert model_gate["status"] == "RUN REQUIRED"


def test_release_decision_is_scoped_to_the_synthetic_demonstration():
    snapshot = assurance.release_snapshot(MANIFEST)
    assert snapshot["decision"] == "GO for synthetic portfolio demonstration"
    boundary = snapshot["production_boundary"].lower()
    assert "does not prove" in boundary
    assert "production" in boundary


@pytest.mark.parametrize("case", ATTACKS, ids=[case["id"] for case in ATTACKS])
def test_every_red_team_case_names_its_risk_control_and_expected_behavior(case):
    assert case["question"].strip()
    assert case["risk"].strip()
    assert case["control"].strip()
    assert case["expected_behavior"].strip()


def test_red_team_set_covers_the_major_text_to_sql_threat_families():
    risks = {case["risk"] for case in ATTACKS}
    assert {
        "destructive_sql",
        "data_exfiltration",
        "direct_prompt_injection",
        "indirect_prompt_injection",
        "authorization_bypass",
        "semantic_cross_domain_join",
        "denial_of_service",
        "prompt_and_secret_disclosure",
        "system_catalog_discovery",
    } <= risks


def test_the_live_model_suite_count_cannot_drift_from_the_red_team_file():
    declared = next(row["cases"] for row in MANIFEST["suites"] if row["id"] == "adversarial_model")
    assert declared == len(ATTACKS) == 15


def test_example_policy_matrix_exposes_every_role_and_sensitive_exception():
    rows = assurance.role_rows()
    assert {row["Role"] for row in rows} == {
        "enterprise_analyst",
        "people_analyst",
        "hr_compensation",
        "platform_admin",
    }
    admin = next(row for row in rows if row["Role"] == "platform_admin")
    assert admin["Domain grants"] == "all"
    assert "hr_fact_employees.base_salary" in admin["Sensitive-column access"]


def test_gate_table_keeps_status_and_evidence_together():
    rows = assurance.gate_rows(MANIFEST)
    assert len(rows) == 5
    assert sum(row["Status"] == "PASS" for row in rows) == 4
    assert all(row["Control"] and row["Evidence"] for row in rows)


def test_suite_table_names_how_each_claim_is_reproduced():
    rows = assurance.suite_rows(MANIFEST)
    assert len(rows) == 6
    assert {row["Run mode"] for row in rows} == {"CI — no model key", "Live model rerun"}
    assert all(row["Release gate"] and row["Evidence"] for row in rows)
