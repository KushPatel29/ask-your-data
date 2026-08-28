"""
The answer-verification layer, proved offline.

Two obligations, and the second matters more than the first:

  1. Every check catches the failure it exists for.
  2. No check fires on the 39 golden queries. A check that flags good SQL is
     worse than no check, so the false-positive rate against the accuracy
     contract is asserted here as zero rather than reported in a comment.
"""

from pathlib import Path

import pytest
import yaml

from engine.query import run_query
from engine.verify import ERROR, NOTE, WARN, Verifier, correction_message, worst

EVALS = Path(__file__).resolve().parent.parent / "evals"
GOLDEN = yaml.safe_load((EVALS / "golden_questions.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verifier(con):
    return Verifier(con)


def checks(findings):
    return {f.check for f in findings}


# --------------------------------------------------------------------------
# The false-positive contract: the golden set must come back clean.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", GOLDEN, ids=[c["id"] for c in GOLDEN])
def test_golden_sql_is_never_flagged(con, verifier, case):
    findings = verifier.check_sql(case["sql"])
    result = run_query(con, case["sql"])
    findings += verifier.check_result(case["sql"], result)
    blocking = [f for f in findings if f.severity in (ERROR, WARN)]
    assert not blocking, f"{case['id']} false-positive: {[str(f) for f in blocking]}"


def test_golden_false_positive_rate_is_zero(con, verifier):
    """The headline number, asserted rather than documented."""
    flagged = []
    for case in GOLDEN:
        findings = verifier.check_sql(case["sql"])
        findings += verifier.check_result(case["sql"], run_query(con, case["sql"]))
        if any(f.severity in (ERROR, WARN) for f in findings):
            flagged.append(case["id"])
    assert flagged == [], f"{len(flagged)}/{len(GOLDEN)} golden queries flagged: {flagged}"


# --------------------------------------------------------------------------
# cross_domain_join — the trap that returns rows.
# --------------------------------------------------------------------------

CROSS_DOMAIN = ("SELECT COUNT(*) FROM hr_fact_employees e "
                "JOIN finance_erp_gl g ON e.employee_id = g.account_id")


def test_cross_domain_join_is_an_error(verifier):
    findings = verifier.check_sql(CROSS_DOMAIN)
    assert checks(findings) == {"cross_domain_join"}
    assert findings[0].severity == ERROR and findings[0].blocking
    assert "hr" in findings[0].message and "finance" in findings[0].message


def test_the_cross_domain_join_really_does_return_rows(con):
    """Why this check has to be static: the query does not error and is not empty,
    so nothing downstream of execution can see that it is meaningless."""
    result = run_query(con, CROSS_DOMAIN)
    assert result.ok and result.rows[0][0] > 0


def test_cross_domain_join_found_in_a_where_clause(verifier):
    findings = verifier.check_sql(
        "SELECT COUNT(*) FROM hr_fact_employees e, finance_erp_gl g "
        "WHERE e.employee_id = g.account_id")
    assert "cross_domain_join" in checks(findings)


def test_same_domain_join_is_clean(verifier):
    findings = verifier.check_sql(
        "SELECT COUNT(*) FROM healthcare_fact_claims c "
        "JOIN healthcare_dim_payer p ON c.payer_id = p.payer_id")
    assert findings == []


def test_two_domains_without_a_join_is_only_a_note(verifier):
    findings = verifier.check_sql(
        "SELECT (SELECT COUNT(*) FROM hr_fact_employees), "
        "(SELECT COUNT(*) FROM healthcare_fact_claims)")
    assert checks(findings) == {"cross_domain_reference"}
    assert worst(findings) == NOTE


# --------------------------------------------------------------------------
# join_fanout — the silent inflation.
# --------------------------------------------------------------------------

FANOUT = ("SELECT SUM(s.completed) FROM clinical_subjects s "
          "LEFT JOIN clinical_query_log q ON s.subject_id = q.subject_id")


def test_fanout_is_detected(verifier):
    findings = verifier.check_sql(FANOUT)
    assert checks(findings) == {"join_fanout"}
    # ERROR, not WARN. A fan-out returns a plausible number that is simply too
    # big, so advisory severity meant the inflated figure was narrated as fact.
    # It fires on 0 of the 39 golden queries, so blocking costs nothing.
    assert findings[0].severity == ERROR
    assert findings[0].blocking


def test_the_fanout_really_does_change_the_number(con):
    truth = con.execute("SELECT SUM(completed) FROM clinical_subjects").fetchone()[0]
    inflated = run_query(con, FANOUT).rows[0][0]
    assert inflated != truth


def test_count_distinct_is_not_a_fanout(verifier):
    findings = verifier.check_sql(
        "SELECT COUNT(DISTINCT s.subject_id) FROM clinical_subjects s "
        "LEFT JOIN clinical_query_log q ON s.subject_id = q.subject_id")
    assert "join_fanout" not in checks(findings)


def test_aggregating_the_many_side_is_not_a_fanout(verifier):
    findings = verifier.check_sql(
        "SELECT SUM(c.paid_amount) FROM healthcare_fact_claims c "
        "JOIN healthcare_dim_payer p ON c.payer_id = p.payer_id")
    assert findings == []


def test_composite_join_key_is_tested_as_a_tuple(verifier):
    """Five columns, none unique alone, unique together. Testing them one at a
    time would report a fan-out that does not exist — this is the golden set's
    injected_defect_detection_rate query."""
    case = next(c for c in GOLDEN if c["id"] == "injected_defect_detection_rate")
    assert verifier.check_sql(case["sql"]) == []


# --------------------------------------------------------------------------
# Result sanity.
# --------------------------------------------------------------------------

def test_empty_join_is_diagnosed_by_key_overlap(con, verifier):
    """Two tables in the SAME domain that still share no identifiers, which is
    why the cross-domain check alone is not fine-grained enough."""
    sql = ("SELECT g.geo, s.session_id FROM marketing_experiment_geo_weekly g "
           "JOIN marketing_fact_sessions s ON g.geo = s.region")
    findings = verifier.check_result(sql, run_query(con, sql))
    empty = [f for f in findings if f.check == "empty_result"]
    assert empty and empty[0].severity == WARN
    assert "no values in common" in empty[0].message


def test_plain_empty_result_is_a_note(con, verifier):
    sql = "SELECT status FROM healthcare_fact_claims WHERE status = 'Nonexistent'"
    findings = verifier.check_result(sql, run_query(con, sql))
    assert checks(findings) == {"empty_result"}
    assert worst(findings) == NOTE


def test_null_scalar_is_caught(con, verifier):
    sql = "SELECT SUM(paid_amount) FROM healthcare_fact_claims WHERE status = 'Nope'"
    findings = verifier.check_result(sql, run_query(con, sql))
    assert "null_scalar" in checks(findings)


def test_share_over_one_hundred_is_noted(con, verifier):
    sql = ("SELECT ROUND(100.0 * COUNT(*) / "
           "NULLIF(COUNT(*) FILTER (WHERE status = 'Denied'), 0), 1) AS denial_rate "
           "FROM healthcare_fact_claims")
    findings = verifier.check_result(sql, run_query(con, sql))
    assert "share_out_of_range" in checks(findings)


def test_a_negative_percentage_change_is_not_flagged(con, verifier):
    """The measured false positive this check was narrowed to avoid: fiscal 2025
    revenue is legitimately -4.52 percent against fiscal 2024."""
    case = next(c for c in GOLDEN if c["id"] == "wholesale_fy2025_revenue_change")
    findings = verifier.check_result(case["sql"], run_query(con, case["sql"]))
    assert "share_out_of_range" not in checks(findings)


# --------------------------------------------------------------------------
# Error enrichment.
# --------------------------------------------------------------------------

def test_missing_column_is_located_on_the_right_table(con, verifier):
    sql = "SELECT payer_type FROM healthcare_fact_claims"
    result = run_query(con, sql)
    assert not result.ok
    enriched = verifier.explain_error(sql, result.error)
    assert "healthcare_dim_payer" in enriched
    assert result.error in enriched          # never loses what DuckDB said
    assert len(enriched) > len(result.error)


def test_qualified_missing_column_is_also_located(con, verifier):
    sql = "SELECT c.payer_type FROM healthcare_fact_claims c"
    enriched = verifier.explain_error(sql, run_query(con, sql).error)
    assert "healthcare_dim_payer" in enriched


def test_unrelated_errors_pass_through_untouched(verifier):
    error = "Catalog Error: Table with name nope does not exist!"
    assert verifier.explain_error("SELECT * FROM nope", error) == error


def test_a_column_that_exists_nowhere_else_is_not_embellished(con, verifier):
    sql = "SELECT zzz_not_a_column FROM hr_fact_employees"
    error = run_query(con, sql).error
    assert verifier.explain_error(sql, error) == error


# --------------------------------------------------------------------------
# Ambiguity: disclosed, never asked.
# --------------------------------------------------------------------------

def test_ambiguous_entity_is_disclosed_with_the_chosen_domain(verifier):
    note = verifier.ambiguity_note(
        "Who is the top customer?",
        "SELECT customer_name FROM retail_customer_analytics ORDER BY total_revenue DESC LIMIT 1")
    assert note is not None and note.severity == NOTE
    assert "retail" in note.message and "customer" in note.message


def test_unambiguous_question_gets_no_note(verifier):
    assert verifier.ambiguity_note(
        "How many active employees are there?",
        "SELECT COUNT(*) FROM hr_fact_employees WHERE is_active = 1") is None


def test_ambiguity_note_never_blocks(verifier):
    note = verifier.ambiguity_note(
        "Who is the top customer?",
        "SELECT customer_name FROM retail_customer_analytics LIMIT 1")
    assert not note.blocking


# --------------------------------------------------------------------------
# Plumbing.
# --------------------------------------------------------------------------

def test_verification_never_raises_on_junk(verifier):
    for sql in ["", "   ", "SELECT", "this is not sql at all", "SELECT * FROM"]:
        assert verifier.check_sql(sql) == []
        assert verifier.explain_error(sql, "some error") == "some error"


def test_correction_message_names_the_problem_and_the_fix(verifier):
    findings = verifier.check_sql(CROSS_DOMAIN)
    message = correction_message(findings)
    assert "hr_fact_employees" in message and "cannot_answer" in message


def test_worst_ranks_severities(verifier):
    assert worst(verifier.check_sql(CROSS_DOMAIN)) == ERROR
    assert worst([]) is None
