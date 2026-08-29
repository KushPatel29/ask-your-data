"""
The cartesian checks: tables combined row-wise with nothing connecting them.

`cross_domain_join` reads join *conditions*, so it only sees a join that has
one. These tests cover the shape it cannot see — no condition at all, or a
condition that is not an equality — which returns rows, does not error, and is
not empty, so nothing else in the pipeline can notice it.

The false-positive obligation is the harder half and is asserted twice: against
the 39 golden queries, and against every fact-to-dimension join that exists in
this warehouse, generated from the catalog in both the explicit `JOIN ... ON`
and the implicit `FROM a, b WHERE` form.
"""

import itertools
from pathlib import Path

import pytest
import yaml

from engine.query import run_query
from engine.verify import DOMAIN_OF, ERROR, WARN, Verifier

EVALS = Path(__file__).resolve().parent.parent / "evals"
GOLDEN = yaml.safe_load((EVALS / "golden_questions.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verifier(con):
    return Verifier(con)


def checks(findings):
    return {f.check for f in findings}


def hard(findings):
    return [f for f in findings if f.severity in (ERROR, WARN)]


# --------------------------------------------------------------------------
# What must be caught.
# --------------------------------------------------------------------------

CROSS_DOMAIN_CARTESIANS = {
    # All three return rows against the real warehouse: 19,000, 19,000, 24,306.
    "cross join": "SELECT COUNT(*) FROM hr_fact_employees CROSS JOIN finance_dim_account",
    "comma join": "SELECT COUNT(*) FROM hr_fact_employees, finance_dim_account",
    "non-equi join": ("SELECT COUNT(*) FROM hr_fact_employees e JOIN finance_erp_gl g "
                      "ON e.employee_id < g.account_id"),
}


@pytest.mark.parametrize("sql", CROSS_DOMAIN_CARTESIANS.values(),
                         ids=list(CROSS_DOMAIN_CARTESIANS))
def test_cross_domain_cartesian_is_blocking(con, verifier, sql):
    """It runs, it returns rows, and it is meaningless — so it must block."""
    result = run_query(con, sql)
    assert result.ok and result.rows and result.rows[0][0] > 0

    findings = verifier.check_sql(sql)
    blocking = [f for f in findings if f.blocking]
    assert [f.check for f in blocking] == ["cross_domain_cartesian"]
    assert "hr" in blocking[0].message and "finance" in blocking[0].message


def test_cross_domain_cartesian_suppresses_the_contradicting_note(verifier):
    """Once the tables are proved combined, "not joined" must not also appear."""
    findings = verifier.check_sql(CROSS_DOMAIN_CARTESIANS["comma join"])
    assert "cross_domain_reference" not in checks(findings)


def test_same_domain_cartesian_warns_but_does_not_block(con, verifier):
    sql = "SELECT COUNT(*) FROM hr_fact_employees, hr_dim_department"
    assert run_query(con, sql).rows[0][0] == 22800   # 1,900 x 12, silently

    findings = verifier.check_sql(sql)
    assert checks(findings) == {"cartesian_join"}
    assert findings[0].severity == WARN and not findings[0].blocking


def test_a_stranded_third_table_is_caught(con, verifier):
    """Two tables joined correctly plus one crossed in is still a cartesian."""
    sql = ("SELECT COUNT(*) FROM healthcare_fact_claims c "
           "JOIN healthcare_dim_payer p ON c.payer_id = p.payer_id "
           "CROSS JOIN healthcare_dim_provider pr")
    assert run_query(con, sql).rows[0][0] == 240000

    findings = verifier.check_sql(sql)
    assert checks(findings) == {"cartesian_join"}
    assert "healthcare_dim_provider" in findings[0].message


# --------------------------------------------------------------------------
# What must NOT be caught. A check that flags good SQL is worse than no check.
# --------------------------------------------------------------------------

LEGITIMATE = {
    # DuckDB parses this as a CROSS join and files the equality under
    # `where_clause`, so reading only the join condition would call it cartesian.
    "implicit join, predicate in WHERE":
        ("SELECT COUNT(*) FROM hr_fact_employees e, hr_dim_department d "
         "WHERE e.department_id = d.department_id"),
    # Two scalars printed next to each other. Each subquery is its own
    # SELECT_NODE, so neither FROM tree holds more than one table.
    "two scalar subqueries side by side":
        ("SELECT a.n, b.m FROM (SELECT COUNT(*) n FROM hr_fact_employees) a, "
         "(SELECT COUNT(*) m FROM finance_dim_account) b"),
    "union of two domains":
        ("SELECT 'hr' AS d, COUNT(*) FROM hr_fact_employees UNION ALL "
         "SELECT 'finance', COUNT(*) FROM finance_dim_account"),
    "self join":
        ("SELECT COUNT(*) FROM hr_fact_employees a JOIN hr_fact_employees b "
         "ON a.department_id = b.department_id"),
    "correlated subquery":
        ("SELECT d.department_id, (SELECT COUNT(*) FROM hr_fact_employees e "
         "WHERE e.department_id = d.department_id) FROM hr_dim_department d"),
    "three-table star schema":
        ("SELECT p.payer_type, COUNT(*) FROM healthcare_fact_claims c "
         "JOIN healthcare_dim_payer p ON c.payer_id = p.payer_id "
         "JOIN healthcare_dim_provider pr ON c.provider_id = pr.provider_id GROUP BY 1"),
}


@pytest.mark.parametrize("sql", LEGITIMATE.values(), ids=list(LEGITIMATE))
def test_legitimate_shapes_are_not_flagged(verifier, sql):
    findings = verifier.check_sql(sql)
    assert not hard(findings), [str(f) for f in findings]


@pytest.mark.parametrize("case", GOLDEN, ids=[c["id"] for c in GOLDEN])
def test_golden_sql_is_never_flagged_as_cartesian(verifier, case):
    findings = verifier.check_sql(case["sql"])
    assert "cartesian_join" not in checks(findings)
    assert "cross_domain_cartesian" not in checks(findings)


def _generated_joins(con):
    """Every fact-to-dimension join this warehouse supports, both syntaxes.

    A 39-question golden set only exercises four multi-table FROM trees, which
    is a thin denominator for a false-positive claim. This widens it from the
    catalog itself: any two same-domain tables sharing an `_id` column, joined
    on it, is known-good SQL by construction.
    """
    columns = {t: [r[0] for r in con.execute(f'DESCRIBE "{t}"').fetchall()]
               for t in DOMAIN_OF}
    by_domain = {}
    for table, domain in DOMAIN_OF.items():
        by_domain.setdefault(domain, []).append(table)
    for _domain, tables in by_domain.items():
        facts = [t for t in tables if "dim_" not in t]
        dims = [t for t in tables if "dim_" in t]
        for fact, dim in itertools.product(facts, dims):
            shared = [c for c in columns[fact]
                      if c in columns[dim] and c.endswith("_id")]
            if not shared:
                continue
            key = shared[0]
            yield (f"{fact}+{dim} explicit",
                   f'SELECT COUNT(*) FROM "{fact}" f JOIN "{dim}" d ON f."{key}" = d."{key}"')
            yield (f"{fact}+{dim} implicit",
                   f'SELECT COUNT(*) FROM "{fact}" f, "{dim}" d WHERE f."{key}" = d."{key}"')


def test_no_generated_fact_dimension_join_is_flagged(con, verifier):
    cases = list(_generated_joins(con))
    assert len(cases) >= 40, "the generator stopped finding joins; the catalog changed"
    flagged = [(label, [str(f) for f in hard(verifier.check_sql(sql))])
               for label, sql in cases]
    flagged = [f for f in flagged if f[1]]
    assert flagged == []


# --------------------------------------------------------------------------
# A NULL metric is not only a 1x1 result.
# --------------------------------------------------------------------------

def test_null_metric_is_caught_beside_a_non_null_column(con, verifier):
    """The shape that reads as an answer: a real count next to a blank average."""
    sql = ("SELECT COUNT(*) AS n, AVG(paid_amount) FILTER (WHERE status = 'Nope') "
           "AS avg_paid FROM healthcare_fact_claims")
    result = run_query(con, sql)
    assert result.rows == [(12000, None)]

    findings = verifier.check_result(sql, result)
    assert checks(findings) == {"null_scalar"}
    # Only the blank column is named — the count next to it is fine.
    assert "avg_paid came back NULL" in findings[0].message


def test_a_null_dimension_in_a_many_row_report_is_ordinary(con, verifier):
    """Missing data in a listing is not a broken calculation; do not warn."""
    sql = ("SELECT claim_id, denial_reason FROM healthcare_fact_claims "
           "WHERE status = 'Paid' LIMIT 5")
    result = run_query(con, sql)
    assert any(row[1] is None for row in result.rows)
    assert "null_scalar" not in checks(verifier.check_result(sql, result))


# --------------------------------------------------------------------------
# A difference in the denominator is still a share.
# --------------------------------------------------------------------------

def test_share_with_a_subtracted_denominator_is_still_range_checked(con, verifier):
    sql = ("SELECT 100.0 * SUM(submitted_amount) / "
           "NULLIF(SUM(submitted_amount) - SUM(allowed_amount), 0) AS collection_rate "
           "FROM healthcare_fact_claims")
    result = run_query(con, sql)
    assert result.rows[0][0] > 100

    findings = verifier.check_result(sql, result, "what is the collection rate?")
    assert "share_out_of_range" in checks(findings)


def test_a_percentage_change_is_still_exempt(con, verifier):
    """`(new - old) / old` is unbounded by construction and must stay quiet."""
    sql = ("SELECT 100.0 * (SUM(submitted_amount) - SUM(allowed_amount)) "
           "/ NULLIF(SUM(allowed_amount), 0) AS pct_change FROM healthcare_fact_claims")
    result = run_query(con, sql)
    assert result.rows[0][0] > 100
    assert "share_out_of_range" not in checks(verifier.check_result(sql, result, "change?"))
