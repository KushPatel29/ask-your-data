"""The SQL guard is the safety boundary, so it is tested harder than anything
else: every mutation verb is rejected, and legitimate analytical SQL is allowed."""

import pytest

from engine.sql_guard import FORBIDDEN, MAX_SQL_CHARS, validate_sql

ALLOWED = [
    "SELECT 1",
    "select count(*) from healthcare_fact_claims",
    "SELECT payer_type, SUM(paid_amount) FROM healthcare_fact_claims GROUP BY payer_type",
    "WITH x AS (SELECT * FROM hr_fact_employees) SELECT COUNT(*) FROM x",
    "SELECT * FROM retail_fact_sales ORDER BY revenue DESC LIMIT 10;",  # trailing ; ok
    "SELECT * FROM t WHERE note = 'please DROP TABLE t'",  # keyword only inside a string
    "SELECT employee_name FROM hr_fact_employees -- drop everything\n WHERE is_active = 1",
    "SELECT $$DROP TABLE t; DELETE FROM x$$ AS s",       # dollar-quoted string literal
    "SELECT $tag$INSERT INTO y$tag$ AS s FROM t",         # tagged dollar quote
]

BLOCKED = [
    "DROP TABLE healthcare_fact_claims",
    "DELETE FROM hr_fact_employees",
    "UPDATE hr_fact_employees SET base_salary = 0",
    "INSERT INTO finance_erp_gl VALUES (1)",
    "CREATE TABLE evil AS SELECT 1",
    "ALTER TABLE t ADD COLUMN c INT",
    "SELECT 1; DROP TABLE t",            # second statement
    "SELECT 1; SELECT 2",                # two statements
    "COPY hr_fact_employees TO 'out.csv'",
    "INSTALL httpfs",
    "ATTACH 'other.db' AS o",
    "PRAGMA database_list",
    "SET memory_limit='1GB'",
    "",
    "   ",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql):
    ok, reason = validate_sql(sql)
    assert ok, f"should allow but blocked ({reason}): {sql}"


@pytest.mark.parametrize("sql", BLOCKED)
def test_blocked(sql):
    ok, _ = validate_sql(sql)
    assert not ok, f"should block but allowed: {sql}"


def test_every_forbidden_keyword_is_caught():
    for kw in FORBIDDEN:
        ok, _ = validate_sql(f"{kw} something")
        assert not ok, f"{kw} slipped through"


def test_offset_is_not_mistaken_for_set():
    ok, _ = validate_sql("SELECT * FROM t ORDER BY x LIMIT 5 OFFSET 10")
    assert ok, "OFFSET must not trip the SET rule"


def test_dollar_quote_cannot_hide_a_second_statement():
    # the ; sits OUTSIDE the dollar-quoted string — must still be caught
    ok, _ = validate_sql("SELECT $$harmless$$ AS s; DROP TABLE t")
    assert not ok, "statement after a dollar-quoted literal slipped through"


def test_parser_input_is_bounded_before_scanning():
    ok, reason = validate_sql("SELECT '" + ("x" * MAX_SQL_CHARS) + "'")
    assert not ok and "safety limit" in reason
