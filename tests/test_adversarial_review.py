"""
Adversarial review of the safety and correctness boundaries.

Every test here was written by trying to break the app and recording what
actually happened, measured against DuckDB 1.5.5 and the real 71-table
warehouse. Every case is now a passing regression lock on behaviour the design
gets right (cursor isolation, read-only enforcement, authorization, and the
verifier's zero false positives on the golden set). Several began as strict
expected failures during the adversarial audit; retaining their attack payloads
here makes this file both the proof-of-concept archive and the release gate.
"""

import threading
from collections import Counter
from pathlib import Path

import pytest
import yaml

from engine.query import run_query
from engine.sql_guard import validate_sql
from engine.verify import Verifier, _relations, parse_sql
from engine.warehouse import schema_catalog, table_columns

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden_questions.yaml"


def _golden_cases():
    return yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. THE GUARD
# --------------------------------------------------------------------------

# `_strip_literals` removes line comments BEFORE it removes string literals, so
# a `--` INSIDE a literal eats the rest of the line - including a `;` and any
# forbidden keyword after it. DuckDB, which lexes properly, sees two statements
# and runs both.
STACKED = [
    ("create", "SELECT '--' ; CREATE TABLE guard_poc AS SELECT 1"),
    ("drop", "SELECT 'a--b' ; DROP TABLE IF EXISTS guard_poc"),
    ("quoted_ident", 'SELECT 1 AS "--" ; CREATE TABLE guard_poc2 AS SELECT 1'),
]


@pytest.mark.parametrize("label,sql", STACKED, ids=[c[0] for c in STACKED])
def test_guard_blocks_statements_stacked_behind_a_comment_in_a_literal(label, sql):
    ok, _reason = validate_sql(sql)
    assert not ok, f"guard admitted a stacked statement: {sql}"


def test_stacked_ddl_cannot_reach_the_database(con):
    con.execute("DROP TABLE IF EXISTS guard_poc_e2e")
    run_query(con, "SELECT '--' ; CREATE TABLE guard_poc_e2e AS SELECT 42 AS x")
    created = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'guard_poc_e2e'").fetchone()[0]
    con.execute("DROP TABLE IF EXISTS guard_poc_e2e")
    assert created == 0, "a CREATE TABLE hidden behind a string literal executed"


def test_guard_prevents_writing_files_to_disk(con, tmp_path):
    target = (tmp_path / "exfil.csv").as_posix()
    run_query(con, f"SELECT '--' ; COPY (SELECT 1 AS a) TO '{target}'")
    assert not (tmp_path / "exfil.csv").exists(), "model-authored SQL wrote a file to disk"


# The other half of the guard's blind spot needs no comment trick at all: these
# are single, genuine SELECT statements. Nothing in FORBIDDEN covers DuckDB's
# filesystem and network table functions, and enable_external_access defaults
# to true on the warehouse connection.
FILESYSTEM_REACH = [
    ("read_text", "SELECT content FROM read_text('/etc/passwd')"),
    ("read_csv_auto", "SELECT * FROM read_csv_auto('/app/.env')"),
    ("read_parquet", "SELECT * FROM read_parquet('/tmp/x.parquet')"),
    ("read_blob", "SELECT * FROM read_blob('/app/key.pem')"),
    ("glob", "SELECT * FROM glob('/app/*')"),
    ("remote_http", "SELECT * FROM read_csv_auto('https://attacker.example/x.csv')"),
]


@pytest.mark.parametrize("label,sql", FILESYSTEM_REACH, ids=[c[0] for c in FILESYSTEM_REACH])
def test_guard_blocks_filesystem_and_network_reach(label, sql):
    ok, _reason = validate_sql(sql)
    assert not ok, f"guard admitted filesystem/network reach: {sql}"


def test_guard_still_blocks_the_naive_attacks():
    """What the guard does get right, locked in so a fix does not regress it."""
    for sql in ("SELECT 1; DROP TABLE t",
                "COPY (SELECT 1) TO 'out.csv'",
                "ATTACH 'x.db' AS x",
                "PRAGMA database_list",
                "INSTALL httpfs; LOAD httpfs",
                "WITH x AS (INSERT INTO t VALUES (1)) SELECT * FROM x"):
        ok, _reason = validate_sql(sql)
        assert not ok, f"expected a block for {sql!r}"
    # ...and a keyword inside a literal is correctly NOT a block
    ok, _reason = validate_sql("SELECT * FROM t WHERE note = 'please DROP TABLE'")
    assert ok


@pytest.mark.parametrize("sql", [
    "SELECT replace(employee_name, '-', ' ') AS n FROM hr_fact_employees",
    "SELECT * REPLACE (base_salary * 2 AS base_salary) FROM hr_fact_employees",
])
def test_guard_does_not_block_legitimate_read_only_sql(sql):
    ok, reason = validate_sql(sql)
    assert ok, f"legitimate read-only SQL was blocked: {reason}"


# --------------------------------------------------------------------------
# 2. THE VERIFIER
# --------------------------------------------------------------------------

def test_verifier_has_no_false_positives_on_the_golden_set(con):
    """Independent reproduction of the 0-false-positive claim."""
    verifier = Verifier(con)
    findings = {}
    for case in _golden_cases():
        pre = verifier.check_sql(case["sql"])
        post = verifier.check_result(case["sql"], run_query(con, case["sql"]), case["question"])
        if pre or post:
            findings[case["id"]] = [str(f) for f in pre + post]
    assert findings == {}, f"false positives: {findings}"


def test_the_golden_set_barely_exercises_the_structural_checks(con):
    """The 0-false-positive claim is true, but measured on a near-empty sample.

    `check_sql` returns [] as soon as fewer than two warehouse tables appear
    (verify.py:462), so most golden queries never reach a single check. Pinned
    so the claim cannot be quoted without this caveat.
    """
    cases = _golden_cases()
    single = sum(1 for case in cases
                 if len(_relations(parse_sql(con, case["sql"]))[1]) < 2)
    assert single >= 30, "sample shape changed - re-check the false-positive claim"
    assert single / len(cases) > 0.8


# Genuinely wrong SQL that every check passes. Each triple is
# (label, wrong, right); the assistant answers the wrong one with the same
# confidence as the right one.
SILENT_WRONG = [
    ("What is the claim denial rate?",
     "SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status='Denied') / COUNT(*), 1) AS denial_rate "
     "FROM healthcare_fact_claims",
     "SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status='Denied') / "
     "COUNT(*) FILTER (WHERE status IN ('Paid','Denied')), 1) AS denial_rate "
     "FROM healthcare_fact_claims"),
    ("What is our current employee headcount?",
     "SELECT COUNT(*) AS n FROM hr_fact_employees",
     "SELECT COUNT(*) AS n FROM hr_fact_employees WHERE is_active = 1"),
    ("Who is the top payer by paid amount?",
     "SELECT payer_id, SUM(paid_amount) AS s FROM healthcare_fact_claims "
     "GROUP BY 1 ORDER BY s ASC LIMIT 1",
     "SELECT payer_id, SUM(paid_amount) AS s FROM healthcare_fact_claims "
     "GROUP BY 1 ORDER BY s DESC LIMIT 1"),
]


@pytest.mark.parametrize("question,wrong,right", SILENT_WRONG, ids=[c[0] for c in SILENT_WRONG])
def test_known_silent_wrong_space_is_now_blocked(con, question, wrong, right):
    """Known, explicit business intent is checked before a wrong query executes."""
    verifier = Verifier(con)
    wrong_result, right_result = run_query(con, wrong), run_query(con, right)
    assert wrong_result.ok and right_result.ok
    assert wrong_result.rows[0][-1] != right_result.rows[0][-1], "not actually a wrong answer"
    findings = verifier.check_sql(wrong, question)
    assert any(f.blocking for f in findings), [str(f) for f in findings]
    assert verifier.check_sql(right, question) == []


def test_self_join_fanout_is_blocked(con):
    sql = (
        "SELECT ROUND(AVG(e.base_salary)) AS avg_salary FROM hr_fact_employees e "
        "JOIN hr_fact_employees m ON e.department_id = m.department_id WHERE e.is_active = 1"
    )
    findings = Verifier(con).check_sql(sql)
    assert any(f.check == "join_fanout" and f.blocking for f in findings)


def test_fanout_warning_blocks_or_corrects(con):
    verifier = Verifier(con)
    findings = verifier.check_sql(
        "SELECT SUM(s.completed) AS n FROM clinical_subjects s "
        "LEFT JOIN clinical_query_log q ON s.subject_id = q.subject_id")
    assert findings, "expected a fan-out finding to exist at all"
    assert any(f.blocking for f in findings), (
        "fan-out is advisory only; the inflated number is answered as fact")


# --------------------------------------------------------------------------
# 3. THE RETRIEVAL PATH
# --------------------------------------------------------------------------

def test_production_retrieval_uses_the_strategy_that_was_measured(con):
    from engine import retrieval

    # top_customer is one of the two questions vector misses at k=10.
    block = retrieval.schema_catalog_for("Who is our top wholesale customer by revenue?", con)
    assert "retail_customer_analytics" in block, (
        "the default strategy omits a table the measured configuration retrieves")


def test_retrieval_failure_falls_back_once_then_rebuilds(con):
    """A dead local index may cost one full-catalogue turn, not the process."""
    from engine import retrieval

    try:
        narrow = retrieval.schema_catalog_for("What is the overall claim denial rate?", con)
        retrieval._collection.invalidate()
        after = [retrieval.schema_catalog_for("What is the overall claim denial rate?", con)
                 for _ in range(3)]
        full = schema_catalog(con)
        assert after[0] == full, "the failing turn should safely use the full catalogue"
        assert all(block != full for block in after[1:]), (
            "later turns should rebuild instead of retaining a dead collection"
        )
        assert len(full) > 3 * len(narrow), "fallback should be dramatically larger"
        assert retrieval._collection is not None, (
            "a healthy collection should be restored after the fallback")
    finally:
        retrieval._forget_index()
        retrieval.build_index(con, rebuild=True)


# --------------------------------------------------------------------------
# 5. CONCURRENCY
# --------------------------------------------------------------------------

def test_run_query_is_safe_under_concurrency(con):
    """The one place that gets it right: run_query uses con.cursor()."""
    errors, results = [], []

    def go():
        try:
            result = run_query(con, "SELECT COUNT(*) AS n FROM healthcare_fact_claims")
            (results if result.ok else errors).append(result.error or result.rows[0][0])
        except Exception as exc:  # pragma: no cover - would itself be the bug
            errors.append(repr(exc))

    threads = [threading.Thread(target=go) for _ in range(24)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert len(set(results)) == 1, f"interleaved results: {set(results)}"


def test_shared_connection_helpers_are_thread_safe(con):
    truth = table_columns(con, "hr_fact_employees")
    bad = []

    def go():
        try:
            if table_columns(con, "hr_fact_employees") != truth:
                bad.append("wrong column list")
        except Exception as exc:
            bad.append(type(exc).__name__)

    threads = [threading.Thread(target=go) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not bad, f"{len(bad)}/40 concurrent DESCRIBEs were wrong or crashed: {Counter(bad)}"


def test_cross_domain_join_is_still_caught_under_concurrency(con):
    cross_domain = ("SELECT COUNT(*) FROM hr_fact_employees e "
                    "JOIN finance_erp_gl g ON e.employee_id = g.account_id")
    verifier = Verifier(con)
    assert verifier.check_sql(cross_domain), "single-threaded baseline must find it"

    missed = []

    def go():
        try:
            if not verifier.check_sql(cross_domain):
                missed.append("cross-domain join MISSED")
        except Exception as exc:
            missed.append(type(exc).__name__)

    threads = [threading.Thread(target=go) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not missed, f"{len(missed)}/40 concurrent verifications failed open: {Counter(missed)}"
