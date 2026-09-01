"""
Read-only guard for model-generated SQL.

The language model writes the SQL, so it is never trusted. Before anything runs,
`validate_sql` proves the statement is a single read-only query: it must be one
statement, start with SELECT or WITH, and contain no data- or schema-modifying
keyword. Comments and string/identifier literals are stripped first so a value
like WHERE note = 'please DROP TABLE' can't trip the check.

This is the safety boundary the whole assistant leans on, which is why it has its
own exhaustive test suite and runs before the executor ever sees the SQL.
"""

import re

# Parser work happens before the statement watchdog can interrupt execution.
# Bound the text at the trust boundary so a pasted multi-megabyte SELECT cannot
# turn validation itself into the denial of service the query clock prevents.
MAX_SQL_CHARS = 20_000

# Anything that writes data, changes schema, touches the filesystem, or loads an
# extension. DuckDB-specific verbs (ATTACH, COPY, INSTALL, LOAD, PRAGMA, EXPORT)
# are included alongside the standard DML/DDL set.
FORBIDDEN = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE",
    "MERGE", "UPSERT", "ATTACH", "DETACH", "COPY", "INSTALL",
    "LOAD", "PRAGMA", "CALL", "SET", "RESET", "EXPORT", "IMPORT", "GRANT",
    "REVOKE", "VACUUM", "ANALYZE", "CHECKPOINT",
]
# REPLACE is deliberately NOT on that list. It only writes as part of
# `CREATE OR REPLACE`, and CREATE is already forbidden - while `replace()` is an
# everyday string function and `SELECT * REPLACE (x*2 AS x)` is DuckDB's column
# replacement syntax. Forbidding the bare word rejected both, so a model doing
# ordinary string cleaning burned a retry on a query that was never dangerous.
# A guard that blocks correct SQL trains the loop to work around the guard.

# Second layer, behind the real one. engine/warehouse._seal() revokes the
# connection's filesystem and network access entirely, which is the actual
# boundary; a name list can never keep up with a surface DuckDB keeps extending.
# These are here anyway because a caller who builds a connection WITHOUT _seal
# should still be refused, and because "forbidden function: read_csv_auto" is a
# better correction message for the retry loop than DuckDB's permission error.
FORBIDDEN_FUNCTIONS = [
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "read_blob", "read_ndjson", "read_ndjson_auto", "glob",
    "sniff_csv", "parquet_scan", "parquet_metadata", "parquet_schema",
    "duckdb_settings", "duckdb_extensions", "iceberg_scan", "delta_scan",
    "postgres_scan", "sqlite_scan", "mysql_scan", "query", "query_table",
]


def _strip_literals(sql: str) -> str:
    """Blank out comments and string/identifier literals in ONE left-to-right scan.

    This used to be a sequence of independent regexes, and the order was a
    vulnerability rather than a style choice. Line comments were stripped before
    string literals, so a `--` INSIDE a quoted string was treated as the start of
    a comment and ate the rest of the line:

        SELECT '--' ; CREATE TABLE pwned AS SELECT 42

    became `SELECT ' ` - no semicolon left for the single-statement check, no
    CREATE left for the keyword scan. The guard passed it, and DuckDB executes
    stacked statements from one execute() call, so it ran. Verified end to end
    against the real warehouse: it created tables, dropped them, and wrote a file
    to disk with COPY.

    Sequenced regexes cannot fix this, because each pass is blind to the context
    the others establish. Whichever construct OPENS first has to win, and only a
    single scan knows that. Everything is replaced by a same-shaped blank so that
    offsets and token boundaries survive for the checks that run afterwards.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]

        if sql.startswith("/*", i):                      # block comment
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")
        elif sql.startswith("--", i):                    # line comment
            end = sql.find("\n", i)
            i = n if end == -1 else end
            out.append(" ")
        elif ch == "'":                                  # single-quoted string
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":  # '' escape
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" '' ")
        elif ch == '"':                                  # quoted identifier
            i += 1
            while i < n:
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(' "" ')
        elif ch == "$":                                  # dollar-quoted string
            m = re.match(r"\$(\w*)\$", sql[i:])
            if m:
                tag = m.group(0)
                end = sql.find(tag, i + len(tag))
                i = n if end == -1 else end + len(tag)
                out.append(" '' ")
            else:
                out.append(ch)
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def validate_sql(sql: str):
    """Return (ok: bool, reason: str). ok=True means the SQL is a single
    read-only statement that is safe to execute."""
    if not sql or not sql.strip():
        return False, "empty query"
    if len(sql) > MAX_SQL_CHARS:
        return False, f"query exceeds the {MAX_SQL_CHARS:,}-character safety limit"

    cleaned = _strip_literals(sql).strip().rstrip(";").strip()

    if ";" in cleaned:
        return False, "only a single statement is allowed"

    upper = cleaned.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "query must start with SELECT or WITH (read-only)"

    for kw in FORBIDDEN:
        if re.search(rf"\b{kw}\b", upper):
            return False, f"forbidden keyword: {kw}"

    # Matched with a trailing "(" so a COLUMN called glob or read_text is not
    # mistaken for the function of the same name.
    for fn in FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{fn.upper()}\s*\(", upper):
            return False, f"forbidden function: {fn}"

    return True, "ok"
