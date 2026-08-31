"""
Builds the read-only analytics warehouse and describes it for the model.

The warehouse is an in-memory DuckDB rebuilt from the vendored CSVs on every run
(they total a few MB, so this is instant and always fresh — nothing binary is
committed). Every table is named `<domain>_<table>` so the several dim_customer /
fact_orders tables from different domains never collide.

`schema_catalog()` renders the tables, their descriptions, and their columns into
the text the language model reads to write SQL. Good text-to-SQL lives or dies on
this catalog, so it is generated from the real loaded schema, not hand-typed.
"""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data_manifest import DOMAINS, MANIFEST, table_name  # noqa: E402

DATA_DIR = ROOT / "data"


def build_warehouse(data_dir: Path = DATA_DIR) -> duckdb.DuckDBPyConnection:
    """Load every vendored CSV into a fresh in-memory DuckDB and return it."""
    con = duckdb.connect(":memory:")
    for domain, table, _source, _desc in MANIFEST:
        # Large facts are committed gzipped. DuckDB's read_csv_auto decompresses
        # transparently, so the only thing that changes is which name we look
        # for - the AML payment fact is 15.7 MB raw and 3.5 MB gzipped, and a
        # portfolio repo should not carry the difference for no gain.
        plain = data_dir / domain / f"{table}.csv"
        gzipped = data_dir / domain / f"{table}.csv.gz"
        csv = plain if plain.exists() else gzipped
        if not csv.exists():
            raise FileNotFoundError(
                f"vendored data missing: {plain} (or {gzipped.name}) - run scripts/vendor_data.py"
            )
        name = table_name(domain, table)
        con.execute(
            f'CREATE TABLE "{name}" AS '
            f"SELECT * FROM read_csv_auto(?, header=true, sample_size=-1)",
            [str(csv)],
        )

    _seal(con)
    return con


def _seal(con) -> None:
    """Drop the connection's access to the filesystem and the network.

    The SQL guard is a mutation-verb filter. It was never a boundary, and a
    plain SELECT is enough to leave the database entirely:

        SELECT * FROM read_csv_auto('/path/to/.env')
        SELECT * FROM glob('*')
        SELECT * FROM read_text('engine/sql_guard.py')

    All three are genuine single SELECT statements with no forbidden verb in
    them, so the guard passes them, the verifier passes them, and DuckDB
    executes them. Verified: glob returned real paths from this repo, read_text
    returned the guard's own source, and read_csv_auto over https AUTOLOADED
    httpfs and made a live request - even though INSTALL and LOAD are both on
    the forbidden list, because the autoloader never issues those statements.

    On a public demo with no login that is arbitrary remote file read.

    Adding function names to the keyword list would not fix it; that is a
    denylist against a surface DuckDB keeps extending. Turning the capability
    off is a real boundary. It has to happen AFTER loading, because loading is
    itself a filesystem read - and enable_external_access is one-way, which is
    exactly the property wanted here: nothing later in the process can re-enable
    it, including SQL the model writes.
    """
    for setting in (
        "SET autoinstall_known_extensions=false",
        "SET autoload_known_extensions=false",
        # Resource ceilings. The filesystem settings above stop the query
        # LEAVING the database; these stop it eating the container from
        # inside. `SELECT COUNT(*) FROM aml_fact_transactions a,
        # aml_fact_transactions b WHERE a.amount_cad > b.amount_cad` is a
        # single SELECT with no forbidden verb — 10,059,889,401 pairs, and it
        # was still running after 20 seconds with memory climbing. The wall
        # clock lives in engine/query.py (DuckDB has no statement_timeout);
        # this is the memory half, so a hash join that would have swallowed
        # the container fails as a query instead of as an outage.
        #
        # 1.5GB is deliberately above the ~290MB the app measures at rest and
        # below what a small container can survive losing. `threads=4` bounds
        # how much CPU one visitor's question can take from the others, since
        # Streamlit serves every session from this one process.
        "SET memory_limit='1500MB'",
        "SET threads=4",
        # A parser bound rather than a runtime one: deeply nested expressions
        # are a cheap way to burn stack before execution ever starts.
        "SET max_expression_depth=500",
        "SET enable_external_access=false",  # must be last: one-way, and the
                                             # settings above cannot be set
                                             # after it
    ):
        con.execute(setting)


def table_columns(con, name):
    """[(column_name, column_type), ...] for one table.

    Goes through con.cursor(), not con.execute(). The Streamlit app shares one
    connection across every session via @st.cache_resource, and a DuckDB
    connection keeps ONE result set: a second caller's execute() overwrites the
    first's before it has fetched. Measured under 40 threads, this returned an
    empty column list 13 times instead of raising - and an empty column list is
    not an error anywhere downstream, it is a table that looks like it has no
    columns. run_query already used a cursor; these read helpers did not.
    """
    return [(r[0], r[1]) for r in con.cursor().execute(f'DESCRIBE "{name}"').fetchall()]


def schema_catalog(con: duckdb.DuckDBPyConnection) -> str:
    """The schema description handed to the model, grouped by business domain."""
    desc = {(d, t): dsc for d, t, _s, dsc in MANIFEST}
    lines = []
    for domain, blurb in DOMAINS.items():
        lines.append(f"\n### Domain: {domain} — {blurb}")
        for d, t, _s, _dsc in MANIFEST:
            if d != domain:
                continue
            name = table_name(d, t)
            cols = ", ".join(f"{c} {ty}" for c, ty in table_columns(con, name))
            lines.append(f"- {name}: {desc[(d, t)]}")
            lines.append(f"    columns: {cols}")
    return "\n".join(lines).strip()


def table_names(con):
    return [r[0] for r in con.cursor().execute(
        "SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall()]


if __name__ == "__main__":
    con = build_warehouse()
    names = table_names(con)
    print(f"warehouse built: {len(names)} tables\n")
    print(schema_catalog(con))
