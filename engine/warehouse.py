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
    return con


def table_columns(con, name):
    """[(column_name, column_type), ...] for one table."""
    return [(r[0], r[1]) for r in con.execute(f'DESCRIBE "{name}"').fetchall()]


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
    return [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall()]


if __name__ == "__main__":
    con = build_warehouse()
    names = table_names(con)
    print(f"warehouse built: {len(names)} tables\n")
    print(schema_catalog(con))
