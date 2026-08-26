"""
Does retrieval actually beat pasting the whole schema, and beat keyword search?

Retrieval is easy to add and hard to justify. The justification has to be a
number, because the alternative — pasting all six domains into every prompt —
already works, and "we added a vector database" is not a reason to stop it
working. So this scores three strategies on the same corpus and the same
questions:

  full     every table, every turn. Recall is 1.0 by construction; the cost is
           the entire catalogue in the context window on every question.
  keyword  token overlap between the question and the table document. The
           honest baseline: if this wins, the embeddings are dead weight.
  vector   Chroma + all-MiniLM-L6-v2 (384-d, ONNX, local, keyless).

GROUND TRUTH COMES FROM THE REFERENCE SQL, NOT FROM MY OPINION
`evals/golden_questions.yaml` pairs each question with the SQL that correctly
answers it. The tables that SQL selects from are, by definition, the tables the
question needs. So the labels are derived, not authored — nobody sat down and
decided which tables "feel" relevant, which is exactly where retrieval
benchmarks usually go soft.

The headline metric is RECALL@k, and only recall. Precision is not the goal: a
retrieved table the query does not use costs a few hundred tokens, while a
MISSED table makes the answer impossible. Those two errors are not
commensurable and averaging them into an F1 would hide the one that matters.

    python scripts/run_retrieval_eval.py
    python scripts/run_retrieval_eval.py --k 4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import retrieval  # noqa: E402
from engine.warehouse import build_warehouse, schema_catalog, table_names  # noqa: E402

GOLDEN = ROOT / "evals" / "golden_questions.yaml"


def tables_in_sql(sql: str, known: set[str]) -> set[str]:
    """The tables a query actually reads.

    Matched against the warehouse's real table list rather than by parsing FROM
    and JOIN clauses: these queries nest subqueries and CTEs, and a regex that
    tries to understand SQL structure will quietly miss one. Every table name
    here is `<domain>_<table>`, distinctive enough that a substring match on
    word boundaries is both sufficient and hard to get wrong.
    """
    found = set()
    lowered = (sql or "").lower()
    for name in known:
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            found.add(name)
    return found


def approx_tokens(text: str) -> int:
    """~4 characters per token. Good enough to compare orders of magnitude."""
    return max(1, len(text) // 4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--k", type=int, default=retrieval.DEFAULT_K)
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Report recall at k = 1..10 instead of a single k",
    )
    args = ap.parse_args(argv)

    con = build_warehouse()
    known = set(table_names(con))
    questions = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

    cases = []
    for row in questions:
        needed = tables_in_sql(row.get("sql", ""), known)
        if not needed:
            # A question whose reference SQL names no known table cannot label
            # anything. Reported, not silently dropped.
            print(f"  ! no ground-truth tables parsed for {row.get('id')!r} - excluded")
            continue
        cases.append({"id": row["id"], "question": row["question"], "needed": needed})

    full_catalog_tokens = approx_tokens(schema_catalog(con))
    print(f"\nWarehouse: {len(known)} tables · golden questions with labels: {len(cases)}")
    print(f"Full-catalogue prompt block: ~{full_catalog_tokens:,} tokens per turn\n")

    ks = range(1, 11) if args.sweep else [args.k]

    for k in ks:
        rows = []
        for strategy in ("keyword", "vector", "hybrid"):
            hit = 0
            missed_tables = 0
            total_tables = 0
            tokens = 0
            misses: list[str] = []
            for case in cases:
                got = {
                    r.table
                    for r in (
                        retrieval.retrieve_keyword(case["question"], k=k, con=con)
                        if strategy == "keyword"
                        else retrieval.retrieve(case["question"], k=k, con=con)
                        if strategy == "vector"
                        else retrieval.retrieve_hybrid(case["question"], k=k, con=con)
                    )
                }
                need = case["needed"]
                total_tables += len(need)
                missing = need - got
                missed_tables += len(missing)
                if not missing:
                    hit += 1
                else:
                    misses.append(f"{case['id']} (missing {', '.join(sorted(missing))})")
                tokens += approx_tokens(
                    retrieval.schema_catalog_for(case["question"], con, k=k, strategy=strategy)
                )
            rows.append(
                {
                    "strategy": strategy,
                    "full_recall": hit / len(cases),
                    "table_recall": (total_tables - missed_tables) / total_tables,
                    "tokens": tokens / len(cases),
                    "misses": misses,
                }
            )

        print(f"k = {k}")
        headings = (
            f"  {'strategy':<10} {'questions fully covered':>24} "
            f"{'tables recalled':>17} {'~tokens/turn':>14}"
        )
        print(headings)
        print(f"  {'full':<10} {'100.0%':>24} {'100.0%':>17} {full_catalog_tokens:>14,}")
        for row in rows:
            print(
                f"  {row['strategy']:<10} {row['full_recall']:>23.1%} {row['table_recall']:>17.1%} "
                f"{row['tokens']:>14,.0f}"
            )
        if not args.sweep:
            for row in rows:
                if row["misses"]:
                    print(f"\n  {row['strategy']} missed:")
                    for miss in row["misses"]:
                        print(f"    - {miss}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
