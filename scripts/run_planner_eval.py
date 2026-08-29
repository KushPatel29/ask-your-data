"""
Grade the keyless planner, on two contracts, and report the difference.

    python scripts/run_planner_eval.py            # both sets
    python scripts/run_planner_eval.py --sweep    # the MIN_COVERAGE curve
    python scripts/run_planner_eval.py --golden   # only the model's contract

WHY TWO SETS

`evals/planner_questions.yaml` is what the planner claims to do: ordinary ad-hoc
questions, phrased the way somebody types them, whose answers are a GROUP BY and
a WHERE away. `evals/golden_questions.yaml` is what the MODEL is graded on, and
its questions deliberately carry business definitions the schema does not
contain -- net collection rate, cost-optimal alert threshold, incremental lift.

Reporting only the first would be marking my own homework. Reporting only the
second would be scoring a compiler on a language it was never given. Both are
printed, and the gap between them is the actual finding: it is a measurement of
where a language model stops being optional.

OUTCOMES

  match     the planner's own SQL returned the value the reference SQL returns
  differs   it ran, and disagreed -- the number is on screen, so this is
            reported separately from a crash rather than folded into "fail"
  refused   the planner declined to compile, which for the two questions marked
            expect_refusal is the PASS condition
  error     the SQL did not execute; this should always be zero, because the
            planner emits SQL from a grammar rather than a language model

`differs` is the count that matters. A refusal costs a visitor an answer; a
`differs` costs them a wrong one, and the whole design of the confidence gate is
aimed at keeping that column at zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import retrieval  # noqa: E402
from engine.planner import MIN_COVERAGE, plan_question  # noqa: E402
from engine.query import run_query  # noqa: E402
from engine.semantics import Layer  # noqa: E402
from engine.warehouse import build_warehouse  # noqa: E402

PLANNER_SET = ROOT / "evals" / "planner_questions.yaml"
GOLDEN_SET = ROOT / "evals" / "golden_questions.yaml"

MATCH, DIFFERS, REFUSED, ERROR = "match", "differs", "refused", "error"


def _same(got, expected) -> bool:
    """Compare one returned value with a contract value.

    The tolerance is relative rather than absolute: `expect` values are baked at
    six decimal places, so an average of 1428.852728 must match 1428.85272829976
    while a percentage of 7.3 must not match 8.2.
    """
    if expected is None:
        return got is None
    if isinstance(expected, bool) or isinstance(got, bool):
        return got == expected
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        scale = max(abs(float(expected)), 1.0)
        return abs(float(got) - float(expected)) <= 1e-5 * scale
    return str(got) == str(expected)


def _key(row: tuple) -> tuple:
    """A row, normalised so two result sets can be compared as multisets."""
    out = []
    for value in row:
        if isinstance(value, bool) or value is None:
            out.append(repr(value))
        elif isinstance(value, (int, float)):
            out.append(f"{float(value):.6g}")
        else:
            out.append(str(value))
    return tuple(out)


def _same_result(got_rows, ref_rows, width: int) -> bool:
    """Do two result sets carry the same answer?

    Compared as MULTISETS of the reference's own columns, and the reason is a
    bug in the first version of this script: it compared `rows[0][0]`, the first
    cell of each side. A GROUP BY with no ORDER BY has no first row -- DuckDB is
    free to return the groups in any order -- so twelve correct breakdowns were
    scored as wrong because the planner happened to emit `outbound` where the
    reference's `ORDER BY 1` emitted `inbound`. The rows were identical.

    Width is the REFERENCE's column count: the reference defines what was asked
    for, and the planner is allowed to return more context alongside it (the
    rank shape returns the measure it sorted on next to the name). It is not
    allowed to return less.
    """
    if len(got_rows) != len(ref_rows):
        return False
    if any(len(row) < width for row in got_rows):
        return False
    return (sorted(_key(row[:width]) for row in got_rows)
            == sorted(_key(row[:width]) for row in ref_rows))


def grade(case: dict, con, layer, *, min_coverage: float) -> tuple[str, str]:
    """Run one case. Returns (outcome, one-line note)."""
    question = case["question"]
    try:
        hits = [h.table for h in retrieval.retrieve_hybrid(question)]
    except Exception:
        hits = []
    result = plan_question(question, layer, retrieved=hits,
                           min_confidence=min_coverage)

    if case.get("expect_refusal"):
        if result.refused:
            return REFUSED, "refused as required"
        return DIFFERS, f"answered a question marked unanswerable: {result.sql!r}"

    if result.refused:
        return REFUSED, result.reason.split(".")[0]

    ran = run_query(con, result.sql)
    if not ran.ok:
        return ERROR, ran.error
    if not ran.rows:
        return DIFFERS, "no rows"

    reference = case.get("sql", "").strip()
    if reference:
        ref = run_query(con, reference)
        if not ref.ok:
            return ERROR, f"reference SQL failed: {ref.error}"
        if not _same_result(ran.rows, ref.rows, len(ref.columns)):
            got = ran.rows[0][0]
            return DIFFERS, (f"{ran.row_count} rows vs {ref.row_count}"
                             if ran.row_count != ref.row_count
                             else f"same shape, different values (e.g. {got!r})")
        return MATCH, f"{ran.row_count} row(s)"

    # A case with no reference SQL is a scalar contract only.
    got = ran.rows[0][0]
    if not _same(got, case["expect"]):
        return DIFFERS, f"got {got!r}, contract says {case['expect']!r}"
    return MATCH, f"{got!r}"


def run_set(path: Path, con, layer, *, label: str, min_coverage: float,
            quiet: bool = False) -> dict[str, int]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    tally = {MATCH: 0, DIFFERS: 0, REFUSED: 0, ERROR: 0}
    if not quiet:
        print(f"\n{label}  ({len(cases)} questions, {path.name})")
        print("-" * 78)
    for case in cases:
        outcome, note = grade(case, con, layer, min_coverage=min_coverage)
        tally[outcome] += 1
        if not quiet:
            mark = {MATCH: "ok  ", DIFFERS: "DIFF", REFUSED: "--  ", ERROR: "ERR "}[outcome]
            print(f"  {mark} {case['id']:34} {note[:60]}")
    if not quiet:
        total = sum(tally.values())
        answered = tally[MATCH] + tally[DIFFERS]
        print("-" * 78)
        print(f"  {tally[MATCH]} match · {tally[DIFFERS]} differs · "
              f"{tally[REFUSED]} refused · {tally[ERROR]} error   of {total}")
        if answered:
            print(f"  when it answered, it was right {100 * tally[MATCH] / answered:.0f}% "
                  f"of the time ({tally[MATCH]}/{answered})")
    return tally


def sweep(con, layer) -> None:
    """The coverage-gate curve. This is where MIN_COVERAGE comes from.

    Every row is the same 64 questions at a different floor, so the trade is
    visible rather than asserted: as the gate falls the planner answers more
    questions and gets more of them wrong, and the point of the exercise is to
    show that `differs` is bought, not avoided by luck.
    """
    print(f"\n{'gate':>6} {'match':>7} {'differs':>9} {'refused':>9} {'error':>7}")
    print("-" * 44)
    for gate in (0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90):
        totals = {MATCH: 0, DIFFERS: 0, REFUSED: 0, ERROR: 0}
        for path in (PLANNER_SET, GOLDEN_SET):
            got = run_set(path, con, layer, label="", min_coverage=gate, quiet=True)
            for key, value in got.items():
                totals[key] += value
        flag = "  <- shipped" if abs(gate - MIN_COVERAGE) < 1e-9 else ""
        print(f"{gate:>6.2f} {totals[MATCH]:>7} {totals[DIFFERS]:>9} "
              f"{totals[REFUSED]:>9} {totals[ERROR]:>7}{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="print the MIN_COVERAGE trade-off curve")
    parser.add_argument("--golden", action="store_true",
                        help="only the model's contract")
    parser.add_argument("--planner", action="store_true",
                        help="only the planner's contract")
    args = parser.parse_args()

    con = build_warehouse()
    layer = Layer(con)
    try:
        retrieval.build_index(con)
    except Exception as exc:  # retrieval is an optimisation, not a requirement
        print(f"(schema index unavailable, planning without it: {exc})")

    if args.sweep:
        sweep(con, layer)
        return 0

    tallies = []
    if not args.golden:
        tallies.append(run_set(PLANNER_SET, con, layer,
                               label="THE PLANNER'S CONTRACT — ordinary ad-hoc questions",
                               min_coverage=MIN_COVERAGE))
    if not args.planner:
        tallies.append(run_set(GOLDEN_SET, con, layer,
                               label="THE MODEL'S CONTRACT — questions written to need a model",
                               min_coverage=MIN_COVERAGE))
    errors = sum(t[ERROR] for t in tallies)
    if errors:
        print(f"\n{errors} question(s) produced SQL that would not execute.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
