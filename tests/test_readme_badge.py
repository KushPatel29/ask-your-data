"""The test-count badge is a claim, so it gets checked like every other one.

Five repositories in this portfolio published a `tests-N passing` badge with
nothing holding it to a test run, and a badge only moves when someone remembers
to move it. Elsewhere the same drift ran the other way and lasted longer: a
guard that counted `^def test_` pinned one badge at 182 while the suite ran 345,
and because correcting the README by hand then failed CI, the wrong number
outlived several rounds of new tests. Count what pytest collects, or do not
claim a count.

Collection runs in a subprocess rather than off the current session, so the
answer does not depend on whether someone invoked the whole suite or one file.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def test_the_badge_matches_what_pytest_collects():
    # shields.io percent-encodes the thousands separator: 1,098 is written
    # tests-1%2C098%20passing. Strip the encoding, not the digits - a regex that
    # grabs runs of digits also finds the "20" in %20passing.
    badge = re.search(r"tests-([\d,]|%2C)+%20passing", README.read_text(encoding="utf-8"))
    assert badge, "README no longer carries a test-count badge"
    claimed = int(re.sub(r"%2C|,", "", badge.group(0)[len("tests-"):-len("%20passing")]))

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT,
    )
    # A module that fails to import is reported as an error and its cases are
    # simply absent from the total, so an environment problem would otherwise
    # surface as "the badge is wrong" and send someone to edit a correct README.
    errors = re.search(r"(\d+) errors?\b", proc.stdout)
    assert not errors, (
        "collection did not complete - " + errors.group(0) + " during collection, "
        "so the count below would be short. Fix the import error, not the badge:\n"
        + proc.stdout[-2000:]
    )
    found = re.search(r"(\d+) tests? collected", proc.stdout)
    assert found, f"could not read a collected-test count from pytest:\n{proc.stdout[-2000:]}"
    actual = int(found.group(1))

    assert claimed == actual, (
        f"badge claims {claimed} tests, the suite collects {actual}. "
        "Update the badge in README.md."
    )
