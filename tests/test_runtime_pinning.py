"""The interpreter this repo is proven on, and the three places that must agree.

`.python-version`, the CI matrix and the container base image are the same
decision written down three times. When they drift, the thing CI proves stops
being the thing the deployment runs — and the drift is silent, because each
file on its own looks reasonable.

This is not hypothetical here. Dependabot opened `Bump python from 3.12-slim to
3.14-slim`, and its own Security workflow failed on the result: onnxruntime and
tokenizers publish compiled wheels per CPython ABI, so a base image ahead of
those publishers has nothing to install and pip falls back to building from
source in a slim image with no toolchain.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PINNED = (ROOT / ".python-version").read_text(encoding="utf-8").strip()


def _repo_file(relative: str) -> str:
    """Read a file that exists in the repository but not in the built image.

    `.dockerignore` excludes `.github`, and it should: CI configuration is not
    something the runtime needs and shipping it would only widen the image.
    But the suite runs INSIDE that image as well — CI builds the container and
    runs pytest in it, which is the point of the docker job — so a test that
    reads a repo-only path fails there for a reason that has nothing to do with
    what it is checking. That is exactly what happened: the tests job went
    green, the docker job went red, and the finding was my own test.

    Skipping is the honest outcome rather than a dodge, and the condition is
    narrow enough to say so: it fires only when the file is ABSENT, so a
    present-but-wrong config still fails everywhere it can be seen.
    """
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not part of the runtime image (see .dockerignore)")
    return path.read_text(encoding="utf-8")


def test_the_pinned_interpreter_is_a_concrete_minor_version():
    assert re.fullmatch(r"3\.\d+", PINNED), PINNED


def test_the_container_runs_the_interpreter_the_suite_was_proven_on():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    bases = re.findall(r"^FROM\s+python:([0-9.]+)", dockerfile, flags=re.M)
    assert bases, "the image must pin a concrete python base"
    for base in bases:
        assert base.startswith(PINNED), (
            f"Dockerfile builds on python:{base} but .python-version pins {PINNED} — "
            "the container is supposed to be the reproducible proof of the suite, "
            "and on a different interpreter it is proof of something else")


def test_ci_runs_the_interpreter_the_container_ships():
    ci = _repo_file(".github/workflows/ci.yml")
    versions = re.findall(r'python-version:\s*"?([0-9.]+)"?', ci)
    assert versions, "CI must state its interpreter"
    for version in versions:
        assert version == PINNED, (
            f"CI runs {version} while the deployment runs {PINNED}")


def test_the_lint_target_matches_the_interpreter():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    target = config.get("tool", {}).get("ruff", {}).get("target-version")
    if target:
        assert target == "py" + PINNED.replace(".", ""), (
            f"ruff targets {target} while the deployment runs {PINNED}")


def test_a_bot_cannot_raise_the_interpreter_on_its_own():
    """The bump has to be a deliberate act, taken with the wheel availability
    of onnxruntime and tokenizers actually checked — not a green checkmark on a
    diff that changes one line and breaks the image."""
    config = _repo_file(".github/dependabot.yml")
    assert "dependency-name: python" in config
    assert "version-update:semver-minor" in config
    assert "version-update:semver-major" in config


def test_the_rail_reports_the_interpreter_the_deployment_really_got():
    """A managed host picks the Python version in a dropdown at app-creation
    time and does not read `.python-version`, so a new deployment can come up
    on an interpreter CI never ran. The three files above agree with each other
    inside the repository; only the rail can say what the DEPLOYMENT got.

    Asserted at source level because the value is read at import time from a
    module that starts Streamlit on import.
    """
    source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "PINNED_PYTHON" in source
    assert '.python-version' in source, "the pin must be read, never typed"
    assert '("runtime", runtime)' in source, "the rail must carry it"
    # A mismatch has to be visible as a mismatch. `<u>` is the rail's alert
    # form; `<em>` is its muted form. Rendering a wrong interpreter in the
    # muted form would put the fact on the page and still hide it.
    assert 'python <u>{running}</u>' in source
