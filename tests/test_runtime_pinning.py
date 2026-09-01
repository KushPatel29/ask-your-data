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

ROOT = Path(__file__).resolve().parent.parent
PINNED = (ROOT / ".python-version").read_text(encoding="utf-8").strip()


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
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
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
    config = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "dependency-name: python" in config
    assert "version-update:semver-minor" in config
    assert "version-update:semver-major" in config
