"""A deploy that changes only a helper module must reach a running app.

Streamlit Cloud pulls a push into the running process. The entry script is
re-read on every run; imported modules are not, so a helper-only change used to
stay invisible until a manual reboot. These tests drive ``refresh`` the way the
entry script does: once to record what is loaded, then again after a file
changes on disk.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import uuid

import pytest

from app.freshness import _SEEN, _STARTED_NS, refresh


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A throwaway package: ``helper`` holds a value, ``user`` imports it by name."""
    name = f"pkg_{uuid.uuid4().hex[:8]}"
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "helper.py").write_text("VALUE = 1\n")
    (root / "user.py").write_text(f"from {name}.helper import VALUE\n")
    _backdate(root)
    monkeypatch.syspath_prepend(str(tmp_path))
    helper = importlib.import_module(f"{name}.helper")
    user = importlib.import_module(f"{name}.user")
    yield tmp_path, root, name, helper, user
    for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        sys.modules.pop(module, None)
        _SEEN.pop(module, None)


def _backdate(root):
    """Date every file before the guard started, as a checkout before the app's start is."""
    for path in root.iterdir():
        os.utime(path, ns=(_STARTED_NS - 60_000_000_000, _STARTED_NS - 60_000_000_000))


def _rewrite(path, text):
    before = path.stat().st_mtime_ns
    path.write_text(text)
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


def test_an_unchanged_tree_reloads_nothing(package):
    tmp_path, _root, _name, helper, _user = package
    assert refresh(tmp_path, clear_caches=False) == []
    original = helper.__dict__.get("VALUE")
    assert refresh(tmp_path, clear_caches=False) == []
    assert helper.VALUE == original == 1


def test_a_changed_helper_reaches_the_modules_that_imported_it(package):
    tmp_path, root, name, helper, user = package
    refresh(tmp_path, clear_caches=False)
    _rewrite(root / "helper.py", "VALUE = 2\n")

    reloaded = refresh(tmp_path, clear_caches=False)

    assert f"{name}.helper" in reloaded and f"{name}.user" in reloaded
    assert reloaded.index(f"{name}.helper") < reloaded.index(f"{name}.user"), (
        "a dependency has to reload before the module that imported from it"
    )
    assert helper.VALUE == 2
    assert user.VALUE == 2, "a `from helper import VALUE` copy must not stay stale"
    assert refresh(tmp_path, clear_caches=False) == [], "one change, one reload"


def test_modules_outside_the_root_are_left_alone(package, tmp_path_factory):
    tmp_path, root, name, helper, _user = package
    refresh(tmp_path, clear_caches=False)
    _rewrite(root / "helper.py", "VALUE = 3\n")
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    assert refresh(elsewhere, clear_caches=False) == []
    assert helper.VALUE == 1


def test_an_import_cycle_still_reloads_once(package):
    tmp_path, root, name, helper, _user = package
    (root / "ping.py").write_text(f"import {name}.pong\nVALUE = 1\n")
    (root / "pong.py").write_text(f"import {name}.ping\n")
    _backdate(root)
    ping = importlib.import_module(f"{name}.ping")
    refresh(tmp_path, clear_caches=False)
    _rewrite(root / "ping.py", f"import {name}.pong\nVALUE = 2\n")
    reloaded = refresh(tmp_path, clear_caches=False)
    assert sorted(reloaded) == sorted(set(reloaded)), "each module reloads exactly once"
    assert ping.VALUE == 2


def test_a_deploy_before_the_first_check_is_not_taken_as_the_baseline(package):
    """The first run imports a helper; a deploy lands before that run records it.
    Recording the new mtime as the baseline would leave the old code loaded for good."""
    tmp_path, root, name, helper, _user = package
    _rewrite(root / "helper.py", "VALUE = 5\n")
    os.utime(root / "helper.py", ns=(time.time_ns(), time.time_ns()))
    reloaded = refresh(tmp_path, clear_caches=False)
    assert f"{name}.helper" in reloaded
    assert helper.VALUE == 5
