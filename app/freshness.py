"""Reload this repository's own modules when a deploy changes them under a running app.

Streamlit Community Cloud applies a git push to a running app without restarting
it. The entry script is read from disk on every run, so it always takes the new
code, but the modules it imported stay in ``sys.modules`` at the version first
imported, and Streamlit's file watcher does not reliably notice them. A push
that changed only a helper module therefore stayed invisible until someone
rebooted the app by hand.

:func:`refresh` runs at the start and end of every script run. It records each
local module's file modification time the first time it sees the module. When
any of them has changed on disk, it reloads every local module, dependencies
first, so a module that did ``from helper import thing`` picks up the new
``thing`` too, and clears Streamlit's data and resource caches so no cached
object outlives the code that built it.

A file first seen with a modification time later than this module's own import
counts as changed. Otherwise a deploy landing while the first run is still going
would be recorded as the baseline against a module that still holds the old
code. The cost of the rule is at most one unnecessary reload.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_SEEN: dict[str, int] = {}
_SELF = Path(__file__).resolve()
_STARTED_NS = time.time_ns()


def _local_modules(root: Path) -> list[tuple[str, object, Path]]:
    """Modules loaded from files under ``root``, in import order."""
    found = []
    for name, module in list(sys.modules.items()):
        if module is None or name == "__main__":
            continue
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if path == _SELF or root not in path.parents or "site-packages" in path.parts:
            continue
        found.append((name, module, path))
    return found


def _imported_names(name: str, module: object, path: Path) -> set[str]:
    """Every module name this file imports, with relative imports resolved."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return set()
    is_package = path.name == "__init__.py"
    package = name if is_package else name.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                try:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                except (ImportError, ValueError):
                    continue
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    expanded = set(names)
    for imported in names:
        parts = imported.split(".")
        expanded.update(".".join(parts[:i]) for i in range(1, len(parts)))
    return expanded


def _dependency_order(modules: list[tuple[str, object, Path]]) -> list[tuple[str, object, Path]]:
    """Every local module after the local modules it imports (cycles keep import order)."""
    by_name = {entry[0]: entry for entry in modules}
    deps = {name: (_imported_names(name, module, path) & by_name.keys()) - {name}
            for name, module, path in modules}
    ordered: list[tuple[str, object, Path]] = []
    state: dict[str, int] = {}

    def visit(name: str) -> None:
        if state.get(name):
            return
        state[name] = 1
        for dep in sorted(deps[name]):
            if not state.get(dep):
                visit(dep)
        state[name] = 2
        ordered.append(by_name[name])

    for name, _module, _path in modules:
        visit(name)
    return ordered


def _stamp(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def refresh(root: str | Path, *, clear_caches: bool = True) -> list[str]:
    """Reload local modules if any changed since last seen; return the names reloaded."""
    root = Path(root).resolve()
    with _LOCK:
        modules = _local_modules(root)
        changed = False
        for name, _module, path in modules:
            stamp = _stamp(path)
            if stamp is None:
                continue
            seen = _SEEN.get(name)
            if seen is None:
                _SEEN[name] = stamp
                changed = changed or stamp > _STARTED_NS
            elif seen != stamp:
                changed = True
        if not changed:
            return []
        reloaded = []
        # Dependencies first, so "from helper import thing" re-binds the new thing.
        for name, module, _path in _dependency_order(modules):
            importlib.reload(module)
            reloaded.append(name)
        for name, _module, path in modules:
            stamp = _stamp(path)
            if stamp is not None:
                _SEEN[name] = stamp
    if clear_caches:
        import streamlit as st

        st.cache_data.clear()
        st.cache_resource.clear()
    return reloaded
