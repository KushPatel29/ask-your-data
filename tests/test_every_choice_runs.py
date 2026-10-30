"""Every widget choice on every page runs without raising.

Render tests run each page with its widgets at their defaults, the one state a
visitor leaves first. On 2026-10-30 a sweep of every portfolio app found a page
that raised only after a radio was changed (the pricing heatmap crashed on
"Gross margin $"). This test runs each choice from the default state, one
widget at a time. For a choice that switches workspace (a radio, segmented
control or pills), every widget that appears only in that workspace gets a run
of its own as well.

It runs pages through Streamlit's own script runner (AppTest), so it is the
real execution path. Page errors are recorded in ``app.exception``; a value the
harness itself cannot set (a radio listed by its formatted label, or a widget a
visitor cannot change because it is disabled) is not a page error and is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# --- per repository -------------------------------------------------------
PAGES: list[str] = [
    "app/streamlit_app.py",
]
ROUTER: str | None = None
EXTRA_PATHS: list[str] = []
# ---------------------------------------------------------------------------

TIMEOUT = 180
#: A ceiling on runs per page, so a page with many widgets stays a bounded test.
MAX_RUNS = 90
SWITCHERS = ("radio", "segmented_control", "pills")

for extra in [str(ROOT)] + [str(ROOT / p) for p in EXTRA_PATHS]:
    if extra not in sys.path:
        sys.path.insert(0, extra)

try:  # Streamlit raises this when a test sets a widget a visitor could not
    from streamlit.testing.v1.errors import AppTestError
except ImportError:  # pragma: no cover - older Streamlit
    AppTestError = RuntimeError
#: Values the harness itself cannot set; none of them is a page error.
HARNESS_ERRORS = (TypeError, ValueError, KeyError, IndexError, AttributeError, AppTestError)


def _enabled(widgets) -> list[tuple[int, object]]:
    """Widgets a visitor can change; a disabled one is shown but not offered."""
    return [
        (i, w)
        for i, w in enumerate(widgets)
        if not getattr(getattr(w, "proto", None), "disabled", False)
    ]


def _fresh(page: str):
    from streamlit.testing.v1 import AppTest

    if ROUTER:
        app = AppTest.from_file(str(ROOT / ROUTER), default_timeout=TIMEOUT)
        app.run()
        app.switch_page(page)
    else:
        app = AppTest.from_file(str(ROOT / page), default_timeout=TIMEOUT)
    app.run()
    return app


def _moves(app) -> list[tuple]:
    out = []
    for kind in SWITCHERS:
        for i, widget in _enabled(getattr(app, kind)):
            for option in list(getattr(widget, "options", []))[:12]:
                out.append((kind, i, widget.label, "value", option))
    for i, widget in _enabled(app.selectbox):
        count = len(widget.options)
        for position in sorted({0, 1, 2, count - 1} & set(range(count))):
            out.append(("selectbox", i, widget.label, "index", position))
    for kind in ("checkbox", "toggle"):
        for i, widget in _enabled(getattr(app, kind)):
            out.append((kind, i, widget.label, "flip", None))
    for i, widget in _enabled(app.multiselect):
        out.append(("multiselect", i, widget.label, "value", []))
        if widget.options:
            out.append(("multiselect", i, widget.label, "value", [widget.options[0]]))
    for i, widget in _enabled(app.slider):
        out.append(("slider", i, widget.label, "min", None))
        out.append(("slider", i, widget.label, "max", None))
    for i, widget in _enabled(app.select_slider):
        out.append(("select_slider", i, widget.label, "first", None))
        out.append(("select_slider", i, widget.label, "last", None))
    return out


def _find(app, kind, index, label):
    widgets = getattr(app, kind)
    same = [w for w in widgets if getattr(w, "label", "") == label]
    if same:
        return same[0]
    return widgets[index] if index < len(widgets) else None


def _apply(widget, how, arg):
    if how == "index":
        return widget.select_index(arg)
    if how == "flip":
        return widget.set_value(not widget.value)
    if how in ("min", "max"):
        low, high, value = widget.min, widget.max, widget.value
        target = low if how == "min" else high
        if isinstance(value, (list, tuple)):
            target = (low, value[1]) if how == "min" else (value[0], high)
        return widget.set_value(target)
    if how in ("first", "last"):
        return widget.set_value(widget.options[0 if how == "first" else -1])
    return widget.set_value(arg)


def _attempt(page, steps):
    """Run ``steps`` from a fresh page; None when the harness cannot set one."""
    app = _fresh(page)
    for kind, index, label, how, arg in steps:
        widget = _find(app, kind, index, label)
        if widget is None:
            return None
        try:
            _apply(widget, how, arg).run()
        except HARNESS_ERRORS:
            return None
        if app.exception:
            return app
    return app


def _describe(steps) -> str:
    return " > ".join(f"{kind} {label!r} {how}={arg!r}" for kind, _i, label, how, arg in steps)


@pytest.mark.parametrize("page", PAGES, ids=[Path(p).stem for p in PAGES])
def test_every_choice_on_the_page_runs_without_raising(page):
    base = _fresh(page)
    assert not base.exception, [e.message for e in base.exception]

    first = _moves(base)
    seen = {(kind, label) for kind, _i, label, _how, _arg in first}
    plans = [[move] for move in first]
    failures, runs = [], 0
    while plans and runs < MAX_RUNS:
        steps = plans.pop(0)
        app = _attempt(page, steps)
        if app is None:
            continue
        runs += 1
        if app.exception:
            failures.append(
                f"{_describe(steps)}: "
                + " | ".join(f"{e.type}: {e.message}" for e in app.exception)
            )
            continue
        if len(steps) == 1 and steps[0][0] in SWITCHERS:
            for move in _moves(app):
                if (move[0], move[2]) not in seen:
                    plans.append(steps + [move])
    assert not failures, "; ".join(failures)
    assert runs or not first, f"none of the {len(first)} choices on {page} could be set"
