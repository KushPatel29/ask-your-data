"""
Demo mode: what the app serves when there is no API key.

The point of these tests is that demo mode is honest and self-checking. It
executes the reference SQL committed in the accuracy contract, so the numbers a
visitor sees are the numbers CI asserts — and if the vendored data ever drifts,
`matches_contract` goes false and says so on screen instead of quietly showing a
wrong figure.
"""

from __future__ import annotations

import pytest

from engine import demo_mode

CASES = demo_mode.load_golden_questions()


def test_golden_file_is_well_formed():
    assert len(CASES) >= 10
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate golden question ids"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_question_answers_without_a_model(con, case):
    """No API key is set in CI. Every one of these must still answer."""
    result = demo_mode.answer(con, case)
    assert result.ok, f"{case['id']}: {result.result.error}"
    assert result.headline != "—"
    assert result.sql.strip()


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_answer_matches_the_contract(con, case):
    result = demo_mode.answer(con, case)
    assert result.matches_contract, (
        f"{case['id']}: demo returned {result.headline!r}, contract expects "
        f"{case['expect']!r}"
    )


def test_contract_mismatch_is_detected_not_hidden(con):
    """
    A guard that never fires is worse than no guard: prove matches_contract
    actually goes false when the answer disagrees.
    """
    case = dict(CASES[0])
    case["expect"] = "definitely-not-the-answer"
    result = demo_mode.answer(con, case)
    assert result.ok
    assert not result.matches_contract


def test_headline_formats_numbers_readably(con):
    by_id = {c["id"]: c for c in CASES}
    if "expected_nrv_total" in by_id:
        result = demo_mode.answer(con, by_id["expected_nrv_total"])
        assert "," in result.headline, "large figures should be thousands-separated"
        assert not result.headline.endswith(".0")
    if "denial_rate" in by_id:
        assert demo_mode.answer(con, by_id["denial_rate"]).headline == "8.2"


def test_questions_group_by_domain():
    grouped = demo_mode.questions_by_domain(CASES)
    assert len(grouped) >= 5
    assert sum(len(v) for v in grouped.values()) == len(CASES)


def test_mode_switch_reads_either_credential_variable():
    assert demo_mode.has_api_key({"ANTHROPIC_API_KEY": "x"})
    assert demo_mode.has_api_key({"ANTHROPIC_AUTH_TOKEN": "x"})
    assert not demo_mode.has_api_key({})
    assert not demo_mode.has_api_key({"ANTHROPIC_API_KEY": ""})


def _imported_modules(source: str) -> set[str]:
    """Every module name this source imports, by AST rather than by substring."""
    import ast

    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_demo_mode_does_not_import_the_anthropic_client():
    """
    Demo mode must not depend on the model client being constructible, so a
    missing or broken anthropic install cannot take the public demo down.
    Checked against actual import statements -- the module's prose mentions the
    assistant, and it should be free to.
    """
    import inspect

    imported = _imported_modules(inspect.getsource(demo_mode))
    assert not {m for m in imported if "anthropic" in m}
    assert "engine.assistant" not in imported


def test_app_defers_the_model_import_until_after_the_demo_branch():
    """
    Source-level check: the demo branch must st.stop(), and no MODULE-LEVEL
    import of engine.assistant may sit above it -- otherwise a broken anthropic
    install takes down the key-free demo before it renders.
    """
    from pathlib import Path

    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    lines = app_path.read_text(encoding="utf-8").splitlines()

    branch_line = next(i for i, line in enumerate(lines) if line.startswith("if not LIVE_MODE:"))
    assert any("st.stop()" in line for line in lines[branch_line:branch_line + 5])

    top_level_assistant_imports = [
        i for i, line in enumerate(lines)
        if line.startswith(("from engine.assistant import", "import engine.assistant"))
    ]
    assert top_level_assistant_imports, "expected the live path to import the assistant"
    for i in top_level_assistant_imports:
        assert i > branch_line, (
            f"line {i + 1} imports engine.assistant at module level above the "
            "demo-mode stop"
        )
