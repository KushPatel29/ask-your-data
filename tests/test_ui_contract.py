"""
The interface's own contract, asserted rather than eyeballed.

app/ui.py makes claims about itself — small text clears WCAG AA, hostile strings
never reach the page unescaped, every colour resolves to a defined token, the
pipeline strip's connectors mean something specific. Those are all arithmetic or
structure, so they belong in the test suite next to the accuracy contract rather
than in a comment saying they were checked once.

Everything here runs headless and deterministically. The things that genuinely
need a browser — computed font-variant-numeric, the 3ch guide grid, whether the
deep plan scrolls instead of pushing the page — were measured against the running
app with getComputedStyle and are documented where they are relied on; they are
not faked here, because a test that asserts a layout it cannot lay out is worse
than no test.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import audit_ui  # noqa: E402


@pytest.fixture(scope="module")
def rendered(con):
    ui = audit_ui.load_ui()
    return ui, audit_ui.render_all(ui, audit_ui.fixtures(con))


# --------------------------------------------------------------------------
# Structure and escaping.
# --------------------------------------------------------------------------

def test_every_panel_produces_balanced_markup(rendered):
    _ui, panels = rendered
    assert audit_ui.check_markup(panels) == []


def test_no_component_lets_hostile_text_through(rendered):
    """The audit pushes `<script>alert(1)</script>` through every text parameter."""
    _ui, panels = rendered
    assert audit_ui.check_escaping(panels) == []


def test_the_hostile_string_actually_reached_the_output(rendered):
    """Guards the guard: an escaping test passes trivially if nothing was rendered."""
    _ui, panels = rendered
    escaped = sum("&lt;script&gt;" in body for body in panels.values())
    assert escaped >= 4, "the hostile fixture stopped reaching the components"


# --------------------------------------------------------------------------
# Colour.
# --------------------------------------------------------------------------

def test_every_colour_reference_resolves_to_a_token(rendered):
    """A typo'd custom property paints the inherited colour and says nothing."""
    ui, _panels = rendered
    assert audit_ui.check_tokens(ui._CSS, audit_ui.token_table(ui._CSS)) == []


def test_small_text_clears_wcag_aa(rendered):
    ui, _panels = rendered
    rows = audit_ui.contrast_rows(audit_ui.token_table(ui._CSS))
    failing = [f"{label} {ratio:.2f}:1" for label, _fg, _bg, ratio, ok in rows if not ok]
    assert failing == []


def test_the_contrast_floor_does_not_drift(rendered):
    """The measured range is quoted in app/ui.py; this is what keeps it true."""
    ui, _panels = rendered
    rows = audit_ui.contrast_rows(audit_ui.token_table(ui._CSS))
    assert min(r[3] for r in rows) == pytest.approx(4.94, abs=0.01)


def test_only_two_accents_carry_meaning(rendered):
    """The whole argument of the file: amber means verified, cyan means machine.

    A third accent would cost both of them their meaning, so the palette is
    asserted rather than left to the next person's taste. --ayd-alert is not a
    third accent — it is a failure state, and it never marks a value.
    """
    ui, _panels = rendered
    tokens = audit_ui.token_table(ui._CSS)
    colours = {name for name, value in tokens.items() if value.startswith("#")}
    assert colours == {
        "--ayd-ground", "--ayd-panel", "--ayd-panel-2", "--ayd-line",
        "--ayd-ink", "--ayd-muted",
        "--ayd-machine", "--ayd-verified", "--ayd-alert",
    }


def test_amber_appears_only_on_the_verified_badge(rendered):
    """Nothing is amber unless a committed file backs it."""
    _ui, panels = rendered
    ambered = {name for name, body in panels.items() if "ayd-verified" in body}
    assert ambered == {"answer"}


# --------------------------------------------------------------------------
# The pipeline strip's connectors, which encode a directional claim.
# --------------------------------------------------------------------------

def _links(ui, **kwargs):
    ui.st.take()
    ui.pipeline(**kwargs)
    body = ui.st.take()
    import re
    return re.findall(r'class="ayd-arrow" data-on="([^"]*)"', body)


def test_a_lit_link_means_the_signal_crossed_it(rendered):
    ui, _panels = rendered
    assert _links(ui, retrieved=True, generated=True, guarded=True, executed=True) \
        == ["1", "1", "1"]


def test_demo_mode_does_not_claim_anything_was_generated(rendered):
    """GENERATE is honestly dark, so both segments touching it must be dark."""
    ui, _panels = rendered
    assert _links(ui, retrieved=True, generated=False, guarded=True, executed=True) \
        == ["0", "0", "1"]


def test_a_guard_block_breaks_the_link_after_the_guard_not_before(rendered):
    """The SQL really did reach the guard; nothing came out of it.

    Marking both adjacent segments as failed — which the first version did —
    claims something crossed out of the guard and went wrong downstream. EXECUTE
    never ran.
    """
    ui, _panels = rendered
    assert _links(ui, retrieved=True, generated=True, guarded="fail",
                  executed=False, attempts=3) == ["1", "1", "fail"]


def test_a_refusal_never_reaches_the_guard(rendered):
    ui, _panels = rendered
    assert _links(ui, retrieved=True, generated=True, guarded=False, executed=False) \
        == ["1", "0", "0"]


# --------------------------------------------------------------------------
# The readouts that report machine state.
# --------------------------------------------------------------------------

def test_the_guard_panel_draws_every_forbidden_verb(rendered):
    """"None of N forbidden verbs" is a claim; the N on screen are the evidence."""
    from engine.sql_guard import FORBIDDEN

    _ui, panels = rendered
    body = panels["guard_pass"]
    assert body.count('class="ayd-verb"') == len(FORBIDDEN)
    for verb in FORBIDDEN:
        assert f">{verb}</span>" in body


def test_only_the_verb_the_guard_named_is_lit(rendered):
    _ui, panels = rendered
    body = panels["guard_blocked"]
    assert body.count('data-hit="1"') == 1
    assert '<span class="ayd-verb" data-hit="1">DROP</span>' in body


def test_a_clean_first_attempt_renders_no_ledger(rendered):
    """There is no ledger to show when nothing was corrected."""
    _ui, panels = rendered
    assert panels["attempts_ok"] == ""


def test_the_ledger_has_one_row_per_attempt_and_names_every_error(rendered):
    """engine.assistant appends one correction per attempt that did not survive."""
    _ui, panels = rendered
    body = panels["attempts_corrected"]
    assert body.count('class="ayd-att-row"') == 3
    assert body.count('data-ok="0"') == 2   # the two that failed
    assert body.count('data-ok="1"') == 1   # the one that ran
    assert "Binder Error" in body           # the first error, not summarised away
    assert "joins two independent domains" in body   # and the second one too


def test_an_exhausted_loop_never_claims_an_attempt_ran(rendered):
    """MAX_ATTEMPTS corrections for MAX_ATTEMPTS attempts: every row is a failure.

    engine.assistant appends the error from the final attempt before falling out
    of the loop, so an exhausted answer has no accepted row at all — the ledger
    must not manufacture one to round the panel off.
    """
    _ui, panels = rendered
    body = panels["attempts_exhausted"]
    assert body.count('data-ok="1"') == 0
    assert body.count('data-ok="0"') == 3
    assert body.count(">ran<") == 0


def test_a_refusal_after_a_correction_still_shows_its_history(rendered):
    """The other way the loop ends early: the model corrected once, then declined."""
    ui, _panels = rendered
    ui.st.take()
    ui.attempt_ledger(attempts=2, corrections=["Binder Error: no such column"],
                      max_attempts=3, ok=False)
    body = ui.st.take()
    assert body.count('class="ayd-att-row"') == 2
    assert ">stopped<" in body and ">ran<" not in body


def test_the_plan_is_duckdbs_and_the_tree_structure_survives(rendered, con):
    """Operator names and branch guides come from EXPLAIN, not from parsing SQL."""
    _ui, panels = rendered
    body = panels["query_plan"]
    for operator in ("HASH_JOIN", "HASH_GROUP_BY", "SEQ_SCAN", "TOP_N"):
        assert operator in body
    # A two-child operator must produce a branch and a continuation line.
    assert "├─" in body and "│" in body


def test_every_guide_level_is_exactly_three_characters(rendered, con):
    """The 3ch boxes in the stylesheet depend on this being exact, not typical.

    IBM Plex Mono has no Box Drawing block — measured in the running app, the
    corner glyphs advance 6.334px against the font's own 6.913px step — so each
    level is boxed at a fixed 3ch and the glyph sits inside. Chunking the guide
    by three is only correct because every level is written as three characters.
    """
    nodes = audit_ui.plan_nodes(
        "SELECT c.channel FROM marketing_dim_channel c "
        "JOIN marketing_fact_spend s ON s.channel = c.channel GROUP BY 1", con)
    guides = [node["guide"] for node in nodes]
    assert any(guides), "the plan produced no tree guides at all"
    for guide in guides:
        assert len(guide) % 3 == 0
        for i in range(0, len(guide), 3):
            assert guide[i:i + 3] in ("   ", "│  ", "└─ ", "├─ ")


def test_the_plan_foot_sets_the_estimate_against_the_outturn(rendered):
    """A prediction is only useful with its outturn beside it."""
    ui, _panels = rendered
    ui.st.take()
    ui.query_plan([{"guide": "", "name": "SEQ_SCAN", "detail": "t", "card": 12000}],
                  plan_ms=0.4, returned=1, truncated=False)
    body = ui.st.take()
    assert "12,000" in body and "1</b> came back" in body


def test_the_shape_line_reports_the_row_cap_being_hit(rendered):
    _ui, panels = rendered
    assert "cap of 200 reached" in panels["result_shape_capped"]
    assert "cap of" not in panels["result_shape"]


def test_the_build_marker_survives_in_the_masthead(rendered):
    ui, panels = rendered
    assert f"build <b>{ui.build_marker()}</b>" in panels["masthead"]


def test_a_capped_plan_says_it_was_capped(rendered):
    """The renderer stops at 200 operators; stopping silently would be a lie."""
    ui, _panels = rendered
    ui.st.take()
    ui.query_plan([{"guide": "", "name": "SEQ_SCAN", "detail": "t", "card": 1}],
                  plan_ms=0.4, total=412, returned=1)
    assert "1</b> of 412 operators shown" in ui.st.take()


def test_an_uncapped_plan_does_not_say_it_was_capped(rendered):
    ui, _panels = rendered
    ui.st.take()
    ui.query_plan([{"guide": "", "name": "SEQ_SCAN", "detail": "t", "card": 1}],
                  plan_ms=0.4, total=1, returned=1)
    body = ui.st.take()
    assert "operators shown" not in body and "<b>1</b> operators" in body


# --------------------------------------------------------------------------
# The compiled path. PLAN and GENERATE are the same position and different
# claims, which is the one thing this strip must never blur.
# --------------------------------------------------------------------------

def _cells(ui, **kwargs):
    import re

    ui.st.take()
    ui.pipeline(**kwargs)
    body = ui.st.take()
    return re.findall(r'class="ayd-step" data-on="[^"]*">([a-z ]+?)(?:<|$)', body)


def test_a_compiled_turn_says_plan_and_never_generate(rendered):
    """Lighting GENERATE for a compiled query is the claim that a model wrote it.

    That would be the single most misleading pixel in this app, so the cell is
    replaced rather than reused: `planned` and `generated` occupy one position
    and exactly one of them can be true of any turn.
    """
    ui, _panels = rendered
    cells = _cells(ui, retrieved=True, planned=True, guarded=True, executed=True)
    assert "plan" in cells
    assert "generate" not in cells


def test_a_model_turn_still_says_generate(rendered):
    ui, _panels = rendered
    cells = _cells(ui, retrieved=True, generated=True, guarded=True, executed=True)
    assert "generate" in cells
    assert "plan" not in cells


def test_a_refused_plan_breaks_the_link_after_plan_not_before(rendered):
    """Same directional rule the guard block already has, applied to PLAN.

    The question DID reach the compiler, so the segment into PLAN is lit; what
    the compiler then decided is PLAN's own cell to report. The segment OUT of
    it carries the failure, because that is where the turn stopped, and
    everything downstream stays dark because nothing crossed.
    """
    ui, _panels = rendered
    assert _links(ui, retrieved=True, planned="fail", guarded=False,
                  executed=False) == ["1", "fail", "0"]


def test_the_trace_marks_missed_words_differently_from_unbindable_ones(rendered):
    """Three word groups, three meanings, and only one of them is a debt.

    `bound` is what the plan used. `loose` is what nothing in 71 tables
    contains, excused from the coverage denominator rather than silently
    dropped. `missed` is what the warehouse HAS and this plan did not use — the
    only group that says the planner left something on the table, and the only
    one drawn in the alert colour.
    """
    _ui, panels = rendered
    body = panels["plan_trace"]
    assert 'data-w="bound"' in body
    assert 'data-w="missed"' in body
    assert 'data-w="loose"' in body


def test_the_trace_reports_a_refusal_as_a_refusal(rendered):
    _ui, panels = rendered
    assert "no plan met the floor" in panels["plan_trace_refused"]
    assert "of the question bound" in panels["plan_trace"]


def test_the_masthead_never_calls_a_keyless_session_a_demo(rendered):
    """It read "demo · committed reference SQL" whenever no key was set.

    That stopped being true the moment the compiler started answering the chat
    box: keyless turns are compiled from the schema, not served from
    evals/golden_questions.yaml. A masthead naming the wrong engine is worse
    than one naming none — it is the first claim a visitor reads, and the SQL
    below it would quietly contradict it.
    """
    ui, _panels = rendered
    ui.st.take()
    ui.masthead(tables=71, domains=11, live=False)
    keyless = ui.st.take()
    assert "compiled from the schema" in keyless
    assert "committed reference SQL" not in keyless

    ui.st.take()
    ui.masthead(tables=71, domains=11, live=True)
    assert "model-authored" in ui.st.take()


def test_the_grounding_panel_claims_no_tokens_when_none_were_spent(rendered):
    """On a compiled turn there is no prompt, so there is no schema budget."""
    ui, _panels = rendered

    class Hit:
        def __init__(self, table, domain, score):
            self.table, self.domain, self.score = table, domain, score

    hits = [Hit("hr_fact_employees", "hr", 0.032)]
    ui.st.take()
    ui.grounding(hits, total_tables=71, tokens_used=0, tokens_full=12741)
    compiled = ui.st.take()
    assert "No prompt and no tokens" in compiled
    assert "tokens of schema in the prompt" not in compiled

    ui.st.take()
    ui.grounding(hits, total_tables=71, tokens_used=3241, tokens_full=12741)
    assert "tokens of schema in the prompt" in ui.st.take()


# --------------------------------------------------------------------------
# Startup robustness. These are source-level checks, in the same spirit as
# tests/test_demo_mode.py's "the app defers the model import" — the app module
# runs Streamlit at import and cannot be imported in a test process.
# --------------------------------------------------------------------------

def _app_source() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parent.parent
            / "app" / "streamlit_app.py").read_text(encoding="utf-8")


def test_the_semantic_layer_cannot_take_the_app_down():
    """`get_layer` runs at import on a public deployment.

    An exception there is not a degraded feature, it is a blank page where the
    app used to be. The compiler cannot work without the layer, but the accuracy
    contract can — so the failure has to be caught and named, exactly as
    `warm_retrieval` already does for the schema index.
    """
    import ast

    tree = ast.parse(_app_source())
    layer_fn = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "get_layer")
    assert any(isinstance(n, ast.Try) for n in ast.walk(layer_fn)), (
        "get_layer must not be able to raise at import")
    assert any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
               and n.value.value is None for n in ast.walk(layer_fn)), (
        "get_layer must return None on failure so the caller can degrade")


def test_the_keyless_page_checks_the_planner_is_available():
    """A None layer must reach a branch that says so, not an AttributeError."""
    source = _app_source()
    assert "PLANNER_READY" in source
    assert "if not PLANNER_READY:" in source


def test_a_pasted_key_is_never_cached_across_sessions():
    """@st.cache_resource is per-container, and a key is per-visitor.

    Caching the Assistant on the connection alone would hand the second visitor
    in a container the first one's client, and therefore the first one's bill.
    """
    import ast

    tree = ast.parse(_app_source())
    live = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_live_assistant")
    decorators = {ast.unparse(d) for d in live.decorator_list}
    assert not any("cache" in d for d in decorators), decorators
    assert "session_state" in ast.unparse(live)


# --------------------------------------------------------------------------
# The two components added in the UI pass: a refusal that belongs to this
# palette, and the empty state's layer summary.
# --------------------------------------------------------------------------

def test_a_refusal_claims_neither_accent(rendered):
    """st.warning's olive was the loudest thing on a near-black page.

    Replacing it raised the real question, and the palette rule above answers
    it: amber means a committed file backs this number, cyan means a machine
    derived it, red is a failure. A refusal is none of the three — it is a
    different KIND of result, not a louder one — so the panel is neutral and
    its mono heading does the work.
    """
    _ui, panels = rendered
    body = panels["refusal"]
    assert "ayd-verified" not in body, "amber is reserved for verified values"
    assert "ayd-alert" not in body, "a refusal is not a failure"
    assert "ayd-refusal-head" in body, "the heading is what marks it"


def test_the_two_refusals_name_themselves_differently(rendered):
    """"the grammar cannot express this" and "this warehouse has no such words"
    are different facts, and the heading is where they are told apart."""
    _ui, panels = rendered
    assert "not compiled" in panels["refusal"]
    assert "nothing to bind" in panels["refusal_unbound"]


def test_the_layer_summary_states_where_its_numbers_came_from(rendered):
    """The panel's whole claim is that nothing in it was hand-written."""
    _ui, panels = rendered
    body = panels["layer_summary"]
    assert "probed from DuckDB" in body
    for n in ("285", "197", "56", "797"):
        assert n in body


def test_the_new_panels_escape_hostile_text(rendered):
    """Guards the guard: both must actually receive the hostile fixture."""
    _ui, panels = rendered
    for name in ("refusal", "layer_summary"):
        assert "&lt;script&gt;" in panels[name], f"{name} never saw the fixture"
        assert "<script>" not in panels[name]
