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


def test_status_rail_allows_formatting_but_escapes_other_markup(rendered):
    ui, _panels = rendered
    ui.st.take()
    ui.status_rail([
        ("provider", '<s>local</s><br><em>ready</em><img src=x onerror="alert(1)">'),
    ])
    body = ui.st.take()
    assert "<s>local</s><br><em>ready</em>" in body
    assert "<img" not in body
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in body


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


def test_the_worked_examples_remain_reachable_from_the_ask_page():
    """Trust center is evidence architecture, not a replacement for discovery.

    These 39 questions used to live under the question box. Moving their only
    copy behind another workspace made a working feature look removed.
    """
    import ast

    tree = ast.parse(_app_source())
    keyless = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "render_keyless")
    body = ast.unparse(keyless)
    assert "Worked examples" in body
    assert "render_demo_mode(connection)" in body


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


# --------------------------------------------------------------------------
# The build marker, which is the only way to tell a fresh deploy from a warm
# container still serving the old one.
# --------------------------------------------------------------------------

def test_the_build_marker_covers_the_engine_not_just_the_ui(rendered):
    """It used to watch four files, and the guess failed the first real test.

    The commit that added `engine/planner.py` — a 1,400-line engine that answers
    every question on the keyless path — touched none of the four and produced
    an identical marker. So did the one that added `engine/semantics.py`. Four
    consecutive commits, two of them entire new subsystems, all reported the
    same build.

    Asserted structurally rather than by hashing: the point is which files are
    consulted, and a hash test would pass while silently watching the wrong set.
    """
    import ast
    from pathlib import Path

    source = (Path(audit_ui.__file__).resolve().parent.parent
              / "app" / "ui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_marker")
    body = ast.unparse(fn)
    assert "'engine'" in body or '"engine"' in body, (
        "the marker must hash the engine, or it cannot see a new one")
    assert "'app'" in body or '"app"' in body
    assert "data_manifest" in body
    assert "evals" in body


def test_the_build_marker_is_stable_and_short(rendered):
    ui, _panels = rendered
    first, second = ui.build_marker(), ui.build_marker()
    assert first == second, "the marker must not move between two reads"
    assert len(first) == 7 and all(c in "0123456789abcdef" for c in first)


def test_the_build_marker_moves_when_the_engine_moves(rendered, tmp_path):
    """Proves the watch list is live, by editing a file inside it.

    engine/planner.py is the file the old marker was blind to, so it is the one
    worth proving. The edit is made and reverted in a temp copy of nothing —
    the real file is restored in a finally, and the test fails loudly if it
    cannot be.
    """
    from pathlib import Path

    ui, _panels = rendered
    target = (Path(audit_ui.__file__).resolve().parent.parent
              / "engine" / "planner.py")
    before = ui.build_marker()
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# marker probe\n")
        assert ui.build_marker() != before, (
            "editing engine/planner.py must change the build marker")
    finally:
        target.write_bytes(original)
    assert ui.build_marker() == before, "the probe did not restore cleanly"


def test_the_example_row_passes_height_down_every_rung(rendered):
    """Equal-height buttons need the whole chain, not just the column.

    The first attempt stretched `stColumn` and stopped. Streamlit puts a
    `stElementContainer` between the column and the button as a display:block
    box sized to its own content, so the columns went to 65px and three of the
    five buttons stayed at 46. Measured on the running app: 46/65/46/65/65
    before, 64/64/64/64/64 after.

    Source-level, because the defect is a computed layout and the suite has no
    browser — but the rung that broke is nameable, so it is named.
    """
    ui, _panels = rendered
    css = ui._CSS
    scope = ".st-key-ayd-examples"
    assert f'{scope} [data-testid="stElementContainer"]' in css, (
        "the container between column and button must be in the chain")
    for rung in (f'{scope} [data-testid="stColumn"]',
                 f"{scope} .stButton",
                 f"{scope} .stButton > button"):
        assert rung in css, f"missing rung: {rung}"


# --------------------------------------------------------------------------
# The hand-written query path. A person is exactly as untrusted as the model.
# --------------------------------------------------------------------------

def test_hand_written_sql_goes_through_the_same_guard():
    """The editor is only defensible because nothing is relaxed for it.

    Source-level: the suite has no browser, but the boundary is nameable. What
    must be true is that `_manual_turn` validates before it executes, and that
    it executes through `run_query` — which validates again — rather than
    touching the connection itself.
    """
    import ast

    tree = ast.parse(_app_source())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_manual_turn")
    body = ast.unparse(fn)
    assert "validate_sql" in body, "user SQL must hit the guard"
    assert "run_query" in body, "user SQL must go through the capped executor"
    assert "_verify_now" in body, "user SQL must be verified like everything else"
    # No raw cursor: the executor is the only thing that touches the warehouse.
    assert ".cursor(" not in body and ".execute(" not in body


def test_a_blocked_hand_written_query_never_reaches_execute():
    """The refusal path must return before any execution is attempted."""
    import ast

    tree = ast.parse(_app_source())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_manual_turn")
    src = ast.unparse(fn)
    guard_at = src.index("validate_sql")
    run_at = src.index("run_query")
    early_return = src.index("return entry")
    assert guard_at < early_return < run_at, (
        "the guard, its early return, and only then the executor")


def test_the_editor_uses_a_form_so_the_draft_cannot_race_the_button():
    """A bare text_area commits on blur, and the button click is processed
    against the value the server last saw — so editing the SQL and pressing Run
    submitted the ORIGINAL query. Measured: the handler answered "that is the
    same query". A form batches the field with its submit."""
    import ast

    tree = ast.parse(_app_source())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_sql_editor")
    body = ast.unparse(fn)
    assert "st.form(" in body
    assert "form_submit_button" in body


def test_results_can_be_taken_away(rendered):
    """An answer you cannot leave with is a demonstration of an answer."""
    source = _app_source()
    assert "download_button" in source
    assert "text/csv" in source


def test_the_schema_browser_reads_roles_from_the_semantic_layer(rendered):
    """The sidebar used to carry eleven paragraphs of prose that could not tell
    you whether a `department` column existed. The layer knows; this is the only
    place in the app that shows what it concluded per column."""
    ui, panels = rendered
    source = _app_source()
    assert "_searchable_schema" in source
    assert "layer.tables" in source
    assert hasattr(ui, "column_list")


def test_the_global_font_rule_does_not_swallow_the_icon_font(rendered):
    """The most visible bug this app ever shipped, and it was one CSS line.

    Streamlit draws icons as LIGATURES: a span containing the literal text
    `keyboard_arrow_right` that the Material Symbols font composes into a
    glyph. Our reset is `html, body, [class*="st-"] { font-family: … }` with
    `!important`, every one of those spans carries an `st-emotion-cache-…`
    class, so the glyph font never applied and the ligature fell back to a font
    with no such ligature — which simply drew the letters.

    Every expander read "keyboard_arrow_right The accuracy contract…", the
    sidebar toggle read "keyboard_double_arrow_left", the password field read
    "visibility". Measured on the deployed app: 15 icon spans, all computing to
    "IBM Plex Sans", each 166px of text inside a 24px box.

    `unset` does not fix it — font-family is inherited, so it resolves to the
    parent, which the same selector also matches. The family has to be named.
    """
    ui, _panels = rendered
    css = ui._CSS
    assert '[data-testid="stIconMaterial"]' in css, (
        "no exception for Streamlit's icon spans")
    block = css[css.index('[data-testid="stIconMaterial"]'):]
    block = block[:block.index("}") + 1]
    assert "Material Symbols" in block, "the icon font must be named, not unset"
    assert "unset" not in block, (
        "unset inherits the overridden font — name the family instead")


def test_there_is_a_way_back_outside_the_sidebar(rendered):
    """The only reset used to be at the bottom of the sidebar, under the schema
    map, the browser and the API-key box — and Streamlit collapses that sidebar
    by default on a phone. On the device most people open a shared link with,
    the example questions vanished on the first click and never came back."""
    source = _app_source()
    assert "_transcript_header" in source
    assert "Start over" in source
    # and both routes must clear the same state, or they will drift
    assert source.count("_reset_conversation") >= 3


# --------------------------------------------------------------------------
# The operations panel, and the wiring that makes it able to tell the truth.
#
# Every other readout in app/ui.py describes ONE turn. This one describes the
# process, which means it is the first panel in this app that could lie by
# omission: a turn that is not recorded is a turn the refusal rate does not
# count. So the tests here are mostly about the RECORDER, not the renderer.
# --------------------------------------------------------------------------

def test_the_operations_panel_claims_no_accent_it_has_not_earned(rendered):
    """Machine measurements of itself are cyan by definition. Amber would say
    CI re-checks these numbers, which nothing does — they are this session's."""
    _ui, panels = rendered
    assert "ayd-verified" not in panels["operations"]
    assert "ayd-ops-fill" in panels["operations"]


def test_an_empty_ledger_says_so_instead_of_drawing_zeroes(rendered):
    """A dashboard of zeroes reads as a broken dashboard. A sentence reads as
    an app that has not been asked anything yet."""
    _ui, panels = rendered
    body = panels["operations_empty"]
    assert "No turns recorded yet" in body
    assert "ayd-ops-grid" not in body, "no stat cells until there are stats"


def test_a_keyless_session_does_not_report_a_token_cell(rendered):
    """Reporting `0 tokens` invites the reader to wonder what it would have
    been. Reporting nothing says the axis does not apply to this session."""
    _ui, panels = rendered
    assert "TOKENS" not in panels["operations"].upper()


def test_the_panel_publishes_the_limits_of_its_own_trail(rendered):
    """An undocumented control is how a reviewer ends up relying on something
    that was never load-bearing."""
    _ui, panels = rendered
    assert "records die with the container" in panels["operations"]


def test_every_path_that_answers_also_records(rendered):
    """The compiler path, the guard-block path and the hand-written-SQL path.

    A trail with a hole in it exactly where the interesting queries go is not a
    trail, and the editor is where a visitor writes the interesting queries.
    """
    import ast

    tree = ast.parse(_app_source())
    for name in ("_plan_turn", "_manual_turn"):
        fn = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == name)
        body = ast.unparse(fn)
        returns = body.count("return entry")
        records = body.count("_audit_turn(")
        assert records >= returns, (
            f"{name} has {returns} exits and only {records} audit calls — "
            "a turn that returns without recording is a turn the refusal rate "
            "does not count")


def test_the_recorder_can_never_take_an_answer_down_with_it():
    """An observer that can break the observed is a worse feature than none."""
    import ast

    tree = ast.parse(_app_source())
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_audit_turn")
    assert any(isinstance(node, ast.Try) for node in ast.walk(fn)), \
        "_audit_turn must swallow its own failures"


def test_the_session_ceiling_admits_what_it_is():
    """A limit that presents itself as a security control is worse than no
    limit, because somebody stops looking for the real one."""
    source = _app_source()
    assert "MAX_QUESTIONS_PER_SESSION" in source
    assert "not a security control" in source
    assert "spend cap" in source


# --------------------------------------------------------------------------
# The chart. It is the most prominent element of a turn, which is exactly why
# it has to be a restatement of the result and never an interpretation of it.
# --------------------------------------------------------------------------

def test_the_chart_uses_the_machine_accent_and_no_other(rendered):
    _ui, panels = rendered
    body = panels["result_chart"]
    assert "ayd-verified" not in body, "a chart is not a verified value"
    assert 'class="bar' in body


def test_the_chart_prints_the_value_so_a_length_is_never_estimated(rendered):
    """A bar you have to measure against an axis to read is a picture of a
    number. The number goes at the end of the bar."""
    _ui, panels = rendered
    assert 'class="val"' in panels["result_chart"]
    assert "41.3M" in panels["result_chart"]


def test_the_chart_says_when_it_is_showing_less_than_the_grid(rendered):
    """Silent truncation reads as "this is all of it"."""
    _ui, panels = rendered
    assert "showing 5 of 10 rows" in panels["result_chart"]


def test_the_chart_carries_a_text_alternative(rendered):
    """An SVG with no accessible name is a decorative rectangle to a screen
    reader, and this one carries the finding."""
    _ui, panels = rendered
    assert 'role="img"' in panels["result_chart"]
    assert 'aria-label=' in panels["result_chart"]


def test_a_time_series_is_drawn_as_a_line_not_a_ranking(rendered):
    """Drawing a series over time as sorted bars hides the one thing it is
    for. The axis type decides, because that is a fact about the data."""
    _ui, panels = rendered
    assert 'class="line"' in panels["result_chart_line"]
    assert 'class="bar' not in panels["result_chart_line"]


def test_a_single_row_gets_no_chart():
    """ui.answer already says the number in a sentence, and one bar at 100% of
    itself is a rectangle rather than a comparison."""
    import audit_ui as _audit

    ui = _audit.load_ui()
    ui.st.take()
    ui.result_chart([("Electronics", 41280624.23)], label="department",
                    measure="sum revenue")
    assert ui.st.take().strip() == ""


def test_negative_values_draw_nothing_rather_than_draw_wrong():
    """A bar for "less than nothing" is a different chart — a baseline in the
    middle. Until it exists, the grid is the honest answer."""
    import audit_ui as _audit

    ui = _audit.load_ui()
    ui.st.take()
    ui.result_chart([("a", 5.0), ("b", -3.0)], label="x", measure="y")
    assert ui.st.take().strip() == ""


def test_the_chart_only_fires_on_a_result_that_is_a_shape():
    """Source-level, because the decision lives in the app and the rule is the
    feature: one label column, one numeric column, more than one row."""
    import ast

    tree = ast.parse(_app_source())
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_chart_shape")
    body = ast.unparse(fn)
    assert "len(frame.columns) != 2" in body
    assert "len(frame) < 2" in body


def test_the_chart_never_reorders_a_query_that_named_its_own_order():
    """A query with an ORDER BY is drawn exactly as it came back: re-sorting a
    top-N would draw the answer to a query nobody ran, and re-sorting a time
    series would scramble the axis.

    A GROUP BY with NO ORDER BY is the opposite case — DuckDB returns those
    groups in any order it likes, and this repo has been bitten once already by
    reading meaning into that. Sorting there is not an interpretation; it is
    the only defensible reading of a result the query left unordered.
    """
    import ast

    tree = ast.parse(_app_source())
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_result_chart")
    body = ast.unparse(fn)
    assert "order by" in body, "the ORDER BY of the query has to be consulted"
    assert "sort_values" in body
    assert "kind != 'line'" in body or 'kind != "line"' in body


def test_a_certified_metric_names_its_own_pipeline_stage(rendered):
    """A committed definition is neither generated by a model nor inferred by
    the planner. Amber is earned here because metrics.yaml carries a value CI
    re-runs, which is the same contract as the answer badge."""
    _ui, panels = rendered
    body = panels["pipeline_metric"]
    assert 'data-on="cert">metric' in body
    assert "generate" not in body and ">plan" not in body
    assert 'data-on="1">verify' in body


def test_voice_has_distinct_ready_and_unconfigured_states(rendered):
    _ui, panels = rendered
    assert 'data-ready="1"' in panels["voice_ready"]
    assert "gpt-transcribe" in panels["voice_ready"]
    assert "review transcript" in panels["voice_ready"]
    assert 'data-ready="0"' in panels["voice_disabled"]
    assert "add key in sidebar" in panels["voice_disabled"]


def test_every_streamlit_data_cache_has_a_hard_entry_bound():
    """Question text and edited SQL are cache keys. A public process must not
    retain an unlimited number of either for the life of its container."""
    lines = [line.strip() for line in _app_source().splitlines()
             if line.strip().startswith("@st.cache_data")]
    assert lines
    assert all("max_entries=" in line for line in lines), lines
