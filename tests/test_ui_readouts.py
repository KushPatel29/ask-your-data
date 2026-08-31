"""
The two readouts added in this pass, held to the contract the rest of app/ui.py
already keeps.

tests/test_ui_contract.py audits every panel that `scripts/audit_ui.py` knows how
to render. The verifier and few-shot panels are not in that script's `render_all`
yet — folding them in is a one-function edit to a file this track does not own —
so their audit lives here instead, running the SAME helpers over the same
hostile fixture. Nothing is asserted here that the audit script could not assert;
it is the same arithmetic on new markup.

Two of these are not really UI tests at all. `test_the_rule_board_matches_the_
module_that_owns_the_rules` reads engine/verify.py's source for every Finding it
constructs and holds the app's board against it, because a board of rule names is
a claim about a module the app does not own and hand-maintained claims go stale
in silence. `test_a_verification_refusal_never_prints_the_models_correction_text`
guards a leak, not a layout.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import audit_ui  # noqa: E402

HOSTILE = audit_ui.HOSTILE

# One of each rung of engine.verify's severity ladder, plus the hostile string
# pushed through both text parameters the panel takes.
FINDINGS = [
    ("cross_domain_join", "error",
     f"hr_fact_employees is joined to {HOSTILE} on employee_id = account_id. "
     "These are independent datasets that share no identifiers."),
    ("join_fanout", "warn",
     "SUM(clinical_subjects.completed) is computed across a one-to-many join."),
    ("cross_domain_reference", "note",
     "this query reads 2 unrelated domains (hr, finance)."),
]

CHECKS = [("cross_domain_join", "error"), ("join_fanout", "warn"),
          ("cross_domain_reference", "note"), ("null_scalar", "warn")]

PICKS = [
    ("How many claims are in the dataset in total?", "healthcare",
     "SELECT COUNT(*) FROM healthcare_fact_claims", 0.5946),
    (HOSTILE, "aml", f"SELECT 1 -- {HOSTILE}", 0.4329),
]


@pytest.fixture(scope="module")
def ui():
    return audit_ui.load_ui()


def render(ui, fn) -> str:
    ui.st.take()
    fn()
    return ui.st.take()


@pytest.fixture(scope="module")
def panels(ui):
    """Every state each new panel can be in, captured separately."""
    return {
        "verification_clean": render(ui, lambda: ui.verification(
            [], checks=CHECKS, verify_ms=0.72)),
        "verification_sub_ms": render(ui, lambda: ui.verification(
            [], checks=CHECKS, verify_ms=0.004)),
        "verification_findings": render(ui, lambda: ui.verification(
            FINDINGS, checks=CHECKS, verify_ms=5.33)),
        "verification_refused": render(ui, lambda: ui.verification(
            FINDINGS[:1], checks=CHECKS, refused=True)),
        "verification_no_board": render(ui, lambda: ui.verification(
            FINDINGS[2:], verify_ms=0.02)),
        "exemplars_prompt": render(ui, lambda: ui.exemplars(
            PICKS, corpus=39, in_prompt=True)),
        "exemplars_demo": render(ui, lambda: ui.exemplars(
            PICKS, corpus=39, in_prompt=False, select_ms=319.0)),
    }


# --------------------------------------------------------------------------
# The same three passes scripts/audit_ui.py makes.
# --------------------------------------------------------------------------

def test_every_new_panel_produces_balanced_markup(panels):
    assert audit_ui.check_markup(panels) == []


def test_no_new_panel_lets_hostile_text_through(panels):
    assert audit_ui.check_escaping(panels) == []


def test_the_hostile_string_actually_reached_the_new_panels(panels):
    """Guards the guard: an escaping test passes trivially on an empty string."""
    escaped = sum("&lt;script&gt;" in body for body in panels.values())
    assert escaped >= 4, "the hostile fixture stopped reaching the new components"


def test_the_new_panels_use_no_colour_the_stylesheet_does_not_define(ui):
    """Re-run over the whole sheet, which now carries the new rules."""
    assert audit_ui.check_tokens(ui._CSS, audit_ui.token_table(ui._CSS)) == []


# Where each new text colour is really painted, in the same form
# audit_ui.contrast_rows uses: (label, fg, bg, is_small_text). Backgrounds are
# the tokens the panels actually set, with rgba fills flattened onto them.
def _new_contrast_rows(tokens):
    panel = tokens["--ayd-panel"]
    ink, muted = tokens["--ayd-ink"], tokens["--ayd-muted"]
    machine, alert = tokens["--ayd-machine"], tokens["--ayd-alert"]
    return [
        ("verifier head", muted, panel, True),
        ("verifier verdict clean", machine, panel, True),
        ("verifier verdict alert", alert, panel, True),
        ("verifier refusal line", alert, panel, True),
        ("severity error chip", alert, audit_ui.blend(alert, .10, panel), True),
        ("severity warn chip", alert, panel, True),
        ("severity note chip", muted, panel, True),
        ("verifier rule name", machine, panel, True),
        ("verifier message", ink, panel, True),
        ("rule board off", muted, panel, True),
        ("rule board hit blocking", alert, audit_ui.blend(alert, .10, panel), True),
        ("rule board hit note", ink, panel, True),
        ("exemplar head", muted, panel, True),
        ("exemplar question", ink, panel, True),
        ("exemplar meta", muted, panel, True),
        ("exemplar score", machine, panel, True),
        ("exemplar sql", ink, panel, True),
        ("exemplar foot", muted, panel, True),
    ]


def test_the_new_panels_small_text_clears_wcag_aa(ui):
    tokens = audit_ui.token_table(ui._CSS)
    failing = [
        f"{label} {audit_ui.contrast(fg, bg):.2f}:1"
        for label, fg, bg, small in _new_contrast_rows(tokens)
        if audit_ui.contrast(fg, bg) < (4.5 if small else 3.0)
    ]
    assert failing == []


def test_the_new_panels_do_not_lower_the_measured_floor(ui):
    """app/ui.py quotes 4.94:1 as the floor; nothing added here may undercut it."""
    tokens = audit_ui.token_table(ui._CSS)
    worst = min(audit_ui.contrast(fg, bg)
                for _label, fg, bg, _small in _new_contrast_rows(tokens))
    assert worst == pytest.approx(4.94, abs=0.01) or worst > 4.94


def test_nothing_new_is_amber(panels, ui):
    """Amber marks a value CI re-checks. Neither new panel shows one.

    The few-shot pairs do come out of a committed file, which is nearly the
    rule — and widening amber to "anything committed" would cost the badge on
    the answer the only thing it means. The selection is the machine's own work,
    so it is cyan.
    """
    tokens = audit_ui.token_table(ui._CSS)
    amber = tokens["--ayd-verified"]
    for name, body in panels.items():
        assert "ayd-verified" not in body, name
        assert amber.lower() not in body.lower(), name


def test_the_new_panels_get_tabular_figures(ui):
    """Forgetting this is the silent regression app/ui.py's stylesheet warns of."""
    rule = re.search(r"([^}]*?)\{\s*font-variant-numeric:tabular-nums",
                     ui._CSS, flags=re.S).group(1)
    assert ".ayd-ver" in rule and ".ayd-ex" in rule


# --------------------------------------------------------------------------
# The verifier panel's claims.
# --------------------------------------------------------------------------

def test_a_clean_verification_still_renders(panels):
    """The whole point: "it ran and found nothing" is not the same as silence."""
    body = panels["verification_clean"]
    assert 'data-worst="clean"' in body
    assert "no findings" in body
    assert body.count('class="ayd-ver-row"') == 0


def test_the_rule_board_is_the_evidence_for_the_clean_verdict(panels):
    body = panels["verification_clean"]
    assert body.count('class="ayd-rule"') == len(CHECKS)
    for name, _severity in CHECKS:
        assert f">{name}</span>" in body
    assert body.count('data-hit="0"') == len(CHECKS)


def test_only_the_rules_that_fired_light_on_the_board(panels):
    body = panels["verification_findings"]
    assert 'data-hit="error"' in body and 'data-hit="warn"' in body
    assert 'data-hit="note"' in body
    # null_scalar was in the roster and did not fire.
    assert body.count('data-hit="0"') == 1


def test_the_worst_severity_present_sets_the_panel_state(panels, ui):
    assert 'data-worst="error"' in panels["verification_findings"]
    assert 'data-worst="note"' in panels["verification_no_board"]
    warn_only = render(ui, lambda: ui.verification([FINDINGS[1]], checks=CHECKS))
    assert 'data-worst="warn"' in warn_only


def test_severity_is_told_apart_by_form_not_by_a_third_hue(panels, ui):
    """error is a filled chip, warn the same colour hollow, note bare.

    That is the fusion panel's filled/hollow tick rule applied to the severity
    ladder, and it is why the palette still has exactly two accents.
    """
    body = panels["verification_findings"]
    for severity in ("error", "warn", "note"):
        assert f'data-sev="{severity}"' in body
    css = ui._CSS
    error_rule = re.search(r'\.ayd-sev\[data-sev="error"\]\{([^}]*)\}', css).group(1)
    warn_rule = re.search(r'\.ayd-sev\[data-sev="warn"\]\{([^}]*)\}', css).group(1)
    assert "background:" in error_rule and "background:" not in warn_rule
    assert "var(--ayd-alert)" in error_rule and "var(--ayd-alert)" in warn_rule


def test_a_verifier_that_did_not_run_draws_nothing(ui):
    """A board of quiet rules claims they were checked. On a refusal with no SQL
    they were not, so there must be no panel at all rather than a clean one."""
    assert render(ui, lambda: ui.verification([], checks=CHECKS, ran=False)) == ""


def test_a_refusal_says_the_query_was_written_and_not_run(panels):
    body = panels["verification_refused"]
    assert "refused" in body
    assert "never executed" in body
    assert 'class="ayd-ver-refused"' in body


def test_a_verification_refusal_never_prints_the_models_correction_text(panels):
    """engine.verify.correction_message() closes with instructions addressed to
    the MODEL. The panel carries the findings, never that text."""
    body = panels["verification_refused"]
    for leaked in ("Write a corrected single SELECT query",
                   "call\ncannot_answer", "cannot_answer instead"):
        assert leaked not in body


def test_sub_millisecond_verification_is_reported_as_a_bound(panels):
    """"0.00ms" reads as "not measured", which is the one thing it is not."""
    assert "&lt;0.01ms" in panels["verification_sub_ms"]
    assert "0.72ms" in panels["verification_clean"]


def test_an_unmeasured_verification_invents_no_timing(panels):
    """Live mode's clock is inside assistant.ask; the panel does not fake one."""
    body = panels["verification_refused"]
    assert "ms<" not in body and "structural + result" in body


def test_the_rule_board_matches_the_module_that_owns_the_rules():
    """A hand-maintained board is a confident wrong answer the day a check lands.

    Reads every Finding engine/verify.py constructs and holds the app's roster
    against it. `ambiguous_entity` is the one deliberate omission: it exists on
    Verifier and engine/assistant.py never calls ambiguity_note(), so it cannot
    fire on any turn this app runs and drawing it would put a dead rule on a
    board whose whole job is to be evidence.
    """
    source = (ROOT / "engine" / "verify.py").read_text(encoding="utf-8")
    in_module = {
        (name, severity.lower())
        for name, severity in re.findall(
            r'Finding\(\s*\n?\s*"([a-z_]+)",\s*(ERROR|WARN|NOTE)', source)
    }
    assert in_module, "the Finding scan matched nothing; the pattern has rotted"

    namespace: dict = {}
    app_source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    start = app_source.index("VERIFY_CHECKS = [")
    exec(compile(app_source[start:app_source.index("]\n", start) + 1],
                 "streamlit_app_slice", "exec"), namespace)
    board = namespace["VERIFY_CHECKS"]

    # A rule may appear at two severities (empty_result is WARN or NOTE); the
    # board shows the blocking one, so compare on names and check the severity
    # the board claims is one the module really assigns.
    module_names = {name for name, _severity in in_module}
    board_names = {name for name, _severity in board}
    assert board_names == module_names - {"ambiguous_entity"}
    for name, severity in board:
        assert (name, severity) in in_module, f"{name} is not {severity} in verify.py"

    from engine.verify import Verifier

    assert hasattr(Verifier, "ambiguity_note")
    assistant_source = (ROOT / "engine" / "assistant.py").read_text(encoding="utf-8")
    assert "ambiguity_note" not in assistant_source, (
        "the assistant now calls ambiguity_note(); ambiguous_entity belongs on "
        "VERIFY_CHECKS in app/streamlit_app.py")


# --------------------------------------------------------------------------
# The VERIFY stage on the pipeline strip.
# --------------------------------------------------------------------------

def _cells(ui, **kwargs):
    body = render(ui, lambda: ui.pipeline(**kwargs))
    return re.findall(r'class="ayd-step" data-on="([^"]*)">([a-z ×0-9]+)', body)


def test_the_verify_stage_is_omitted_rather_than_dark_when_it_did_not_run(ui):
    """A dark cell claims a stage exists and did not run. On a path where the
    verifier is not wired in, the honest strip never mentions it."""
    labels = [label for _state, label in _cells(ui, retrieved=True, generated=True,
                                                guarded=True, executed=True)]
    assert "verify" not in labels
    assert labels == ["retrieve", "generate", "guard", "execute"]


def test_the_verify_stage_sits_where_the_blocking_half_runs(ui):
    """check_sql is structural and runs BEFORE run_query, which is what makes a
    refusal possible; check_result is advisory and cannot fail the turn."""
    labels = [label for _state, label in _cells(ui, retrieved=True, generated=True,
                                                verified=True, guarded=True,
                                                executed=True)]
    assert labels == ["retrieve", "generate", "verify", "guard", "execute"]


def test_a_verification_refusal_leaves_the_guard_and_execute_dark(ui):
    """Nothing crossed out of the verifier: run_query, and the guard inside it,
    genuinely never ran."""
    body = render(ui, lambda: ui.pipeline(retrieved=True, generated=True,
                                          verified="fail", guarded=False,
                                          executed=False))
    assert re.findall(r'class="ayd-arrow" data-on="([^"]*)"', body) == \
        ["1", "1", "fail", "0"]
    assert '<span class="ayd-step" data-on="fail">verify' in body


def test_the_verify_stage_reports_its_own_measured_cost(ui):
    body = render(ui, lambda: ui.pipeline(retrieved=True, generated=True,
                                          verified=True, guarded=True, executed=True,
                                          timings={"verify": 0.72}))
    assert "&lt;1ms" in body


# --------------------------------------------------------------------------
# The few-shot panel's claims.
# --------------------------------------------------------------------------

def test_the_demo_panel_never_claims_a_prompt_that_was_not_built(panels):
    """No model runs in demo mode, so nothing can be "in" its prompt."""
    body = panels["exemplars_demo"]
    assert "no prompt is built in demo mode" in body
    assert "system prompt" not in body


def test_the_live_panel_says_the_pairs_are_in_the_prompt(panels):
    assert "in this turn&#x27;s system prompt" in panels["exemplars_prompt"] \
        or "system prompt" in panels["exemplars_prompt"]


def test_the_panel_names_the_file_the_pairs_are_committed_in(panels):
    assert "evals/golden_questions.yaml" in panels["exemplars_demo"]


def test_leave_one_out_is_stated_where_a_reader_can_check_it(panels):
    assert "never shown its own pair" in panels["exemplars_demo"]


def test_the_reference_sql_is_shown_exactly(panels):
    body = panels["exemplars_demo"]
    assert "SELECT COUNT(*) FROM healthcare_fact_claims" in body


def test_an_empty_bank_draws_nothing(ui):
    """Chroma being unavailable must cost a panel, never an empty frame."""
    assert render(ui, lambda: ui.exemplars([], corpus=39, in_prompt=True)) == ""


def test_the_similarity_is_labelled_as_the_text_signal_not_the_ranking_key(panels):
    """The order is the fused one when retrieval had tables to fuse against, so
    the score is presented as `sim`, not as what produced the ranking."""
    body = panels["exemplars_demo"]
    assert "sim <b>0.59</b>" in body


# --------------------------------------------------------------------------
# The leave-one-out rule and the verifier, measured against the real corpus.
# --------------------------------------------------------------------------

def test_no_golden_question_is_ever_shown_its_own_pair():
    """The panel states this; this is what makes the statement true.

    39 of 39, over the same selector the app calls.
    """
    from engine import demo_mode
    from engine import exemplars as ex

    ex.build_index()
    for case in demo_mode.load_golden_questions():
        key = ex._normalise(case["question"])
        picks = ex.select_exemplars(case["question"], k=ex.DEFAULT_K)
        assert picks, case["id"]
        assert all(ex._normalise(p.question) != key for p in picks), case["id"]


def test_exemplar_selection_recovers_from_a_dead_collection():
    from engine import exemplars as ex

    ex.build_index()
    ex._client.delete_collection(ex._COLLECTION)
    picks = ex.select_exemplars("What is the overall claim denial rate?")
    assert picks
    assert ex._collection is not None


def test_the_verifier_is_silent_on_every_committed_query(con):
    """Demo mode runs the verifier on reference SQL, so a check that fired there
    would be a false positive in front of every visitor without an API key."""
    from engine import demo_mode
    from engine.query import run_query
    from engine.verify import Verifier

    verifier = Verifier(con)
    noisy = []
    for case in demo_mode.load_golden_questions():
        result = run_query(con, case["sql"])
        findings = verifier.check_sql(case["sql"])
        findings += verifier.check_result(case["sql"], result, case["question"])
        noisy += [f"{case['id']}: {f}" for f in findings]
    assert noisy == []
