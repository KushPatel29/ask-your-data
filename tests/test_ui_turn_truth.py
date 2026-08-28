"""
Does the instrument panel report the turn that actually happened?

Every other UI test asks whether a panel renders correctly given its arguments.
These ask the prior question: are the arguments true? Three readouts were
derived rather than observed, and all three were wrong on paths this suite can
reach without an API key.

  the guard ladder    app/streamlit_app.py._guard_readout listed three of the
                      four rungs engine.sql_guard.validate_sql actually walks,
                      so a query blocked by the FORBIDDEN_FUNCTIONS list fell
                      through to the fallback and drew `non-empty query ✕` on a
                      query that was not empty.
  the GUARD cell      was lit from the attempt count, so any retry claimed the
                      guard had rejected something. A turn that died three times
                      on a DuckDB binder error drew the identical strip to one
                      the guard really blocked.
  the EXECUTE cell    was lit unconditionally, including on a turn where no
                      query ever ran.

The loop states are produced by driving the real Assistant with a scripted
client — the same offline harness tests/test_assistant_harness.py uses — so the
AskResult under test is one engine/assistant.py genuinely returns, not a
hand-built stand-in that could agree with the UI by construction. The render
code is exec'd out of app/streamlit_app.py's own source for the same reason: a
reimplementation here could pass against logic the app does not run.
"""

import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import audit_ui  # noqa: E402

from engine.assistant import MAX_ATTEMPTS, Assistant  # noqa: E402
from engine.sql_guard import FORBIDDEN, FORBIDDEN_FUNCTIONS, validate_sql  # noqa: E402
from engine.warehouse import schema_catalog  # noqa: E402

APP_SOURCE = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Running the app's own code without running Streamlit.
# --------------------------------------------------------------------------

def app_slice(start: str, end: str) -> str:
    """The source between two literal markers, dedented so it can be exec'd.

    app/streamlit_app.py calls st.set_page_config at import time, so importing
    it is not available here. Slicing is what scripts/audit_ui.py already does
    to reach `_plan_rows`, and it has the property that matters: the code under
    test is the code that ships, and a rename breaks the test loudly instead of
    letting it pass against a stale copy.
    """
    found = APP_SOURCE.index(start)
    # Back up to the start of the line. A marker taken mid-line leaves the first
    # line of the slice unindented while the rest keep their indent, and dedent
    # then finds a common prefix of nothing.
    head = APP_SOURCE.rfind("\n", 0, found) + 1
    return textwrap.dedent(APP_SOURCE[head:APP_SOURCE.index(end, found)])


@pytest.fixture(scope="module")
def ui():
    return audit_ui.load_ui()


@pytest.fixture(scope="module")
def guard_readout(ui):
    """app/streamlit_app.py._guard_readout, wired to the real guard."""
    namespace = {
        "validate_sql": validate_sql,
        "FORBIDDEN": FORBIDDEN,
        "FORBIDDEN_FUNCTIONS": FORBIDDEN_FUNCTIONS,
        "ui": ui,
    }
    exec(compile(app_slice("def _guard_readout(", "\n\n# The prefix engine.query"),
                 "streamlit_app_slice", "exec"), namespace)
    return namespace["_guard_readout"]


@pytest.fixture(scope="module")
def strip_for(ui):
    """render_entry's pipeline block, exec'd against one transcript entry.

    Returns a callable taking the entry dict and giving back the strip's cells
    as [(state, label), ...], which is what the assertions are about.
    """
    body = app_slice('ran = bool(entry.get("ran"))', "_show_grounding(bundle)")
    code = compile(body, "streamlit_app_slice", "exec")
    prefix = re.search(r'^GUARD_BLOCK_PREFIX = "([^"]+)"', APP_SOURCE, re.M).group(1)

    def run(entry):
        ui.st.take()
        exec(code, {"ui": ui, "GUARD_BLOCK_PREFIX": prefix},
             {"entry": entry, "bundle": None})
        return re.findall(r'class="ayd-step" data-on="([^"]*)">([a-z ×0-9]+)',
                          ui.st.take())

    return run


def cell(cells, name):
    for state, label in cells:
        if label.strip() == name:
            return state
    raise AssertionError(f"no {name!r} cell in {cells}")


# --------------------------------------------------------------------------
# The guard ladder: every refusal validate_sql can make.
# --------------------------------------------------------------------------

# One query per return path in engine.sql_guard.validate_sql, with the rung the
# panel must draw as failed. If a path is added to that function without a rung
# here, `test_no_guard_refusal_falls_through_the_ladder` is what notices.
REFUSALS = [
    ("   ", "non-empty query"),
    ("SELECT 1; DROP TABLE t", "single statement"),
    ("EXPLAIN SELECT 1", "starts SELECT / WITH"),
    ("SELECT * FROM t WHERE x IN (DELETE)", f"none of {len(FORBIDDEN)} forbidden verbs"),
    ("SELECT * FROM read_csv_auto('/etc/passwd')",
     f"none of {len(FORBIDDEN_FUNCTIONS)} forbidden functions"),
]


def checks_in(markup):
    """(label, passed) for every check chip the guard panel drew."""
    return [(re.sub(r"&#\w+;", "", label), state == "1")
            for state, label in re.findall(
                r'class="ayd-check" data-pass="(\d)">([^<]*)<', markup)]


@pytest.mark.parametrize("sql,rung", REFUSALS)
def test_the_panel_names_the_boundary_the_guard_actually_crossed(
        ui, guard_readout, sql, rung):
    ui.st.take()
    guard_readout(sql)
    checks = checks_in(ui.st.take())
    failed = [label for label, passed in checks if not passed]
    assert failed == [rung], f"{sql!r} drew {checks}"


def test_a_forbidden_function_is_not_reported_as_an_empty_query(ui, guard_readout):
    """The regression this file was opened for.

    sql_guard enforces two lists. Only the verbs were on the ladder, so a query
    blocked by the second one matched no rung, fell into the fallback branch and
    was drawn as `non-empty query ✕` — the panel naming a boundary the guard had
    not touched, on the one path where naming the right one matters most.
    """
    ok, reason = validate_sql("SELECT * FROM read_csv_auto('/etc/passwd')")
    assert not ok and reason.startswith("forbidden function:")
    ui.st.take()
    guard_readout("SELECT * FROM read_csv_auto('/etc/passwd')")
    markup = ui.st.take()
    assert "non-empty query" not in markup
    assert "forbidden functions" in markup


def test_no_guard_refusal_falls_through_the_ladder(ui, guard_readout):
    """Every `return False` in validate_sql is covered by a case above.

    A rung that goes missing does not fail loudly — it silently redraws as the
    fallback — so the count is held against the module itself.
    """
    source = (ROOT / "engine" / "sql_guard.py").read_text(encoding="utf-8")
    body = source[source.index("def validate_sql("):]
    assert body.count("return False,") == len(REFUSALS)


def test_a_passing_query_shows_every_rung_cleared(ui, guard_readout):
    ui.st.take()
    guard_readout("SELECT 1 AS n")
    checks = checks_in(ui.st.take())
    assert [passed for _label, passed in checks] == [True] * 4
    assert any("forbidden functions" in label for label, _ in checks)


def test_the_rail_states_both_of_the_guards_lists():
    """The rail claimed only the verbs, which understated the boundary."""
    rail = app_slice("def _status_rail(", "_status_rail()\n")
    assert "len(FORBIDDEN)" in rail and "len(FORBIDDEN_FUNCTIONS)" in rail


# --------------------------------------------------------------------------
# The pipeline strip, over loop outcomes the real assistant produces.
# --------------------------------------------------------------------------

GOOD_SQL = "SELECT COUNT(*) AS n FROM healthcare_fact_claims"
BAD_SQL = "SELECT no_such_column FROM healthcare_fact_claims"
EVIL_SQL = "DROP TABLE healthcare_fact_claims"


class FakeClient:
    """The scripted stand-in from tests/test_assistant_harness.py."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        if not self._responses:
            raise AssertionError("assistant made more API calls than scripted")
        return self._responses.pop(0)


def tool_use(name, **inp):
    return SimpleNamespace(type="tool_use", name=name, input=inp)


def msg(*blocks):
    return SimpleNamespace(content=list(blocks), usage=None)


def summary(text):
    return msg(SimpleNamespace(type="text", text=text))


def ask(con, responses):
    return Assistant(
        con, client=FakeClient(responses),
        catalog_builder=lambda _q, con, **_k: schema_catalog(con),
    ).ask("how many claims?")


def entry_for(result):
    """The transcript entry app/streamlit_app.py builds, for the keys the strip reads."""
    return {
        "refused": result.refused,
        "answer": result.answer,
        "attempts": result.attempts,
        "corrections": result.corrections,
        "ran": bool(result.ok),
        "rows": ([1] if (result.result and result.result.ok and result.result.rows)
                 else None),
        "error": result.result.error if (result.result and not result.result.ok) else "",
    }


@pytest.fixture(scope="module")
def exhausted(con):
    """Three attempts, three DuckDB binder errors, no query ever ran."""
    return ask(con, [msg(tool_use("answer_with_sql", sql=BAD_SQL, explanation=""))
                     for _ in range(MAX_ATTEMPTS)])


@pytest.fixture(scope="module")
def guard_blocked(con):
    """The guard rejects attempt 1; the model corrects and the second one runs."""
    return ask(con, [msg(tool_use("answer_with_sql", sql=EVIL_SQL, explanation="")),
                     msg(tool_use("answer_with_sql", sql=GOOD_SQL, explanation="")),
                     summary("There are 12,000 claims.")])


@pytest.fixture(scope="module")
def clean(con):
    return ask(con, [msg(tool_use("answer_with_sql", sql=GOOD_SQL, explanation="")),
                     summary("There are 12,000 claims.")])


def test_the_exhausted_loop_really_never_executed_anything(exhausted):
    """The premise of the two tests below, asserted rather than assumed."""
    assert not exhausted.ok and exhausted.attempts == MAX_ATTEMPTS
    assert len(exhausted.corrections) == MAX_ATTEMPTS
    assert all(c.startswith("Binder Error") for c in exhausted.corrections)


def test_execute_is_not_lit_when_no_query_came_back(strip_for, exhausted):
    assert cell(strip_for(entry_for(exhausted)), "execute") == "fail"


def test_a_binder_error_is_not_reported_as_a_guard_rejection(strip_for, exhausted):
    """The guard passed all three attempts; the warehouse is what refused."""
    assert cell(strip_for(entry_for(exhausted)), "guard") == "1"


def test_a_real_guard_block_still_lights_the_guard_cell(strip_for, guard_blocked):
    assert guard_blocked.ok and guard_blocked.attempts == 2
    assert guard_blocked.corrections[0].startswith("blocked by SQL guard")
    cells = strip_for(entry_for(guard_blocked))
    assert cell(cells, "guard") == "fail"
    assert cell(cells, "execute") == "1"


def test_the_two_failure_modes_no_longer_draw_the_same_strip(
        strip_for, exhausted, guard_blocked):
    """Both were `guard=fail, execute=1` before, which is the whole complaint."""
    assert strip_for(entry_for(exhausted)) != strip_for(entry_for(guard_blocked))


def test_a_clean_turn_lights_every_stage(strip_for, clean):
    cells = strip_for(entry_for(clean))
    assert [state for state, _label in cells] == ["1"] * 5
    assert "retry" not in " ".join(label for _s, label in cells)


def test_a_turn_that_stopped_at_the_guard_leaves_execute_dark(strip_for):
    """Dark, not failed: nothing crossed into the executor to fail there."""
    entry = {"attempts": MAX_ATTEMPTS, "ran": False, "answer": "",
             "corrections": ["blocked by SQL guard: forbidden keyword: DROP"] * MAX_ATTEMPTS}
    cells = strip_for(entry)
    assert cell(cells, "guard") == "fail"
    assert cell(cells, "execute") == "0"


def test_the_guard_marker_matches_the_module_that_writes_it(strip_for):
    """The prefix is re-typed in the app, so it is held against engine/query.py.

    engine.query builds the string inline, so there is nothing to import; a
    reworded message would otherwise turn every guard block into a silent
    binder error on the strip.
    """
    prefix = re.search(r'^GUARD_BLOCK_PREFIX = "([^"]+)"', APP_SOURCE, re.M).group(1)
    query_source = (ROOT / "engine" / "query.py").read_text(encoding="utf-8")
    assert f'error=f"{prefix}' in query_source


def test_an_answerless_turn_draws_no_empty_headline():
    """ui.answer("") emitted a 1.75rem hole where the product goes."""
    branch = app_slice('if entry["answer"]:', "if entry.get(\"elapsed_ms\"):")
    assert "st.error(" in branch


# --------------------------------------------------------------------------
# The fusion funnel.
# --------------------------------------------------------------------------

class Hit:
    def __init__(self, table, domain, score):
        self.table, self.domain, self.score = table, domain, score


HITS = [Hit("retail_customer_analytics", "retail", 0.0328),
        Hit("hr_fact_employees", "hr", 0.0161)]
RANKS = ({"retail_customer_analytics": 17, "hr_fact_employees": 4},
         {"retail_customer_analytics": 3, "hr_fact_employees": 9})


def ground(ui, **kwargs):
    ui.st.take()
    ui.grounding(HITS, total_tables=71, tokens_used=3241, tokens_full=12741,
                 vector_ranks=RANKS[0], keyword_ranks=RANKS[1], pool=20, **kwargs)
    return ui.st.take()


def test_the_funnels_middle_stage_is_reported_when_it_was_measured(ui):
    markup = ground(ui, candidates=28)
    assert "<b>28</b>" in markup and "20 deep" in markup


def test_no_candidate_count_is_invented_when_the_caller_has_none(ui):
    """The panel degrades to what it can prove, the way the ranks already do."""
    assert "distinct tables" not in ground(ui)


def test_the_funnel_adds_no_rows_to_the_ranking(ui):
    """Restraint, checked: the claim is one clause in the foot, not a table."""
    with_it, without = ground(ui, candidates=28), ground(ui)
    assert with_it.count('class="ayd-row"') == without.count('class="ayd-row"')


def test_the_candidate_count_is_what_the_bundle_actually_measures():
    """The union of the two rankings, not a constant and not the pool depth."""
    bundle = app_slice("def _retrieval_bundle(", "def _show_grounding(")
    assert '"candidates": len(set(vector) | set(keyword))' in bundle


def test_the_funnel_reaches_the_panel(ui):
    show = app_slice("def _show_grounding(", "def _guard_readout(")
    assert 'candidates=bundle.get("candidates")' in show


# --------------------------------------------------------------------------
# Where the exemplar panel sits.
# --------------------------------------------------------------------------

def test_provenance_reads_after_the_answer_it_vouches_for():
    """Three solved questions with their full SQL used to stand between the
    grounding panel and the number. Both modes now render it below the answer,
    with the guard and the verifier."""
    demo = APP_SOURCE.index("def render_demo_mode(")
    live = APP_SOURCE.index("def render_entry(")
    for start, end in ((demo, live), (live, len(APP_SOURCE))):
        body = APP_SOURCE[start:end]
        answer_at = body.index("ui.answer(")
        bank_at = min(i for i in (body.find("_show_exemplars("), body.find("ui.exemplars("))
                      if i > 0)
        assert bank_at > answer_at, "the few-shot bank is drawn above the answer"
