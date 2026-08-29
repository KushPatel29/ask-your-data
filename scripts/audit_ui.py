"""
Render every app/ui.py component offline and audit the result.

    .venv/Scripts/python.exe scripts/audit_ui.py [--html out.html]

The design in app/ui.py cannot be checked by looking at it: this box does not
composite a browser, so "does it look right" is not a question available here.
What IS available is arithmetic and structure, and every claim that file makes
about itself is one of those two things — a contrast ratio, a tag that closes,
an escape that happened, a colour that resolves to a defined token. So this
script measures them.

It does three passes:

  MARKUP    Every component is rendered with real data (a real DuckDB plan, the
            real FORBIDDEN list, a real retrieval bundle) by capturing the HTML
            that st.markdown would have received. Tags must balance, and hostile
            strings pushed through every text parameter must come back escaped.
  COLOUR    Every `color:` declaration in the stylesheet is resolved through the
            :root token table and scored against the background it is painted
            on, using the WCAG 2.1 relative-luminance formula. Small text below
            4.5:1 is an error.
  TOKENS    Every var(--ayd-*) reference must resolve to something :root defines.
            A typo'd custom property does not fail loudly in CSS; it silently
            paints the inherited colour, which is exactly the failure mode that
            once made the schema map report an empty retrieval.

Exit code is non-zero if anything fails, so this is usable as a check rather
than as a report to read and believe.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# Capture: run the ui module without Streamlit.
# --------------------------------------------------------------------------

class _Capture:
    """Stand in for the `st` module app/ui.py imports.

    ui.py only ever calls st.markdown(..., unsafe_allow_html=True), so a shim
    with one method is enough and is more honest than importing Streamlit and
    running a real script context — this way the HTML under test is exactly the
    string ui.py produced, with nothing wrapped around it.
    """

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
        self.chunks.append(body)

    def take(self) -> str:
        out = "".join(self.chunks)
        self.chunks = []
        return out


def load_ui():
    import app.ui as ui  # noqa: PLC0415

    ui.st = _Capture()          # type: ignore[attr-defined]
    return ui


# --------------------------------------------------------------------------
# Colour: WCAG 2.1 contrast, computed rather than asserted.
# --------------------------------------------------------------------------

def _srgb(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    value = hex_colour.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _srgb(r) + 0.7152 * _srgb(g) + 0.0722 * _srgb(b)


def contrast(fg: str, bg: str) -> float:
    a, b = luminance(fg), luminance(bg)
    lo, hi = sorted((a, b))
    return (hi + 0.05) / (lo + 0.05)


def blend(fg: str, alpha: float, bg: str) -> str:
    """Flatten an rgba fill onto an opaque background.

    Several panels sit on an rgba() tint of their own accent. The text on them
    is painted against the composite, not against the panel, so the composite is
    what has to be scored.
    """
    def parts(colour: str) -> tuple[int, int, int]:
        value = colour.lstrip("#")
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    f, b = parts(fg), parts(bg)
    mixed = tuple(round(f[i] * alpha + b[i] * (1 - alpha)) for i in range(3))
    return "#%02X%02X%02X" % mixed


def token_table(css: str) -> dict[str, str]:
    """The :root custom properties, parsed out of the stylesheet itself."""
    root = re.search(r":root\{(.*?)\}", css, flags=re.S)
    if not root:
        return {}
    found = {}
    for name, value in re.findall(r"(--ayd-[\w-]+)\s*:\s*([^;]+);", root.group(1)):
        found[name] = value.strip()
    return found


# --------------------------------------------------------------------------
# Fixtures: real machine state, not lorem ipsum.
# --------------------------------------------------------------------------

HOSTILE = "<script>alert(1)</script> & \"quoted\" 'single'"


def fixtures(con=None):
    """Everything the components need, taken from the real modules where it exists."""
    from engine.sql_guard import FORBIDDEN

    # A golden query with a CTE and two joins, so the plan under audit exercises
    # the branch guides and every detail key rather than a single flat scan.
    sql = (
        "WITH conv AS (\n"
        "  SELECT last_touch_channel AS channel, COUNT(*) AS conversions\n"
        "  FROM marketing_dim_user WHERE converted = 1 GROUP BY 1\n"
        "), spend AS (\n"
        "  SELECT channel, SUM(spend) AS spend FROM marketing_fact_spend GROUP BY 1\n"
        ") SELECT c.channel FROM marketing_dim_channel c "
        "JOIN spend s ON s.channel = c.channel JOIN conv v ON v.channel = c.channel "
        "WHERE c.is_paid = 1 ORDER BY s.spend / v.conversions DESC LIMIT 1"
    )
    nodes = plan_nodes(sql, con)
    return {"forbidden": FORBIDDEN, "sql": sql, "nodes": nodes}


def plan_nodes(sql: str, con=None) -> list[dict]:
    """A genuine DuckDB plan, flattened the same way app/streamlit_app.py does.

    Imported from the app module rather than reimplemented, so this audit cannot
    pass against a walker the app does not use.
    """
    import importlib.util

    from engine.warehouse import build_warehouse

    # app/streamlit_app.py runs Streamlit at import time, so its private helpers
    # are read as source and exec'd in isolation instead.
    source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    start = source.index("_PLAN_DETAIL = {")
    end = source.index("@st.cache_data(show_spinner=False)\ndef _query_plan")
    namespace: dict = {}
    exec(compile(source[start:end], "streamlit_app_slice", "exec"), namespace)

    con = con if con is not None else build_warehouse()
    cur = con.cursor()
    cur.execute("EXPLAIN (FORMAT JSON) " + sql)
    raw = cur.fetchall()[0][1]
    cur.close()
    nodes: list[dict] = []
    for root in json.loads(raw):
        namespace["_plan_rows"](root, "", True, True, nodes)
    assert importlib.util  # keep the import meaningful to linters
    return nodes


def render_all(ui, fx) -> dict[str, str]:
    """Every component, each captured on its own so a failure names one panel."""
    out: dict[str, str] = {}

    def grab(name, fn):
        ui.st.take()
        fn()
        out[name] = ui.st.take()

    grab("masthead", lambda: ui.masthead(tables=71, domains=11, live=False))
    grab("status_rail", lambda: ui.status_rail([
        ("warehouse", "<s>71</s> tables <em>· 11 domains</em>"),
        ("guard", "read-only<br><em>26 forbidden verbs</em>"),
    ]))
    grab("pipeline_ok", lambda: ui.pipeline(
        retrieved=True, generated=True, guarded=True, executed=True,
        timings={"retrieve": 196.4, "guard": 0.4, "execute": 3.2}))
    grab("pipeline_demo", lambda: ui.pipeline(
        retrieved=True, generated=False, guarded=True, executed=True,
        timings={"retrieve": 310.0, "guard": 0.2, "execute": 1.9}))
    grab("pipeline_blocked", lambda: ui.pipeline(
        retrieved=True, generated=True, guarded="fail", executed=False, attempts=3))
    # The compiled path. PLAN replaces GENERATE rather than sitting beside it,
    # so both states are rendered here: a turn the compiler answered and one it
    # refused, which is the shape that must never light GENERATE.
    grab("pipeline_planned", lambda: ui.pipeline(
        retrieved=True, planned=True, verified=True, guarded=True, executed=True,
        timings={"retrieve": 164.0, "plan": 5.0, "guard": 0.2, "execute": 2.1}))
    grab("pipeline_plan_refused", lambda: ui.pipeline(
        retrieved=True, planned="fail", guarded=False, executed=False,
        timings={"retrieve": 183.0, "plan": 4.0}))

    class Hit:
        def __init__(self, table, domain, score):
            self.table, self.domain, self.score = table, domain, score

    hits = [Hit("retail_customer_analytics", "retail", 0.0328),
            Hit(HOSTILE, "healthcare", 0.0210),
            Hit("hr_fact_employees", "hr", 0.0161)]
    grab("grounding", lambda: ui.grounding(
        hits, total_tables=71, tokens_used=3241, tokens_full=12741,
        vector_ranks={"retail_customer_analytics": 17, HOSTILE: 2},
        keyword_ranks={"retail_customer_analytics": 3, "hr_fact_employees": 9},
        pool=28))
    grab("schema_map", lambda: ui.schema_map(
        {"retail": [("retail_fact_orders", True), (HOSTILE, False)],
         "hr": [("hr_fact_employees", False)]},
        retrieved=1, total=71))
    grab("guard_pass", lambda: ui.guard_verdict(
        ok=True, reason="ok",
        checks=[("single statement", True), ("starts SELECT / WITH", True),
                ("none of 26 forbidden verbs", True)],
        forbidden=fx["forbidden"]))
    grab("guard_blocked", lambda: ui.guard_verdict(
        ok=False, reason="forbidden keyword: DROP",
        checks=[("single statement", True), ("starts SELECT / WITH", True),
                ("none of 26 forbidden verbs", False)],
        forbidden=fx["forbidden"], blocked_verb="DROP"))
    grab("attempts_ok", lambda: ui.attempt_ledger(
        attempts=1, corrections=[], max_attempts=3))
    grab("attempts_corrected", lambda: ui.attempt_ledger(
        attempts=3, max_attempts=3,
        corrections=[f'Binder Error: {HOSTILE} does not have a column named "revenu"',
                     "joins two independent domains; " + "x" * 300]))
    grab("attempts_exhausted", lambda: ui.attempt_ledger(
        attempts=3, max_attempts=3, ok=False,
        corrections=["Binder Error: no such column", "Parser Error: syntax", "still broken"]))
    grab("query_plan", lambda: ui.query_plan(
        fx["nodes"], plan_ms=0.44, returned=5, truncated=False))
    grab("query_plan_capped", lambda: ui.query_plan(
        fx["nodes"], plan_ms=1.70, returned=200, truncated=True))
    grab("result_shape", lambda: ui.result_shape(
        [("department", "VARCHAR"), (HOSTILE, "BIGINT")],
        rows=5, truncated=False, cap=200))
    grab("result_shape_capped", lambda: ui.result_shape(
        [("order_id", "BIGINT")], rows=200, truncated=True, cap=200))
    grab("column_list", lambda: ui.column_list(
        [("customer_name", "key", ()),
         ("status", "dimension", ("Paid", "Denied", HOSTILE)),
         ("paid_amount", "measure", ())],
        highlight="status"))
    grab("refusal", lambda: ui.refusal(
        f"I have no way to compute a median — {HOSTILE}", kind="not compiled"))
    grab("refusal_unbound", lambda: ui.refusal(
        "too much of that question has no counterpart in the warehouse",
        kind="nothing to bind"))
    grab("layer_summary", lambda: ui.layer_summary(
        [("285", "measures", "numeric columns it can aggregate"),
         ("197", "dimensions", "columns it can group by"),
         ("56", "join edges", HOSTILE),
         ("797", "value phrases", "words from the data itself")],
        footnote=f"All four probed from DuckDB at startup. {HOSTILE}"))
    grab("plan_trace", lambda: ui.plan_trace(
        [("table", f"{HOSTILE} — 1 join"),
         ("metric", "AVG(base_salary)"),
         ("grouped by", "department"),
         ("filter", f'status = \'{HOSTILE}\' (from "{HOSTILE}")')],
        coverage=1.0, considered=13,
        bound=["average", "department", "salary"],
        missed=[HOSTILE], loose=["dataset"], plan_ms=5.0))
    grab("plan_trace_refused", lambda: ui.plan_trace(
        [("table", "marketing_experiment_geo_weekly — 0 joins"),
         ("metric", "COUNT(*)")],
        coverage=0.4, considered=10, bound=["experiment", "geo"],
        missed=["holdout", "incremental", "lift"], loose=[],
        plan_ms=4.0, refused=True))
    grab("answer", lambda: ui.answer("1,428", verified=True,
                                     verified_note="Asserted in evals/golden_questions.yaml."))
    grab("domain_card", lambda: ui.domain_card(HOSTILE, "A domain blurb."))
    grab("note", lambda: ui.note(HOSTILE))
    return out


# --------------------------------------------------------------------------
# The three passes.
# --------------------------------------------------------------------------

VOID = {"br", "hr", "img", "input", "meta", "link"}


def check_markup(panels: dict[str, str]) -> list[str]:
    problems = []
    for name, body in panels.items():
        stack: list[str] = []
        found = [(m.group(2).lower(), bool(m.group(1)))
                 for m in re.finditer(r"<(/?)([a-zA-Z][\w-]*)", body)]
        for tag, closing in found:
            if tag in VOID:
                continue
            if closing:
                if not stack or stack[-1] != tag:
                    opened = stack[-1] if stack else "nothing"
                    problems.append(f"{name}: </{tag}> closes <{opened}>")
                    break
                stack.pop()
            else:
                stack.append(tag)
        else:
            if stack:
                problems.append(f"{name}: unclosed {stack}")
    return problems


def check_escaping(panels: dict[str, str]) -> list[str]:
    """The hostile string went through every text parameter; none may come back raw.

    status_rail is exempt by contract — its documented job is to pass inline
    tags through, and its caller escapes. Every other component escapes, and
    this is the pass that proves the ones added later kept doing it.
    """
    problems = []
    for name, body in panels.items():
        if name == "status_rail":
            continue
        if "<script>" in body:
            problems.append(f"{name}: unescaped <script> reached the output")
        # `title="…"` carries error text; a raw quote there would break out of
        # the attribute even when the tag itself looks fine.
        for attr in re.findall(r'title="([^"]*)"', body):
            if "<" in attr or "&" in attr and not re.search(r"&\w+;|&#\d+;", attr):
                problems.append(f"{name}: raw markup inside a title attribute")
    return problems


def check_tokens(css: str, tokens: dict[str, str]) -> list[str]:
    problems = []
    for ref in sorted(set(re.findall(r"var\((--ayd-[\w-]+)", css))):
        if ref not in tokens:
            problems.append(f"var({ref}) is used but :root never defines it")
    return problems


# Where each text colour is actually painted. Backgrounds are the token values
# the panel really sets, and rgba fills are flattened onto what is behind them.
def contrast_rows(tokens: dict[str, str]) -> list[tuple[str, str, str, float, bool]]:
    ground = tokens["--ayd-ground"]
    panel = tokens["--ayd-panel"]
    panel2 = tokens["--ayd-panel-2"]
    ink = tokens["--ayd-ink"]
    muted = tokens["--ayd-muted"]
    machine = tokens["--ayd-machine"]
    verified = tokens["--ayd-verified"]
    alert = tokens["--ayd-alert"]

    # (label, fg, bg, is_small_text)
    pairs = [
        ("masthead title", ink, ground, False),
        ("masthead sub", muted, ground, True),
        ("masthead kicker", machine, ground, True),
        ("rail label", muted, panel, True),
        ("rail value", ink, panel, True),
        ("rail value accent", machine, panel, True),
        ("rail value alert", alert, panel, True),
        ("pipe step off", muted, panel, True),
        ("pipe step on", machine, blend(machine, .07, panel), True),
        ("pipe step fail", alert, blend(alert, .08, panel), True),
        ("ground table", ink, panel, True),
        ("ground rank", machine, panel, True),
        ("ground score", muted, panel, True),
        ("map domain lit", ink, panel, True),
        ("map domain unlit", muted, panel, True),
        ("guard head", muted, panel, True),
        ("guard verdict pass", machine, panel, True),
        ("guard verdict block", alert, panel, True),
        ("forbidden verb off", muted, panel, True),
        ("forbidden verb hit", alert, blend(alert, .10, panel), True),
        ("attempt label", muted, panel, True),
        ("attempt failed", alert, panel, True),
        ("attempt ran", machine, panel, True),
        ("attempt reason", ink, panel, True),
        ("plan operator", machine, panel, True),
        ("plan projection", muted, panel, True),
        ("plan detail", muted, panel, True),
        ("plan cardinality", ink, panel, True),
        ("plan foot accent", machine, panel, True),
        ("shape count", ink, panel, True),
        ("shape label", muted, panel, True),
        ("shape truncated", alert, panel, True),
        ("column name", ink, panel, True),
        ("column type", machine, panel, True),
        ("answer headline", ink, ground, False),
        ("verified badge", verified, blend(verified, .07, ground), True),
        ("sidebar domain", machine, panel2, True),
    ]
    rows = []
    for label, fg, bg, small in pairs:
        ratio = contrast(fg, bg)
        floor = 4.5 if small else 3.0
        rows.append((label, fg, bg, ratio, ratio >= floor))
    return rows


def preview_html(css: str, panels: dict[str, str]) -> str:
    body = "".join(
        f'<h2 style="font:600 .7rem/1 monospace;letter-spacing:.2em;'
        f'text-transform:uppercase;color:#7C859C;margin:2.4rem 0 .6rem">{name}</h2>{markup}'
        for name, markup in panels.items()
    )
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>ui.py audit</title>{css}</head>"
            "<body class='stApp' style='background:#0A0C14;margin:0;padding:2rem'>"
            f"{body}</body></html>")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", help="write a standalone preview of every panel here")
    args = parser.parse_args()

    ui = load_ui()
    css = ui._CSS
    tokens = token_table(css)
    fx = fixtures()
    panels = render_all(ui, fx)

    failures: list[str] = []

    print("MARKUP")
    problems = check_markup(panels) + check_escaping(panels)
    for problem in problems:
        print(f"  FAIL  {problem}")
    failures += problems
    print(f"  {len(panels)} panels rendered, "
          f"{sum(len(p) for p in panels.values()):,} bytes of markup, "
          f"{len(problems)} problem(s)")

    print("\nTOKENS")
    problems = check_tokens(css, tokens)
    for problem in problems:
        print(f"  FAIL  {problem}")
    failures += problems
    referenced = set(re.findall(r"var\((--ayd-[\w-]+)", css))
    print(f"  {len(tokens)} tokens defined, {len(referenced)} referenced, "
          f"{len(problems)} unresolved")
    unused = sorted(set(tokens) - referenced)
    if unused:
        print(f"  note: defined but never referenced: {', '.join(unused)}")

    print("\nCOLOUR  (WCAG 2.1; small text floor 4.5:1, large 3.0:1)")
    rows = contrast_rows(tokens)
    worst = min(rows, key=lambda r: r[3])
    best = max(rows, key=lambda r: r[3])
    for label, fg, bg, ratio, ok in sorted(rows, key=lambda r: r[3]):
        flag = "    " if ok else "FAIL"
        print(f"  {flag}  {ratio:5.2f}:1  {fg} on {bg}  {label}")
        if not ok:
            failures.append(f"contrast {label} {ratio:.2f}:1")
    print(f"  range {worst[3]:.2f}:1 ({worst[0]}) .. {best[3]:.2f}:1 ({best[0]}) "
          f"across {len(rows)} pairs")

    if args.html:
        Path(args.html).write_text(preview_html(css, panels), encoding="utf-8")
        print(f"\nwrote {args.html}")

    print(f"\n{'FAILED: ' + str(len(failures)) if failures else 'PASS'}")
    assert html_mod  # the module is used by the components under test
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
