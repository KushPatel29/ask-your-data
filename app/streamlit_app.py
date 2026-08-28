"""
Ask-Your-Data chat UI.

    streamlit run app/streamlit_app.py

A real conversation: follow-up questions ("and by region?") carry the earlier
turns as context. Every answer shows the plain-English result, the SQL the model
wrote, and the returned rows — so a reader can always check the number against
the query.

With ANTHROPIC_API_KEY set, that is what runs. Without one there is no model, so
the app falls back to DEMO MODE: the questions from the project's accuracy
contract, each executing its reference SQL live against DuckDB. That is a
genuinely different thing from the model writing SQL, and the UI says so rather
than blurring the two.

WHAT THIS FILE PUTS ON SCREEN, AND WHY IT IS CHEAP
The readouts in app/ui.py need more than the fused ranking: the fusion panel
draws the vector and keyword ranks that RRF consumed, and the schema map needs
the whole catalogue. Fetching that per render would be ruinous — the retrieval
work behind one transcript entry measured 310 ms as this file was originally
written, re-paid for every entry on every Streamlit rerun (a five-turn
conversation spent 1.55 s re-retrieving what it had already retrieved, purely to
redraw it). So retrieval is gathered once per question into `_retrieval_bundle`
and cached on the question text. The bundle also carries the measured cost of
the hybrid call, so the timing shown next to the RETRIEVE stage is the time
retrieval actually took when it ran, not the near-zero cost of a cache hit.

Two more readouts ask DuckDB about a statement without running it — EXPLAIN
(FORMAT JSON) for the physical plan and DESCRIBE for the output schema — and
they are cached the same way and for the same reason, even though both are
cheap (measured across the 39 golden queries: EXPLAIN 0.20-1.70 ms, DESCRIBE
0.18-0.61 ms). Neither executes the query, and neither runs on SQL the guard
has not already passed, so showing the plan cannot become a second execution
path around the safety boundary.

WHO COMPUTES WHAT, AND WHY IT IS NOT SYMMETRIC
The verifier and the few-shot bank are each rendered from two different sources
depending on the mode, and the asymmetry is the honest part rather than an
oversight.

  verifier   LIVE mode reads the findings engine.assistant already computed and
             put on AskResult. It does NOT re-run the checks: that would be a
             second verifier which could disagree with the one whose verdict
             actually gated execution, the same trap `_guard_readout` avoids by
             never re-scanning the SQL itself. DEMO mode has no assistant, so it
             runs them here — the checks are deterministic and key-free, and
             this is the only way a visitor without an API key ever sees the
             stage at all. Measured on the 39 golden queries with the
             container's caches warm: 0.30-2.37 ms (median 0.66) for both
             halves, and zero findings on every one of them.

  few-shot   The reverse. DEMO mode selects here, because no prompt exists to
             report and the ranking is the true thing to show. LIVE mode does
             NOT reconstruct: the assistant selects against the question PLUS
             the last two turns and the tables prior SQL used, and a panel that
             re-derived that could disagree with what was really sent. It draws
             only from a field the engine hands over, and until that field
             exists it draws nothing — which is the correct failure.

Selecting exemplars is a MiniLM forward pass, not a search: measured 276-1152 ms
(median 319) end to end, against 1.1-6.6 ms (median 2.1) when the caller supplies
a vector it already has. That is why the collection is built at page load — left
lazy, the first question in a fresh container paid 2,916 ms inside its own render
— and why the one engine change worth asking for is a way to hand this the
embedding engine.retrieval computed for the identical question a moment earlier.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_manifest import DOMAINS, MANIFEST, table_name  # noqa: E402
from engine import demo_mode  # noqa: E402
from engine.query import MAX_ROWS  # noqa: E402
from engine.sql_guard import FORBIDDEN, FORBIDDEN_FUNCTIONS, validate_sql  # noqa: E402
from engine.warehouse import build_warehouse, schema_catalog, table_names  # noqa: E402
from engine import exemplars, retrieval  # noqa: E402
from engine.verify import Verifier  # noqa: E402
from app import ui  # noqa: E402

st.set_page_config(page_title="Ask Your Data", page_icon="💬", layout="wide")

ui.inject()

LIVE_MODE = demo_mode.has_api_key()


@st.cache_resource
def get_connection():
    # Shared across sessions; queries run on isolated cursors.
    return build_warehouse()


@st.cache_resource(show_spinner="Preparing the schema index (first run downloads the embedding model)…")
def warm_retrieval(_con):
    """Build the Chroma index once, at startup, with the wait made visible.

    Chroma fetches all-MiniLM-L6-v2 the first time it embeds anything - 79 MB,
    into ~/.cache/chroma. Left lazy, that download lands in the middle of
    someone's first question and looks like a hang; on Render it landed during
    the health check and got the deploy cancelled. Doing it here moves the cost
    to page load, where a spinner can explain it, and @st.cache_resource means
    it happens once per container rather than once per session.

    Returns False rather than raising: retrieval is an optimisation, and an app
    that cannot embed should still answer from the full catalogue.
    """
    try:
        retrieval.build_index(_con)
    except Exception:
        return False
    # The exemplar bank embeds against its own Chroma collection, and the FIRST
    # embedding in a process pays for the ONNX session as well as the forward
    # pass. Measured by driving this script headlessly: leaving it lazy put
    # 2,916 ms inside the first question's render — nine times the 319 ms median
    # a warm process pays — landing exactly where nothing can explain it. Built
    # here, it lands under the spinner that already exists for the schema index,
    # which is the same argument this function was written for.
    #
    # Failure is swallowed separately from the line above: the schema index is
    # what RETRIEVAL_READY reports on, and a bank that will not build must not
    # make the rail say retrieval fell back to the full catalogue.
    try:
        exemplars.build_index()
    except Exception:
        pass
    return True


@st.cache_resource
def get_verifier(_con):
    """One Verifier for the container.

    It caches a column list and a uniqueness probe per (table, key) and both
    describe a warehouse that is built from CSV at startup and never written to,
    so the caches are valid for the process's life — and a fresh Verifier per
    turn would re-pay the fan-out probes that make `join_fanout` affordable.

    Constructed in BOTH modes on purpose. engine.verify imports nothing beyond
    the manifest and DuckDB; it needs no API key, and the checks it runs are the
    same checks whether the SQL was written by the model or committed to the
    accuracy contract. Demo mode running them on the reference SQL is a real
    thing happening, not a simulation of one.
    """
    return Verifier(_con)


@st.cache_resource
def get_assistant(_con):
    # Imported and constructed only in live mode, so demo mode never depends on
    # the anthropic client being usable.
    from engine.assistant import Assistant

    return Assistant(_con)


con = get_connection()
RETRIEVAL_READY = warm_retrieval(con)
verifier = get_verifier(con)
assistant = get_assistant(con) if LIVE_MODE else None
st.session_state.setdefault("turns", [])      # engine context (Turn objects)
st.session_state.setdefault("transcript", [])  # everything we rendered, incl. refusals

TABLE_COUNT = len(table_names(con))
POOL = max(retrieval.DEFAULT_K * 2, 12)  # the depth retrieve_hybrid reads each ranking to

# The rules Verifier can actually produce a finding for on this app's paths, in
# the order engine/verify.py's own docstring argues them, each with the severity
# that module assigns it. Drawn as a board for the same reason the guard draws
# its 26 verbs: "nothing structural was found" is a claim, and the rules that
# could have fired are the evidence for it.
#
# `ambiguous_entity` is deliberately ABSENT. It exists on Verifier, it is
# tested, and engine/assistant.py never calls ambiguity_note() — so drawing it
# would put a rule on the board that cannot fire on any turn this app runs, and
# a board with a rule that never runs is worse than no board. The moment the
# assistant calls it, it belongs here; tests/test_ui_readouts.py is what fails
# when this list and engine/verify.py disagree, so it cannot go stale quietly.
VERIFY_CHECKS = [
    ("cross_domain_join", "error"),
    ("cross_domain_cartesian", "error"),
    ("cross_domain_reference", "note"),
    ("cartesian_join", "warn"),
    # Promoted from warn to error: a fan-out returns a plausible number that is
    # merely too large, so advisory severity meant the inflated figure reached
    # the user narrated as fact. Fires on 0 of the 39 golden queries.
    ("join_fanout", "error"),
    ("empty_result", "warn"),
    ("null_scalar", "warn"),
    ("share_out_of_range", "note"),
]

ui.masthead(tables=TABLE_COUNT, domains=len(DOMAINS), live=LIVE_MODE)


def _status_rail() -> None:
    """The constants this session is running under, read from the modules that own them.

    Every value here is imported rather than typed: the guard's verb count comes
    from sql_guard.FORBIDDEN, the row cap from query.MAX_ROWS, k from
    retrieval.DEFAULT_K. A rail that is hand-maintained goes stale silently and
    then it is worse than no rail, because it is a confident wrong answer about
    what the machine is doing.
    """
    import duckdb

    index = ("<s>ready</s>" if RETRIEVAL_READY
             else "<u>fallback</u>")
    ui.status_rail([
        ("warehouse", f"<s>{TABLE_COUNT}</s> tables <em>· {len(DOMAINS)} domains</em>"),
        ("engine", f"duckdb <em>{duckdb.__version__}</em>"),
        ("index", f"MiniLM-L6-v2 · {index}<br><em>384-d · onnx · local</em>"),
        ("retrieval", f"hybrid rrf · k={retrieval.DEFAULT_K}<br><em>pool {POOL} · rrf_k={retrieval.RRF_K}</em>"),
        # Both of the guard's lists, because it enforces both. The rail said
        # "26 forbidden verbs" and the second list — the filesystem and
        # remote-scan readers sql_guard calls FORBIDDEN_FUNCTIONS — was never
        # mentioned anywhere on the page, which understated the boundary the
        # whole app leans on. Counts are read from the module, never typed.
        ("guard", f"read-only<br><em>{len(FORBIDDEN)} verbs · "
                  f"{len(FORBIDDEN_FUNCTIONS)} functions</em>"),
        # The guard's neighbour, and the same kind of session constant: a
        # boundary that is set before any question is asked. The count comes
        # from the roster below, which a test holds against engine/verify.py.
        ("verifier", f"structural<br><em>{len(VERIFY_CHECKS)} rules · deterministic</em>"),
        ("row cap", f"{MAX_ROWS:,} <em>/ query</em>"),
    ])


_status_rail()


@st.cache_data(show_spinner=False)
def _full_catalog_tokens() -> int:
    """Cost of the un-retrieved prompt block, for the grounding readout."""
    return max(1, len(schema_catalog(con)) // 4)


@st.cache_data(show_spinner=False)
def _catalog_by_domain() -> dict[str, list[str]]:
    """Every table this warehouse actually loaded, grouped by domain.

    Filtered against table_names(con) rather than trusting MANIFEST: a manifest
    entry whose source CSV is missing must not appear on the schema map as an
    unlit cell, because that would draw a table the retriever could never have
    selected and quietly inflate the denominator.
    """
    loaded = set(table_names(con))
    grouped: dict[str, list[str]] = {}
    for domain, table, _source, _description in MANIFEST:
        name = table_name(domain, table)
        if name in loaded:
            grouped.setdefault(domain, []).append(name)
    return grouped


@st.cache_data(show_spinner=False)
def _retrieval_bundle(question: str):
    """Everything the readouts need about one question's retrieval, fetched once.

    Cached on the question text, which is the only thing retrieval depends on -
    the warehouse and the index are @st.cache_resource singletons for the life
    of the container, so a question that has been retrieved once cannot retrieve
    differently later in the same session.

    Returns None rather than raising. Retrieval is an optimisation over pasting
    the whole catalogue, so a failure here has to cost a panel, not an answer:
    schema_catalog_for() already falls back to the full catalogue internally and
    the assistant keeps working.
    """
    try:
        started = time.perf_counter()
        hits = retrieval.retrieve_hybrid(question, con=con)
        # Only the hybrid call is the RETRIEVE stage. The two rankings gathered
        # below are re-run purely so the panel can show the ranks RRF consumed,
        # and charging the pipeline for the display's own overhead would
        # overstate what the assistant pays by roughly 50%.
        hybrid_ms = 1000 * (time.perf_counter() - started)

        vector = {hit.table: rank for rank, hit
                  in enumerate(retrieval.retrieve(question, k=POOL, con=con), 1)}
        keyword = {hit.table: rank for rank, hit
                   in enumerate(retrieval.retrieve_keyword(question, k=POOL, con=con), 1)}
        tokens_used = max(1, len(retrieval.schema_catalog_for(question, con)) // 4)
    except Exception:
        return None
    return {
        "hits": hits,
        "vector": vector,
        "keyword": keyword,
        # The set the fusion chose from: every table at least one retriever put
        # inside the pool. Free here — both dicts are already in hand — and it
        # is the funnel's missing middle stage, since "10 of 71" says nothing
        # about how many were ranked and then dropped.
        "candidates": len(set(vector) | set(keyword)),
        "tokens_used": tokens_used,
        "ms": hybrid_ms,
    }


def _show_grounding(bundle) -> None:
    """Which tables the retriever selected for this question, why, and what it saved."""
    if not bundle:
        return
    # Hybrid is what schema_catalog_for() actually uses, so the readout shows
    # the ranking the model was really given - not a prettier one.
    ui.grounding(
        bundle["hits"],
        total_tables=TABLE_COUNT,
        tokens_used=bundle["tokens_used"],
        tokens_full=_full_catalog_tokens(),
        vector_ranks=bundle["vector"],
        keyword_ranks=bundle["keyword"],
        pool=POOL,
        candidates=bundle.get("candidates"),
    )


def _guard_readout(sql: str) -> None:
    """Re-run the guard on this SQL and show what it checked.

    Built only from validate_sql's public return, never from its private
    helpers, and it reproduces the guard's short-circuit: when a check fails the
    later checks genuinely did not run, so they are not drawn as passing.

    The verb the panel lights is parsed out of the guard's own reason string
    rather than found by re-scanning the SQL here. A second scanner would have
    to reimplement _strip_literals to avoid firing on the word DELETE inside a
    string literal, and a UI that disagreed with the guard about what the guard
    blocked would be worse than one that showed nothing.
    """
    ok, reason = validate_sql(sql)
    # Every rung validate_sql actually walks, in its order. The FOURTH one was
    # missing and the omission did not fail quietly: sql_guard has a second list
    # behind the verbs — FORBIDDEN_FUNCTIONS, the filesystem and remote-scan
    # readers — and a query blocked by it matched none of the three markers
    # below, fell into the `else` branch, and drew the guard panel as
    # `non-empty query ✕`. Reproduced against the real guard:
    #
    #   validate_sql("SELECT * FROM read_csv_auto('/etc/passwd')")
    #     -> (False, 'forbidden function: read_csv_auto')
    #     -> checks [('non-empty query', False)]
    #
    # The query was not empty. The panel named the wrong boundary, and it named
    # it on the one path where naming the right one matters most.
    order = [
        ("single statement", "only a single statement"),
        ("starts SELECT / WITH", "must start with SELECT or WITH"),
        (f"none of {len(FORBIDDEN)} forbidden verbs", "forbidden keyword"),
        (f"none of {len(FORBIDDEN_FUNCTIONS)} forbidden functions", "forbidden function"),
    ]
    checks: list[tuple[str, bool]] = []
    if not ok and not any(marker in reason for _label, marker in order):
        # With all four rungs listed, "empty query" is genuinely the only
        # refusal left — the guard declining before the ladder starts.
        checks = [("non-empty query", False)]
    else:
        for label, marker in order:
            failed = (not ok) and marker in reason
            checks.append((label, not failed))
            if failed:
                break
    verb = reason.split("forbidden keyword:", 1)[1].strip() if "forbidden keyword:" in reason else ""
    ui.guard_verdict(ok=ok, reason=reason, checks=checks,
                     forbidden=FORBIDDEN, blocked_verb=verb)


# The prefix engine.query.run_query puts on the ONE error the guard produces, as
# opposed to the many DuckDB produces. It is the only thing that separates "the
# guard refused this" from "the warehouse could not run this" once both have
# been flattened into a correction string, and the pipeline strip needs that
# distinction to light the GUARD cell honestly.
#
# Re-typed here rather than imported because engine.query builds it inline, so
# tests/test_ui_turn_truth.py holds this literal against that module's source —
# the same arrangement VERIFY_CHECKS has with engine/verify.py.
GUARD_BLOCK_PREFIX = "blocked by SQL guard"


# --------------------------------------------------------------------------
# The verifier.
# --------------------------------------------------------------------------

def _verify_now(sql: str, result, question: str):
    """Run both halves of the verifier on SQL this app is about to show.

    Used by DEMO MODE only. There is no model there, so nothing has run the
    verifier for us — but the checks are deterministic, need no key, and cost
    almost nothing (measured over the 39 golden queries on this warehouse:
    check_sql 0.36-5.33 ms, median 0.72; check_result 0.01-1.09 ms, median
    0.02), so the panel can show real state instead of an empty frame.

    Live mode does NOT come through here. There the findings are the ones
    engine.assistant already computed and put on AskResult, and re-running the
    checks in the UI would be a second verifier that could disagree with the one
    whose verdict actually gated execution — the same trap `_guard_readout`
    avoids by never re-scanning the SQL itself.
    """
    started = time.perf_counter()
    findings = verifier.check_sql(sql)
    if result is not None and getattr(result, "ok", False):
        findings = findings + verifier.check_result(sql, result, question)
    return findings, 1000 * (time.perf_counter() - started)


def _verification_readout(findings, *, verify_ms=None, refused=False) -> None:
    """Findings as (check, severity, message), which is all app/ui.py accepts.

    Unpacked here rather than in ui.py so that module keeps importing nothing
    from engine — the same reason `forbidden` is passed in rather than imported
    there.
    """
    ui.verification(
        [(f.check, f.severity, f.message) for f in findings],
        checks=VERIFY_CHECKS, verify_ms=verify_ms, refused=refused,
    )


# --------------------------------------------------------------------------
# The few-shot bank.
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _exemplar_picks(question: str):
    """The k nearest solved questions to this one, and what selecting them cost.

    Cached on the question text for the same reason `_retrieval_bundle` is: this
    is a MiniLM forward pass, measured over the 39 golden questions at 276-1152
    ms (median 319), and Streamlit reruns the whole script on every widget
    interaction. Once per question per container, not once per keystroke.

    Almost all of that is the embedding, not the search: handing
    select_exemplars a vector it already has drops the same call to 1.1-6.6 ms
    (median 2.1). engine.retrieval embeds this identical question a moment
    earlier and does not expose the vector, which is the one change that would
    make this panel free — see the spec note in the track report.

    Returns None rather than raising. Chroma being unavailable must cost a
    panel, never an answer.
    """
    # The tables retrieval already chose for this question, which is the second
    # ranking select_exemplars fuses against — and passing them is not optional
    # dressing. Without it the selector is text-only, and the panel's foot says
    # the pairs were fused with the retrieved tables. A foot that described a
    # signal the call did not use would be the panel asserting its own mechanism
    # instead of showing it.
    bundle = _retrieval_bundle(question)
    tables = tuple(hit.table for hit in bundle["hits"]) if bundle else ()
    try:
        started = time.perf_counter()
        picks = exemplars.select_exemplars(question, k=exemplars.DEFAULT_K,
                                           retrieved_tables=tables)
        elapsed = 1000 * (time.perf_counter() - started)
    except Exception:
        return None
    if not picks:
        return None
    return {
        "picks": [(p.question, p.domain, p.sql, p.score) for p in picks],
        "corpus": len(exemplars.load_cases()),
        "fused": bool(tables),
        "ms": elapsed,
    }


def _show_exemplars(question: str, *, in_prompt: bool = False) -> None:
    """The few-shot bank's ranking for this question.

    `in_prompt=False` is the demo-mode reading and it is the true one there: no
    model runs, so no prompt exists and nothing can be "in" it. What IS true is
    the ranking — the same selector, over the same 39 committed pairs, with the
    leave-one-out rule dropping the question's own pair. That rule is what makes
    an eval over this corpus mean anything and it has never been visible; here
    it is something you can watch happen on a question you know is in the file.

    Live mode passes in_prompt=True and its own picks (see `render_entry`); this
    reconstruction is NOT used there, because the assistant selects against a
    context that includes prior turns and a required-table set the UI does not
    hold, and a panel that guessed at prompt content would be asserting what it
    cannot check.
    """
    bank = _exemplar_picks(question)
    if not bank:
        return
    ui.exemplars(bank["picks"], corpus=bank["corpus"], in_prompt=in_prompt,
                 fused=bank["fused"], select_ms=bank["ms"])


# --------------------------------------------------------------------------
# The plan and the result shape, read from DuckDB rather than described.
# --------------------------------------------------------------------------
# Both of these ask DuckDB about a statement WITHOUT running it: EXPLAIN builds
# the plan and stops, DESCRIBE resolves the output schema and stops. Measured
# across the 39 golden queries on this warehouse, EXPLAIN (FORMAT JSON) costs
# 0.20-1.70 ms (median 0.44) and DESCRIBE 0.18-0.61 ms (median 0.29), so a panel
# that shows the plan costs about a thousandth of what retrieval does.
#
# Both are still cached on the SQL text, because Streamlit reruns the whole
# script on every widget interaction and a transcript of ten answers would
# otherwise re-explain all ten on every keystroke.

# Which piece of extra_info actually says what an operator is doing. DuckDB
# attaches up to fifteen different keys and dumping all of them turns a plan
# into a wall; these are the ones that carry the operator's own decision.
_PLAN_DETAIL = {
    "SEQ_SCAN": ("Table", "Filters"),
    "HASH_JOIN": ("Join Type", "Conditions"),
    "PIECEWISE_MERGE_JOIN": ("Join Type", "Conditions"),
    "NESTED_LOOP_JOIN": ("Join Type", "Conditions"),
    "HASH_GROUP_BY": ("Groups",),
    "PERFECT_HASH_GROUP_BY": ("Groups",),
    "UNGROUPED_AGGREGATE": ("Aggregates",),
    "TOP_N": ("Top", "Order By"),
    "ORDER_BY": ("Order By",),
    "FILTER": ("Expression",),
    "CTE": ("CTE Name",),
    "CTE_SCAN": ("CTE Name", "CTE Index"),
    # PROJECTION and CROSS_PRODUCT are plumbing: the expression list is long,
    # repeats what the SQL above already says, and is not a decision the
    # optimiser made. They render as a bare operator name.
    "PROJECTION": (),
    "CROSS_PRODUCT": (),
}
_PLAN_FALLBACK = ("Table", "Expression", "Conditions", "Groups", "Aggregates",
                  "Order By", "Filters")
# A one-word lead-in for the keys whose value is unreadable without one. DuckDB
# writes a group key as "#0" and a TOP_N limit as "1"; on their own those are
# two bare numbers on a line. The values themselves are never rewritten - "#0"
# is the optimiser's own column reference and rendering it as a column name
# would be this panel inventing something DuckDB did not say.
_PLAN_LEAD = {"Top": "top", "Groups": "by", "Order By": "order",
              "Conditions": "on", "Filters": "where", "CTE Name": "cte",
              "CTE Index": "cte"}
_PLAN_MAX_NODES = 200


def _plan_value(info: dict, key: str) -> str:
    """One extra_info entry as a single line. DuckDB gives str or list of str."""
    value = info.get(key)
    if value in (None, "", []):
        return ""
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value if str(item).strip())
    text = " ".join(str(value).split())
    if not text:
        return ""
    if key == "Table":
        # Every table in this warehouse is memory.main.<name>; the catalog and
        # schema are constant and repeating them 47 times says nothing.
        text = text.rsplit(".", 1)[-1]
    lead = _PLAN_LEAD.get(key)
    return f"{lead} {text}" if lead else text


def _plan_rows(node: dict, prefix: str, root: bool, last: bool, out: list) -> None:
    """Depth-first walk, pre-rendering the tree guide for each row.

    The guide is built here rather than in app/ui.py because only the walk knows
    whether an ancestor still has siblings below it — the difference between a
    branch that continues down the left margin and one that has ended. A flat
    list of (depth, name) rows cannot reconstruct that, and a two-child operator
    is exactly where it shows.
    """
    if len(out) >= _PLAN_MAX_NODES:
        return
    guide = "" if root else prefix + ("└─ " if last else "├─ ")
    child_prefix = "" if root else prefix + ("   " if last else "│  ")
    name = str(node.get("name", "?"))
    info = node.get("extra_info") or {}
    keys = _PLAN_DETAIL.get(name, _PLAN_FALLBACK)
    detail = " · ".join(filter(None, (_plan_value(info, key) for key in keys)))
    try:
        card = int(str(info.get("Estimated Cardinality", "")).replace(",", ""))
    except (TypeError, ValueError):
        card = None
    out.append({"guide": guide, "name": name, "detail": detail, "card": card})
    children = node.get("children") or []
    for index, child in enumerate(children):
        _plan_rows(child, child_prefix, False, index == len(children) - 1, out)


@st.cache_data(show_spinner=False)
def _query_plan(sql: str):
    """DuckDB's physical plan for this SQL, flattened for display.

    Returns None rather than raising, and only ever runs on SQL the guard has
    already passed. EXPLAIN does not execute the statement — it stops after
    planning — so this cannot become a second, unguarded execution path.
    """
    import json

    if not validate_sql(sql)[0]:
        return None
    try:
        started = time.perf_counter()
        cur = con.cursor()
        try:
            cur.execute("EXPLAIN (FORMAT JSON) " + sql)
            raw = cur.fetchall()[0][1]
        finally:
            cur.close()
        elapsed = 1000 * (time.perf_counter() - started)
        tree = json.loads(raw)
        nodes: list[dict] = []
        for root in tree:
            _plan_rows(root, "", True, True, nodes)
        # The walk stops at _PLAN_MAX_NODES, and a panel that stops without
        # saying so is claiming the plan ended where the renderer gave up. The
        # true size is counted separately so the head can report both.
        total = sum(_plan_size(root) for root in tree)
    except Exception:
        return None
    return {"nodes": nodes, "ms": elapsed, "total": total}


def _plan_size(node: dict) -> int:
    return 1 + sum(_plan_size(child) for child in (node.get("children") or []))


@st.cache_data(show_spinner=False)
def _result_columns(sql: str) -> list[tuple[str, str]]:
    """The output schema in DuckDB's type names, via DESCRIBE.

    pandas would also answer this, but it would answer with its own inference
    over the returned values — int64 where the warehouse holds a BIGINT, object
    where it holds a VARCHAR. The point of the line is what the warehouse says.
    """
    if not validate_sql(sql)[0]:
        return []
    try:
        cur = con.cursor()
        try:
            cur.execute("DESCRIBE " + sql)
            rows = cur.fetchall()
        finally:
            cur.close()
    except Exception:
        return []
    return [(str(row[0]), str(row[1])) for row in rows]


def _result_readout(sql: str, frame, *, truncated: bool = False) -> None:
    """The plan, then the shape line, for one executed statement.

    In that order because the shape line has to sit directly on top of the
    dataframe it describes — it is a caption for the table, and a plan panel
    between the two would orphan it.
    """
    if frame is None:
        return
    plan = _query_plan(sql)
    if plan:
        ui.query_plan(plan["nodes"], plan_ms=plan["ms"], total=plan["total"],
                      returned=len(frame), truncated=truncated)
    ui.result_shape(_result_columns(sql), rows=len(frame),
                    truncated=truncated, cap=MAX_ROWS)


def _render_sidebar(active_question: str | None) -> None:
    """The catalogue, with this question's selection lit.

    Rendered at the END of the script rather than the top, even though it lands
    in the sidebar either way: the map is only worth drawing once the active
    question is known, and in both modes that is decided further down the page.
    """
    grouped = _catalog_by_domain()
    bundle = _retrieval_bundle(active_question) if active_question else None
    selected = {hit.table for hit in bundle["hits"]} if bundle else set()

    with st.sidebar:
        ui.schema_map(
            {domain: [(table, table in selected) for table in tables]
             for domain, tables in grouped.items()},
            retrieved=len(selected), total=TABLE_COUNT,
        )
        st.subheader("What you can ask about")
        for domain, blurb in DOMAINS.items():
            ui.domain_card(domain, blurb)
        st.divider()
        if st.button("Start a new conversation"):
            st.session_state.turns = []
            st.session_state.transcript = []
            st.rerun()


EXAMPLES = [
    "Which payer type collects the least of what it bills?",
    "How many active employees do we have, and how many left voluntarily?",
    "What's the overall order fill rate?",
    "Who is the top wholesale customer by revenue?",
    "How many migration artifacts passed parallel-run validation?",
]


def render_demo_mode(connection) -> None:
    """
    No key, no model. Serve the accuracy contract instead of a broken chat box.

    Each question executes the reference SQL committed alongside it, so the
    numbers on screen are the ones CI asserts on every push.
    """
    st.info(demo_mode.DEMO_NOTICE, icon=":material/science:")

    cases = demo_mode.load_golden_questions()
    grouped = demo_mode.questions_by_domain(cases)
    st.caption(
        f"{len(cases)} pre-registered questions across {len(grouped)} domains, "
        "each with reference SQL under test."
    )

    labels = {f"[{c['domain']}]  {c['question']}": c for c in cases}
    # Answer on selection rather than behind a button press: a visitor who has
    # never seen this app should land on a worked example, not an empty panel.
    choice = st.selectbox(
        "Pick a question", list(labels), index=0,
        help="Runs the reference SQL for this question against the warehouse now.",
    )
    active = labels[choice]
    bundle = _retrieval_bundle(active["question"])

    # Timed separately from execution so each pipeline cell reports its own
    # stage. The guard is pure and re-running it costs nothing, which is what
    # makes an honest measurement of it possible at all.
    guard_started = time.perf_counter()
    validate_sql(active["sql"])
    guard_ms = 1000 * (time.perf_counter() - guard_started)

    exec_started = time.perf_counter()
    result = demo_mode.answer(connection, active)
    exec_ms = 1000 * (time.perf_counter() - exec_started)

    # The verifier runs here too. No model wrote this SQL, but the checks are
    # deterministic and key-free, and running them on the reference SQL is the
    # only way a visitor without a key ever sees this stage at all. It is also
    # the stage's own strongest evidence: measured over all 39 golden queries,
    # both halves produce ZERO findings, which is what a check that fires on
    # good SQL would not do.
    findings, verify_ms = _verify_now(active["sql"], result.result, active["question"])
    timings = {"guard": guard_ms, "verify": verify_ms, "execute": exec_ms}
    if bundle:
        timings["retrieve"] = bundle["ms"]

    st.chat_message("user").write(active["question"])
    with st.chat_message("assistant"):
        if not result.ok:
            ui.pipeline(retrieved=True, generated=False, verified=True,
                        guarded=False, executed="fail", timings=timings)
            st.error(f"Reference SQL failed: {result.result.error}")
        else:
            # Demo mode runs committed SQL, so GENERATE is honestly dark: no
            # model wrote this. Lighting it would be the one lie this app
            # cannot afford. VERIFY, by contrast, genuinely ran — see above.
            ui.pipeline(retrieved=True, generated=False, verified=True,
                        guarded=True, executed=True, timings=timings)
            _show_grounding(bundle)
            ui.answer(
                result.headline,
                verified=result.matches_contract,
                verified_note="Asserted in evals/golden_questions.yaml and re-checked by CI on every push.",
            )
            if not result.matches_contract:
                st.warning(
                    "This does not match the contract's expected value — the "
                    "vendored data has drifted and the golden test should be red."
                )
            _guard_readout(result.sql)
            _verification_readout(findings, verify_ms=verify_ms)
            # Below the answer, with the guard and the verifier. Three solved
            # questions with their full reference SQL is a lot of page, and it
            # used to sit between the grounding panel and the number — so the
            # first thing a visitor met after "which tables" was three OTHER
            # questions' SQL. The panel is provenance; provenance reads after
            # the thing it vouches for.
            _show_exemplars(active["question"])
            st.code(result.sql, language="sql")
            frame = pd.DataFrame(result.result.rows, columns=result.result.columns)
            _result_readout(result.sql, frame, truncated=result.result.truncated)
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.caption(
                "Reference SQL, executed live — not written by the model. "
                "The model-authored path is what the API key unlocks."
            )

    st.divider()
    st.subheader("What the live mode adds")
    st.markdown(
        "With a key, the box below becomes a real chat: you ask anything in "
        "plain English, the model writes the SQL, a read-only guard validates "
        "it before execution, failed queries are fed their own error back for "
        "up to three attempts, and out-of-scope questions are refused rather "
        "than guessed at. Every answer still shows its SQL."
    )
    st.chat_input("Ask a question about the data...", disabled=True)
    st.caption("Disabled in demo mode — no model is configured.")
    _render_sidebar(active["question"])


if not LIVE_MODE:
    render_demo_mode(con)
    st.stop()

# Live mode only, so demo mode never imports the anthropic client.
# MAX_ATTEMPTS comes from the loop that enforces it rather than being typed into
# the panel: the ledger's denominator has to move when that constant does.
from engine.assistant import MAX_ATTEMPTS, AssistantUnavailable  # noqa: E402

if not st.session_state.transcript:
    st.write("Try one of these, or type your own:")
clicked = None
if not st.session_state.transcript:
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES, strict=True):
        if col.button(ex, use_container_width=True):
            clicked = ex


def render_entry(entry):
    bundle = _retrieval_bundle(entry["question"])
    st.chat_message("user").write(entry["question"])
    with st.chat_message("assistant"):
        findings = entry.get("findings") or []
        if entry["refused"]:
            # TWO different refusals arrive here and they were being narrated as
            # one. engine.assistant returns refused=True either because the
            # model called cannot_answer — the question is out of scope — or
            # because the verifier still had a blocking finding on the final
            # attempt and the loop declined to execute SQL it could not stand
            # behind. Only the first is "I can't answer that from the loaded
            # data"; the second wrote a query and refused to run it, which is a
            # much more interesting thing to say and the reason the verifier
            # panel exists.
            #
            # It also stops a leak. On the verification path `reason` is
            # verify.correction_message(), whose closing lines are addressed to
            # the MODEL ("Write a corrected single SELECT query… call
            # cannot_answer instead"), and they were being pasted into a warning
            # a human reads. The findings say the same thing in the reader's
            # direction, so the panel carries it and the raw prompt text does
            # not reach the page.
            blocked = bool(findings) and bool(entry["sql"])
            ui.pipeline(retrieved=True, generated=True,
                        verified="fail" if blocked else None,
                        guarded=False, executed=False,
                        timings={"retrieve": bundle["ms"]} if bundle else None)
            if blocked:
                st.warning("I wrote a query for this and then refused to run it — "
                           "the structural checks below say the result would not "
                           "have meant anything.")
                _verification_readout(findings, refused=True)
            else:
                st.warning(f"I can't answer that from the loaded data: {entry['reason']}")
            # A refusal can arrive on attempt 2 or 3 — the model corrected once,
            # then declined — and that history is the interesting part of a
            # refusal. It used to be dropped on the floor by this early return.
            ui.attempt_ledger(attempts=entry["attempts"],
                              corrections=entry.get("corrections") or [],
                              max_attempts=MAX_ATTEMPTS, ok=False)
            if blocked:
                # The refused SQL, shown unrun. A refusal that hides its own
                # query asks to be taken on trust, which is the one currency
                # this app does not deal in.
                st.caption("The query that was written and not executed:")
                st.code(entry["sql"], language="sql")
            return
        # What the turn ACTUALLY did, read from the loop's own record. Two of
        # these three cells were previously inferred, and both inferences were
        # wrong on the same path — reproduced offline against the scripted
        # client in tests/test_ui_turn_truth.py, on a turn whose three attempts
        # all died with `Binder Error: Referenced column ... not found`:
        #
        #   guard    was `True if attempts == 1 else "fail"`, so ANY retry lit
        #            GUARD in the alert colour. The guard had passed all three
        #            of those attempts; what failed was the warehouse. A retry
        #            is what the `retry ×N` cell is for, and conflating it with
        #            a guard rejection made a guard block and a binder error
        #            draw exactly the same strip — which is the one thing this
        #            strip exists not to do.
        #   execute  was `True` unconditionally. Nothing executed on that turn.
        #            The strip lit EXECUTE over a page whose expander said
        #            "Query error".
        #
        # So the guard cell now keys off the guard's own marker in the
        # corrections, and EXECUTE off whether a query really came back.
        ran = bool(entry.get("ran"))
        corrections = entry.get("corrections") or []
        guard_blocked = any(str(c).startswith(GUARD_BLOCK_PREFIX) for c in corrections)
        # A turn whose LAST correction was a guard block never reached the
        # executor at all, so EXECUTE is dark rather than failed: the segment
        # before it already carries where the signal stopped.
        stopped_at_guard = (not ran and bool(corrections)
                            and str(corrections[-1]).startswith(GUARD_BLOCK_PREFIX))
        ui.pipeline(retrieved=True, generated=True, verified=True,
                    guarded="fail" if guard_blocked else True,
                    executed=True if ran else (False if stopped_at_guard else "fail"),
                    attempts=entry["attempts"],
                    # Only RETRIEVE is separable here. The assistant's own
                    # generate/verify/guard/execute happen inside one call, so
                    # the rest of the clock is reported as a round trip below
                    # rather than split across cells on a guess.
                    timings={"retrieve": bundle["ms"]} if bundle else None)
        _show_grounding(bundle)
        if entry["answer"]:
            ui.answer(entry["answer"])
        else:
            # The loop spent every attempt and the last query still failed, so
            # there is no answer to headline. This used to call ui.answer("")
            # anyway and emit `<div class="ayd-answer"></div>` — an empty
            # 1.75rem hole where the product goes, directly under a strip that
            # was claiming EXECUTE had lit. Saying what happened is cheaper than
            # both and truer than either.
            st.error(
                f"{entry['attempts']} attempts, and the query still failed. The "
                "loop stopped rather than guessing at a number; the errors it "
                "fed back to the model are below."
            )
        if entry.get("elapsed_ms"):
            ui.note(f"Model round trip {entry['elapsed_ms']:,.0f} ms "
                    f"(generate → guard → execute, {entry['attempts']} attempt"
                    f"{'s' if entry['attempts'] != 1 else ''}).")
        # The loop's own record, when it ran more than once. This replaces a
        # one-line note that quoted only corrections[0] and dropped the rest —
        # on a three-attempt answer that silently hid the second error, which is
        # the one that says whether the model was converging or thrashing.
        # `ok` is the same authoritative flag the strip reads, not a second
        # inference from it. The old expression (`rows is not None or not
        # error`) agreed with it on every path but got there by accident: a
        # query that succeeds and returns NO rows also has rows=None, so the
        # first clause was load-bearing for a reason that had nothing to do with
        # whether the attempt ran.
        ui.attempt_ledger(attempts=entry["attempts"], corrections=corrections,
                          max_attempts=MAX_ATTEMPTS, ok=ran)
        if entry["sql"]:
            _guard_readout(entry["sql"])
            # The findings engine.assistant already computed, not a second run.
            # Re-verifying here would be a second verifier that could disagree
            # with the one whose verdict actually gated execution — the trap
            # `_guard_readout` avoids by never re-scanning the SQL itself. No
            # timing is shown for the same reason: the clock that measured it
            # is inside assistant.ask, and inventing one here would be the panel
            # reporting a number it did not take.
            _verification_readout(findings)
        if entry.get("exemplars"):
            # Only ever the assistant's own selection, handed over on the
            # result. The UI does not reconstruct it: the prompt is built from
            # the question PLUS the last two turns and the tables prior SQL
            # used, and a panel that re-derived that could disagree with what
            # was actually sent. Until engine.assistant carries the field this
            # branch simply does not draw, which is the correct failure.
            #
            # It sits BELOW the answer now. It is three verified pairs with
            # their full SQL, and above the answer it was the thing standing
            # between the question and its number — provenance charged as if it
            # were the product. Down here it joins the guard and the verifier,
            # which is the group it belongs to: everything that argues the
            # answer, read after the answer.
            _bank = entry["exemplars"]
            ui.exemplars(_bank["picks"], corpus=_bank["corpus"], in_prompt=True,
                         fused=bool(_bank.get("fused")))
        with st.expander("Show the SQL and the data behind this answer"):
            st.code(entry["sql"], language="sql")
            if entry["rows"] is not None:
                _result_readout(entry["sql"], entry["rows"],
                                truncated=bool(entry["truncated"]))
                st.dataframe(entry["rows"], use_container_width=True, hide_index=True)
            elif entry["error"]:
                st.error(f"Query error: {entry['error']}")
            if entry.get("usage"):
                u = entry["usage"]
                bits = [f"tokens in {u.get('input_tokens', 0):,}",
                        f"out {u.get('output_tokens', 0):,}"]
                if u.get("cache_read_input_tokens"):
                    bits.append(f"cache read {u['cache_read_input_tokens']:,} "
                                "(the schema catalog served from cache)")
                st.caption(" · ".join(bits))


for entry in st.session_state.transcript:
    render_entry(entry)

question = st.chat_input("Ask a question about the data...") or clicked

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Writing SQL and running it..."):
            started = time.perf_counter()
            try:
                result = assistant.ask(question, history=st.session_state.turns)
            except AssistantUnavailable as e:
                st.error(f"The language model is unavailable: {e}\n\n"
                         "Set `ANTHROPIC_API_KEY` (and check your credit balance), "
                         "then ask again.")
                st.stop()
            elapsed_ms = 1000 * (time.perf_counter() - started)

    entry = {
        "question": question,
        "refused": result.refused,
        "reason": result.reason,
        "answer": result.answer,
        "sql": result.sql,
        "attempts": result.attempts,
        "corrections": result.corrections,
        # Previously dropped here. engine.assistant computes these on every turn
        # and puts them on AskResult; the transcript entry was the point at
        # which they stopped existing, so the verifier ran on every question and
        # nothing it decided ever reached the page.
        "findings": result.findings,
        # Whether a query really came back. AskResult.ok is the loop's own
        # answer to that and it is the only honest source for the EXECUTE cell:
        # `rows is not None` is false for a query that succeeded and matched
        # nothing, and `not error` is true for a refusal that never ran one.
        "ran": bool(result.ok),
        "rows": (pd.DataFrame(result.result.rows, columns=result.result.columns)
                 if (result.result and result.result.ok and result.result.rows) else None),
        "truncated": bool(result.result and result.result.truncated),
        "error": result.result.error if (result.result and not result.result.ok) else "",
        "usage": result.usage,
        "elapsed_ms": elapsed_ms,
    }
    st.session_state.transcript.append(entry)
    if result.ok:
        st.session_state.turns.append(result.as_turn())
    st.rerun()

_render_sidebar(
    st.session_state.transcript[-1]["question"] if st.session_state.transcript else None
)
