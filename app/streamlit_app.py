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
from engine.sql_guard import FORBIDDEN, validate_sql  # noqa: E402
from engine.warehouse import build_warehouse, schema_catalog, table_names  # noqa: E402
from engine import retrieval  # noqa: E402
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
        return True
    except Exception:
        return False


@st.cache_resource
def get_assistant(_con):
    # Imported and constructed only in live mode, so demo mode never depends on
    # the anthropic client being usable.
    from engine.assistant import Assistant

    return Assistant(_con)


con = get_connection()
RETRIEVAL_READY = warm_retrieval(con)
assistant = get_assistant(con) if LIVE_MODE else None
st.session_state.setdefault("turns", [])      # engine context (Turn objects)
st.session_state.setdefault("transcript", [])  # everything we rendered, incl. refusals

TABLE_COUNT = len(table_names(con))
POOL = max(retrieval.DEFAULT_K * 2, 12)  # the depth retrieve_hybrid reads each ranking to

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
        ("guard", f"read-only<br><em>{len(FORBIDDEN)} forbidden verbs</em>"),
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
    )


def _guard_readout(sql: str) -> None:
    """Re-run the guard on this SQL and show what it checked.

    Built only from validate_sql's public return, never from its private
    helpers, and it reproduces the guard's short-circuit: when a check fails the
    later checks genuinely did not run, so they are not drawn as passing.
    """
    ok, reason = validate_sql(sql)
    order = [
        ("single statement", "only a single statement"),
        ("starts SELECT / WITH", "must start with SELECT or WITH"),
        (f"none of {len(FORBIDDEN)} forbidden verbs", "forbidden keyword"),
    ]
    checks: list[tuple[str, bool]] = []
    if not ok and not any(marker in reason for _label, marker in order):
        checks = [("non-empty query", False)]  # the only other refusal the guard makes
    else:
        for label, marker in order:
            failed = (not ok) and marker in reason
            checks.append((label, not failed))
            if failed:
                break
    ui.guard_verdict(ok=ok, reason=reason, checks=checks)


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

    timings = {"guard": guard_ms, "execute": exec_ms}
    if bundle:
        timings["retrieve"] = bundle["ms"]

    st.chat_message("user").write(active["question"])
    with st.chat_message("assistant"):
        if not result.ok:
            ui.pipeline(retrieved=True, generated=False, guarded=False, executed="fail",
                        timings=timings)
            st.error(f"Reference SQL failed: {result.result.error}")
        else:
            # Demo mode runs committed SQL, so GENERATE is honestly dark: no
            # model wrote this. Lighting it would be the one lie this app
            # cannot afford.
            ui.pipeline(retrieved=True, generated=False, guarded=True, executed=True,
                        timings=timings)
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
            st.code(result.sql, language="sql")
            st.dataframe(
                pd.DataFrame(result.result.rows, columns=result.result.columns),
                use_container_width=True, hide_index=True,
            )
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
from engine.assistant import AssistantUnavailable  # noqa: E402

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
        if entry["refused"]:
            ui.pipeline(retrieved=True, generated=True, guarded=False, executed=False,
                        timings={"retrieve": bundle["ms"]} if bundle else None)
            st.warning(f"I can't answer that from the loaded data: {entry['reason']}")
            return
        ui.pipeline(retrieved=True, generated=True,
                    guarded=True if entry["attempts"] == 1 else "fail",
                    executed=True, attempts=entry["attempts"],
                    # Only RETRIEVE is separable here. The assistant's own
                    # generate/guard/execute happen inside one call, so the rest
                    # of the clock is reported as a round trip below rather than
                    # split across cells on a guess.
                    timings={"retrieve": bundle["ms"]} if bundle else None)
        _show_grounding(bundle)
        ui.answer(entry["answer"])
        if entry.get("elapsed_ms"):
            ui.note(f"Model round trip {entry['elapsed_ms']:,.0f} ms "
                    f"(generate → guard → execute, {entry['attempts']} attempt"
                    f"{'s' if entry['attempts'] != 1 else ''}).")
        if entry["attempts"] > 1:
            ui.note(f"Self-corrected after {entry['attempts']} attempts "
                    f"(first error: {entry['corrections'][0]})")
        if entry["sql"]:
            _guard_readout(entry["sql"])
        with st.expander("Show the SQL and the data behind this answer"):
            st.code(entry["sql"], language="sql")
            if entry["rows"] is not None:
                st.dataframe(entry["rows"], use_container_width=True, hide_index=True)
                if entry["truncated"]:
                    st.caption("Showing the first rows only.")
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
