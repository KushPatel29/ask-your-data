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
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_manifest import DOMAINS  # noqa: E402
from engine import demo_mode  # noqa: E402
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

ui.masthead(tables=len(table_names(con)), domains=len(DOMAINS), live=LIVE_MODE)


@st.cache_data(show_spinner=False)
def _full_catalog_tokens() -> int:
    """Cost of the un-retrieved prompt block, for the grounding readout."""
    return max(1, len(schema_catalog(con)) // 4)


def _show_grounding(question: str) -> None:
    """Which tables the retriever selected for this question, and what it saved."""
    try:
        # Hybrid is what schema_catalog_for() actually uses, so the readout
        # shows the ranking the model was really given - not a prettier one.
        hits = retrieval.retrieve_hybrid(question, con=con)
        used = max(1, len(retrieval.schema_catalog_for(question, con)) // 4)
    except Exception:
        # Retrieval is an optimisation, not a dependency: if the index cannot be
        # built the assistant still answers from the full catalogue, and the
        # panel simply does not render.
        return
    ui.grounding(hits, total_tables=len(table_names(con)),
                 tokens_used=used, tokens_full=_full_catalog_tokens())

with st.sidebar:
    st.subheader("What you can ask about")
    for domain, blurb in DOMAINS.items():
        ui.domain_card(domain, blurb)
    st.divider()
    st.caption(f"{len(table_names(con))} tables loaded across {len(DOMAINS)} domains.")
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
    result = demo_mode.answer(connection, active)

    st.chat_message("user").write(active["question"])
    with st.chat_message("assistant"):
        if not result.ok:
            ui.pipeline(retrieved=True, generated=False, guarded=False, executed="fail")
            st.error(f"Reference SQL failed: {result.result.error}")
        else:
            # Demo mode runs committed SQL, so GENERATE is honestly dark: no
            # model wrote this. Lighting it would be the one lie this app
            # cannot afford.
            ui.pipeline(retrieved=True, generated=False, guarded=True, executed=True)
            _show_grounding(active["question"])
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
    st.chat_message("user").write(entry["question"])
    with st.chat_message("assistant"):
        if entry["refused"]:
            ui.pipeline(retrieved=True, generated=True, guarded=False, executed=False)
            st.warning(f"I can't answer that from the loaded data: {entry['reason']}")
            return
        ui.pipeline(retrieved=True, generated=True,
                    guarded=True if entry["attempts"] == 1 else "fail",
                    executed=True, attempts=entry["attempts"])
        _show_grounding(entry["question"])
        ui.answer(entry["answer"])
        if entry["attempts"] > 1:
            ui.note(f"Self-corrected after {entry['attempts']} attempts "
                    f"(first error: {entry['corrections'][0]})")
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
            try:
                result = assistant.ask(question, history=st.session_state.turns)
            except AssistantUnavailable as e:
                st.error(f"The language model is unavailable: {e}\n\n"
                         "Set `ANTHROPIC_API_KEY` (and check your credit balance), "
                         "then ask again.")
                st.stop()

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
    }
    st.session_state.transcript.append(entry)
    if result.ok:
        st.session_state.turns.append(result.as_turn())
    st.rerun()
