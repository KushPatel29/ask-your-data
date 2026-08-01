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
from engine.warehouse import build_warehouse, table_names  # noqa: E402

st.set_page_config(page_title="Ask Your Data", page_icon="💬", layout="wide")

LIVE_MODE = demo_mode.has_api_key()


@st.cache_resource
def get_connection():
    # Shared across sessions; queries run on isolated cursors.
    return build_warehouse()


@st.cache_resource
def get_assistant(_con):
    # Imported and constructed only in live mode, so demo mode never depends on
    # the anthropic client being usable.
    from engine.assistant import Assistant

    return Assistant(_con)


con = get_connection()
assistant = get_assistant(con) if LIVE_MODE else None
st.session_state.setdefault("turns", [])      # engine context (Turn objects)
st.session_state.setdefault("transcript", [])  # everything we rendered, incl. refusals

st.title("💬 Ask Your Data")
st.caption("A natural-language layer over the analytics datasets from my portfolio "
           "projects. Ask in plain English — it writes the SQL, runs it, and shows "
           "its work. Follow-up questions welcome.")

with st.sidebar:
    st.subheader("What you can ask about")
    for domain, blurb in DOMAINS.items():
        st.markdown(f"**{domain}** — {blurb}")
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
            st.error(f"Reference SQL failed: {result.result.error}")
        else:
            st.markdown(f"## {result.headline}")
            if result.matches_contract:
                st.caption(
                    "Matches the value asserted in `evals/golden_questions.yaml`, "
                    "which CI re-checks on every push."
                )
            else:
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
            st.warning(f"I can't answer that from the loaded data: {entry['reason']}")
            return
        st.markdown(f"**{entry['answer']}**")
        if entry["attempts"] > 1:
            st.caption(f"Self-corrected after {entry['attempts']} attempts "
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
