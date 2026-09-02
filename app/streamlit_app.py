"""
Ask-Your-Data chat UI.

    streamlit run app/streamlit_app.py

A real conversation: follow-up questions ("and by region?") carry the earlier
turns as context. Every answer shows the plain-English result, the SQL the model
wrote, and the returned rows — so a reader can always check the number against
the query.

With an Anthropic key or an explicitly configured local OpenAI-compatible
provider, that is what runs. Without one the deterministic compiler writes SQL
locally. That is genuinely different from a model writing SQL, and the UI says
so rather than blurring the two.

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

import hashlib
import re
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import ui  # noqa: E402
from data_manifest import DOMAINS, MANIFEST, table_name  # noqa: E402
from engine import (  # noqa: E402
    access,
    audit,
    deadline,
    demo_mode,
    exemplars,
    narrate,
    planner,
    providers,
    retrieval,
    voice,
)
from engine import metrics as metric_layer
from engine.limits import MAX_QUESTION_CHARS  # noqa: E402
from engine.query import MAX_ROWS, run_query  # noqa: E402
from engine.semantics import Layer  # noqa: E402
from engine.sql_guard import (  # noqa: E402
    FORBIDDEN,
    FORBIDDEN_FUNCTIONS,
    MAX_SQL_CHARS,
    validate_sql,
)
from engine.verify import Verifier  # noqa: E402
from engine.warehouse import build_warehouse, table_names  # noqa: E402

st.set_page_config(page_title="Ask Your Data", page_icon="💬", layout="wide")

ui.inject()

def _session_key() -> str:
    """A key the visitor pasted, if any. Never persisted, never logged.

    The deployed app has no key of its own -- putting one on a public URL is an
    unmetered spend surface -- so the model path was unreachable for everybody
    who visited. This makes it reachable for anyone who brings their own,
    without the deployment ever holding one. It lives in st.session_state, which
    is per-browser-session and dies with the tab.
    """
    return str(st.session_state.get("byok", "") or "").strip()


def _session_actor() -> str:
    """A stable, meaningless-by-design identifier for this browser session.

    Not an identity. Demo mode has no login, so the honest thing an audit record
    can say is "the same visitor asked these four questions", and nothing more.
    A field called `actor` holding an email address the app never verified
    would be a lie the trail has no way to check, and a trail with one lie in
    it is not evidence of anything.
    """
    if "actor" not in st.session_state:
        st.session_state.actor = "session-" + uuid.uuid4().hex[:10]
    return str(st.session_state.actor)


def _principal() -> access.Principal:
    """Return a demo principal or a cryptographically verified OIDC identity."""
    if access.auth_mode() == access.AUTH_DISABLED:
        return access.Principal.demo(_session_actor())
    return access.authenticate_headers(st.context.headers)


def _actor() -> str:
    """The verified subject in OIDC mode; an explicit session label in demo mode."""
    return PRINCIPAL.subject


# The spend ceiling. A demo whose README tells the reader to deploy it with
# their own key needs a documented bound on what one visitor can spend, and
# "the compiler is free" is not an answer once a key is pasted in.
#
# This is a SESSION counter and it is therefore trivially bypassed by opening a
# new tab. Saying so here matters more than the number does: a limit that
# presents itself as a security control is worse than no limit, because
# somebody stops looking for the real one. The real ceiling is a spend cap on
# the key itself, and the README says that where a reader will see it.
MAX_QUESTIONS_PER_SESSION = 60


def _questions_asked() -> int:
    return int(st.session_state.get("asked", 0))


def _session_budget_exhausted() -> bool:
    return _questions_asked() >= MAX_QUESTIONS_PER_SESSION


def _count_question() -> None:
    st.session_state.asked = _questions_asked() + 1


try:
    SERVER_MODEL_ENABLED = providers.model_provider_configured()
except providers.ProviderUnavailable as exc:
    st.error(f"Model provider configuration error: {exc}")
    st.stop()
LIVE_MODE = SERVER_MODEL_ENABLED or bool(_session_key())


@st.cache_resource
def get_connection():
    # Shared across sessions; queries run on isolated cursors.
    return build_warehouse()


@st.cache_resource(show_spinner="Preparing the schema index "
                                "(first run downloads the embedding model)…")
def warm_retrieval(_con):
    """Build the local exact index once, at startup, with the wait made visible.

    The embedder fetches all-MiniLM-L6-v2 the first time it runs - 79 MB, into
    the app's model cache. Left lazy, that download lands in the middle of
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
    # The exemplar bank embeds against its own exact index, and the FIRST
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


@st.cache_resource(show_spinner="Profiling the warehouse (roles, grains, join graph)…")
def get_layer(_con):
    """The semantic layer the keyless planner compiles against.

    Built once per container. It probes every column's cardinality and every
    table's grain, which is ~90 extra COUNT(DISTINCT) statements on top of one
    pass per table -- measured at 2.4 s over these 71 tables, which is why it
    sits behind the same kind of cache and spinner as the schema index rather
    than inside the first question's render.

    Returns None rather than raising, for the same reason `warm_retrieval` does.
    This runs at import on a public deployment, so an exception here is not a
    degraded feature -- it is a blank page where the app used to be. The
    compiler cannot work without the layer, but the accuracy contract can, and
    an app that says "the compiler is unavailable, here are 39 questions whose
    SQL is committed" is a working app. One that stack-traces is not.
    """
    try:
        return Layer(_con)
    except Exception:
        return None


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
def get_metric_registry():
    """The committed definitions whose expected values CI re-runs."""
    return metric_layer.load_metrics()


def _live_assistant(_con):
    """The assistant for THIS session, honouring a pasted key.

    Not cached on the connection alone: two visitors can be in the same
    container with different keys, and a @st.cache_resource assistant would hand
    the second one the first one's client. Every instance is also bound to this
    session's immutable access scope, so neither credentials nor policy can
    cross a browser-session boundary.
    """
    key = _session_key()
    assistant_key = (key, ACCESS.fingerprint)
    if st.session_state.get("_assistant_key") != assistant_key:
        from engine.assistant import Assistant

        if key:
            import anthropic

            instance = Assistant(
                _con, client=anthropic.Anthropic(api_key=key), access=ACCESS,
            )
        else:
            instance = Assistant(_con, access=ACCESS)
        st.session_state["_assistant"] = instance
        st.session_state["_assistant_key"] = assistant_key
    return st.session_state["_assistant"]


def _voice_api_key() -> str:
    """A server voice key or a per-session key the visitor supplied."""
    return str(st.session_state.get("voice_byok", "") or voice.server_api_key()).strip()


def _voice_engine():
    """(client, label) for the speech engine this session will use.

    Resolution lives in engine/voice.resolve so the precedence is testable
    without Streamlit: an explicitly configured endpoint or key wins, then the
    in-process open models, then nothing. What changed for the app is that the
    third case is now rare — voice used to require an operator, and the public
    deployment has none.

    Never raises. A misconfigured remote provider must cost the page its voice
    control, not its ability to answer questions, so the failure is captured
    and shown beside the control that caused it.
    """
    key = _voice_api_key()
    endpoint = voice.base_url()
    fingerprint = hashlib.sha256(
        f"{endpoint}\0{key}\0{voice.engine_preference()}".encode("utf-8")
    ).hexdigest()
    if st.session_state.get("_voice_key_fingerprint") != fingerprint:
        st.session_state.pop("_voice_error", None)
        try:
            client, label = voice.resolve(key)
        except voice.VoiceUnavailable as exc:
            client, label = None, ""
            st.session_state["_voice_error"] = str(exc)
        st.session_state["_voice_client"] = client
        st.session_state["_voice_label"] = label
        st.session_state["_voice_key_fingerprint"] = fingerprint
    return st.session_state.get("_voice_client"), st.session_state.get("_voice_label", "")


def _voice_client():
    return _voice_engine()[0]


def _voice_label() -> str:
    return _voice_engine()[1]


try:
    PRINCIPAL = _principal()
    ACCESS = access.access_scope(PRINCIPAL)
except (access.AuthenticationError, access.PolicyConfigurationError) as exc:
    st.error(f"Access denied: {exc}")
    st.stop()

con = get_connection()
try:
    access.validate_scope_schema(con, ACCESS)
except access.PolicyConfigurationError as exc:
    st.error(f"Access denied: {exc}")
    st.stop()
RETRIEVAL_READY = warm_retrieval(con)


@st.cache_resource(show_spinner=False)
def _warm_voice() -> bool:
    """Start loading the speech voice in the background, once per container.

    Not for speed in the abstract — for autoplay. A cold first synthesis costs
    21.5s against 0.17s warm, and a browser only lets audio start by itself
    within about five seconds of the click that asked for it. So on a cold
    container the first answer's audio arrived long after its click and the
    browser refused to play it, silently, while every later answer worked.

    @st.cache_resource so it fires once per container rather than once per
    session, and the thread is what keeps it off the boot path.
    """
    from engine import local_voice

    try:
        return local_voice.prewarm()
    except Exception:  # noqa: BLE001 - never let a warm-up break the page
        return False


VOICE_WARMING = _warm_voice()
layer = get_layer(con)
PLANNER_READY = layer is not None
verifier = get_verifier(con)
metric_registry = tuple(
    metric for metric in get_metric_registry() if metric.domain in {
        name.split("_", 1)[0] for name in ACCESS.allowed_tables
    }
)
assistant = _live_assistant(con) if LIVE_MODE else None
st.session_state.setdefault("turns", [])      # engine context (Turn objects)
st.session_state.setdefault("transcript", [])  # everything we rendered, incl. refusals

TABLE_COUNT = len(set(table_names(con)) & set(ACCESS.allowed_tables))
ACCESS_DOMAINS = {name.split("_", 1)[0] for name in ACCESS.allowed_tables}
POOL = max(retrieval.DEFAULT_K * 2, 12)  # the depth retrieve_hybrid reads each ranking to

# The interpreter this repository is proven on, read from the file that pins it
# rather than typed here — the point of the rail is that nothing on it is
# hand-maintained. Missing on an exotic checkout is not worth an exception on
# page load, so it degrades to the running version and the rail simply agrees
# with itself.
try:
    PINNED_PYTHON = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
except OSError:
    PINNED_PYTHON = ".".join(__import__("platform").python_version_tuple()[:2])

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
    ("intent_order", "error"),
    ("required_business_filter", "error"),
    ("governed_metric_filter", "error"),
    ("empty_result", "warn"),
    ("null_scalar", "warn"),
    ("share_out_of_range", "note"),
]

CURRENT_VIEW = str(st.session_state.get("workspace-nav", "Ask"))
ui.masthead(
    tables=TABLE_COUNT,
    domains=len(ACCESS_DOMAINS),
    live=LIVE_MODE,
    compact=CURRENT_VIEW != "Ask",
)

WORKSPACE_VIEW = st.segmented_control(
    "Workspace",
    ("Ask", "Data catalog", "Trust center"),
    default="Ask",
    key="workspace-nav",
    label_visibility="collapsed",
    width="stretch",
) or "Ask"


def _status_rail() -> None:
    """The constants this session is running under, read from the modules that own them.

    Every value here is imported rather than typed: the guard's verb count comes
    from sql_guard.FORBIDDEN, the row cap from query.MAX_ROWS, k from
    retrieval.DEFAULT_K. A rail that is hand-maintained goes stale silently and
    then it is worse than no rail, because it is a confident wrong answer about
    what the machine is doing.
    """
    import platform

    import duckdb

    index = ("<s>ready</s>" if RETRIEVAL_READY
             else "<u>fallback</u>")
    # The interpreter the deployment actually got, set against the one the
    # suite was proven on. These are the same decision in two places and they
    # drift silently: a managed host picks the Python version in a dropdown at
    # app-creation time and does not read `.python-version`, so a new
    # deployment can come up on an interpreter CI never ran — and the first
    # symptom is a wheel that will not install, or worse, one that does.
    # `tests/test_runtime_pinning.py` holds the repository's three copies of
    # this to one answer; the rail is what carries it onto the deployed page,
    # where the version is otherwise invisible.
    running = ".".join(platform.python_version_tuple()[:2])
    pinned = PINNED_PYTHON
    runtime = (f"python <em>{running}</em>" if running == pinned
               else f"python <u>{running}</u><br><em>pinned {pinned}</em>")
    ui.status_rail([
        ("warehouse", f"<s>{TABLE_COUNT}</s> tables <em>· {len(ACCESS_DOMAINS)} domains</em>"),
        ("engine", f"duckdb <em>{duckdb.__version__}</em>"),
        ("runtime", runtime),
        ("index", f"MiniLM-L6-v2 · {index}<br><em>exact cosine · onnx · local</em>"),
        ("retrieval", f"hybrid rrf · k={retrieval.DEFAULT_K}<br>"
                      f"<em>pool {POOL} · rrf_k={retrieval.RRF_K}</em>"),
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
        # Which speech engine really resolved, not which one is configurable.
        # The rail used to answer "is an endpoint or key set", so on the public
        # deployment it said "optional · configure local voice or cloud key"
        # while voice was, in fact, running on models inside this process.
        ("voice", voice.describe_engine(_voice_label())),
        ("row cap", f"{MAX_ROWS:,} <em>/ query</em>"),
        ("deadline", f"{deadline.configured_timeout_s():g}s <em>/ request</em>"),
    ])


if WORKSPACE_VIEW == "Ask":
    ui.lifecycle(live=LIVE_MODE)
with st.expander("Runtime controls and model status"):
    _status_rail()


@st.cache_data(show_spinner=False, max_entries=256)
def _full_catalog_tokens_cached(scope_fingerprint: tuple) -> int:
    """Cost of the un-retrieved prompt block, for the grounding readout."""
    del scope_fingerprint
    block = retrieval.schema_catalog_for(
        "", con,
        allowed_tables=ACCESS.allowed_tables,
        denied_columns=ACCESS.denied_by_table,
    )
    return max(1, len(block) // 4)


def _full_catalog_tokens() -> int:
    return _full_catalog_tokens_cached(ACCESS.fingerprint)


@st.cache_data(show_spinner=False, max_entries=256)
def _catalog_by_domain_cached(scope_fingerprint: tuple) -> dict[str, list[str]]:
    """Every table this warehouse actually loaded, grouped by domain.

    Filtered against table_names(con) rather than trusting MANIFEST: a manifest
    entry whose source CSV is missing must not appear on the schema map as an
    unlit cell, because that would draw a table the retriever could never have
    selected and quietly inflate the denominator.
    """
    del scope_fingerprint
    loaded = set(table_names(con)) & set(ACCESS.allowed_tables)
    grouped: dict[str, list[str]] = {}
    for domain, table, _source, _description in MANIFEST:
        name = table_name(domain, table)
        if name in loaded:
            grouped.setdefault(domain, []).append(name)
    return grouped


def _catalog_by_domain() -> dict[str, list[str]]:
    return _catalog_by_domain_cached(ACCESS.fingerprint)


@st.cache_data(show_spinner=False, max_entries=256)
def _retrieval_bundle_cached(question: str, scope_fingerprint: tuple):
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
        del scope_fingerprint
        hits = [hit for hit in retrieval.retrieve_hybrid(question, con=con)
                if hit.table in ACCESS.allowed_tables]
        # Only the hybrid call is the RETRIEVE stage. The two rankings gathered
        # below are re-run purely so the panel can show the ranks RRF consumed,
        # and charging the pipeline for the display's own overhead would
        # overstate what the assistant pays by roughly 50%.
        hybrid_ms = 1000 * (time.perf_counter() - started)

        vector = {hit.table: rank for rank, hit
                  in enumerate(retrieval.retrieve(question, k=POOL, con=con), 1)
                  if hit.table in ACCESS.allowed_tables}
        keyword = {hit.table: rank for rank, hit
                   in enumerate(retrieval.retrieve_keyword(question, k=POOL, con=con), 1)
                   if hit.table in ACCESS.allowed_tables}
        tokens_used = max(1, len(retrieval.schema_catalog_for(
            question, con,
            allowed_tables=ACCESS.allowed_tables,
            denied_columns=ACCESS.denied_by_table,
        )) // 4)
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


def _retrieval_bundle(question: str):
    bundle = _retrieval_bundle_cached(question, ACCESS.fingerprint)
    if bundle is not None:
        # Re-derive this cheap field at the session boundary as a guard against
        # stale cached payloads from an older code version.
        vector, keyword = bundle["vector"], bundle["keyword"]
        bundle = {**bundle, "candidates": len(set(vector) | set(keyword))}
    return bundle


def _model_retrieval_bundle(entry: dict):
    """Build the readout from the exact context and table set the model saw.

    Prior-turn tables can be mandatory additions to a follow-up prompt. Those
    are appended with a neutral score so membership is visible without
    inventing a retrieval rank.
    """
    bundle = _retrieval_bundle(entry.get("retrieval_context") or entry["question"])
    if not bundle or not entry.get("tables"):
        return bundle
    wanted = set(entry["tables"])
    hits = [hit for hit in bundle["hits"] if hit.table in wanted]
    seen = {hit.table for hit in hits}
    if seen != wanted:
        corpus = {row["id"]: row for row in retrieval.build_corpus(con)}
        for name in entry["tables"]:
            if name in seen or name not in corpus:
                continue
            meta = corpus[name]["metadata"]
            hits.append(retrieval.RetrievedTable(
                table=name,
                domain=str(meta["domain"]),
                description=str(meta["description"]),
                score=0.0,
            ))
    return {
        **bundle,
        "hits": hits,
        "tokens_used": int(entry.get("schema_tokens") or bundle["tokens_used"]),
    }


def _show_grounding(bundle, *, tokens: bool = True) -> None:
    """Which tables the retriever selected for this question, why, and what it saved.

    `tokens=False` on the compiled path. The retrieval is identical and worth
    showing -- the same ranking decides which tables the planner may bind
    against -- but no prompt is assembled and no tokens are spent, so the panel
    must not report a schema budget it did not pay.
    """
    if not bundle:
        return
    # Hybrid is what schema_catalog_for() actually uses, so the readout shows
    # the ranking the model was really given - not a prettier one.
    ui.grounding(
        bundle["hits"],
        total_tables=TABLE_COUNT,
        tokens_used=bundle["tokens_used"] if tokens else 0,
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
        (f"at most {MAX_SQL_CHARS:,} characters", "query exceeds"),
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
    verb = (reason.split("forbidden keyword:", 1)[1].strip()
            if "forbidden keyword:" in reason else "")
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
    findings = verifier.check_sql(sql, question)
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

@st.cache_data(show_spinner=False, max_entries=256)
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

    Returns None rather than raising. Local embedding being unavailable must cost a
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


@st.cache_data(show_spinner=False, max_entries=256)
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


@st.cache_data(show_spinner=False, max_entries=256)
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


@st.cache_data(show_spinner=False, max_entries=256)
def _searchable_schema_cached(scope_fingerprint: tuple):
    """Every table and column, flattened once, for the sidebar's search box.

    Read off engine.semantics rather than the manifest, because the interesting
    thing about a column here is its ROLE — measure, dimension, key, date, flag
    — and the manifest does not know that. Nothing else in the app exposes what
    the layer inferred at the column level, which is a strange omission for the
    module the whole keyless engine is built on.
    """
    del scope_fingerprint
    if layer is None:
        return []
    rows = []
    for name, table in sorted(layer.tables.items()):
        if name not in ACCESS.allowed_tables:
            continue
        hidden = ACCESS.denied_by_table.get(name, frozenset())
        visible = [column for column in table.columns if column.name.lower() not in hidden]
        rows.append({
            "table": name,
            "domain": table.domain,
            "description": table.description,
            "rows": table.rows,
            "columns": [(c.name, c.role, c.values[:4]) for c in visible],
            "haystack": " ".join(
                [name, table.domain, table.description]
                + [c.name for c in visible]
                + [v for c in visible for v in c.values[:6]]
            ).lower(),
        })
    return rows


def _searchable_schema():
    return _searchable_schema_cached(ACCESS.fingerprint)


def _schema_browser(ranked: list[str] | None = None) -> None:
    """A tool where eleven paragraphs of prose used to be.

    The sidebar used to carry a domain card per domain — 11 blocks of text
    describing the data in general terms, which is documentation and not
    something you can act on. It could not tell you whether a `department`
    column existed, which table held it, or what values it takes, so the only
    way to find out what was askable was to ask and be refused.

    This searches names, descriptions, column names AND indexed values, so
    typing "denied" finds healthcare_fact_claims through a value in `status` —
    the same binding the compiler makes, exposed as a thing you can browse.
    """
    schema = _searchable_schema()
    if not schema:
        st.subheader("What you can ask about")
        for domain, blurb in DOMAINS.items():
            if domain not in ACCESS_DOMAINS:
                continue
            ui.domain_card(domain, blurb)
        return

    st.subheader("Browse the schema")
    needle = st.text_input(
        "Search tables, columns and values", key="schema_search",
        placeholder="denied, salary, channel…", label_visibility="collapsed",
    ).strip().lower()

    if needle:
        matches = [r for r in schema if needle in r["haystack"]]
        cap = f"{len(matches)} of {len(schema)} tables matching “{needle}”"
        limit = 8
    else:
        # With no search, show the tables THIS question retrieved rather than
        # the first four alphabetically. The alphabetical default opened on
        # aml_cases every time — an arbitrary table nobody asked about — while
        # the retriever had already worked out which ten were relevant. The
        # sidebar should answer "what is this answer standing on", and it has
        # the answer already.
        #
        # In RETRIEVAL ORDER, best first. Sorting these by name put the table
        # the answer came from below the fold.
        order = {name: i for i, name in enumerate(ranked or [])}
        by_name = {r["table"]: r for r in schema}
        matches = [by_name[n] for n in (ranked or []) if n in by_name] or schema
        cap = (f"{len(matches)} tables retrieved for this question, best first"
               if order else f"{len(schema)} tables — search to narrow")
        limit = 6 if order else 4
    st.caption(cap)
    for row in matches[:limit]:
        # The table name alone. Appending "· N rows" pushed the label onto two
        # lines in a 300px sidebar and broke the name mid-word
        # ("retail_cross_sell_recommend / ations"), which is worse than making
        # someone open it to see the count.
        with st.expander(row["table"]):
            st.caption(f"{row['rows']:,} rows"
                       + (f" · {row['description']}" if row["description"] else ""))
            ui.column_list(row["columns"], highlight=needle)
    if len(matches) > limit:
        st.caption(f"…and {len(matches) - limit} more. Search to narrow.")


def _render_sidebar(active_question: str | None) -> None:
    """The catalogue, with this question's selection lit.

    Rendered at the END of the script rather than the top, even though it lands
    in the sidebar either way: the map is only worth drawing once the active
    question is known, and in both modes that is decided further down the page.
    """
    grouped = _catalog_by_domain()
    transcript = st.session_state.get("transcript") or []
    latest = (transcript or [{}])[-1]
    latest_engine = latest.get("engine", "")
    # Certified definitions already contain their SQL, and a manual turn already
    # contains the person's SQL. Running retrieval just to paint the sidebar on
    # either path would report a stage the answer never used.
    uses_retrieval = bool(transcript) and latest_engine not in {"metric", "manual"}
    bundle = (_retrieval_bundle(active_question)
              if active_question and uses_retrieval else None)
    # ORDERED, not a set. The browser lists these in the order given, and a set
    # forced it back to alphabetical — so for "how many denied claims are
    # there?" the sidebar opened on aml_cases and aml_dim_entity while
    # healthcare_fact_claims, the table that actually produced the answer, sat
    # under "…and 4 more". The retriever already ranked them; throwing that
    # ranking away and re-sorting by name is how the most relevant table ends up
    # last.
    # Only when a question was actually ASKED. On the landing page
    # `active_question` falls back to whatever the accuracy-contract expander
    # has selected — a golden question sitting collapsed further down the page —
    # and the sidebar then announced "10 tables retrieved for this question"
    # about a question the reader had not asked and could not see.
    asked = bool(st.session_state.get("transcript"))
    ranked = [hit.table for hit in bundle["hits"]] if (bundle and asked) else []
    selected = set(ranked)

    with st.sidebar:
        st.caption(
            f"{WORKSPACE_VIEW} workspace · roles "
            f"{', '.join(sorted(PRINCIPAL.roles)) or 'none'}"
        )
        if transcript:
            if st.button(
                "Start a new conversation",
                width="stretch",
                help="Clear this session's answers and return to the starting questions",
            ):
                _reset_conversation()
                st.rerun()
        if latest_engine == "metric":
            destination = "used · certified SQL bypassed retrieval"
        elif latest_engine == "manual":
            destination = "used · manual SQL bypassed retrieval"
        else:
            destination = ("sent to the model" if LIVE_MODE
                           else "handed to the compiler")
        with st.expander("Query context", expanded=asked):
            ui.schema_map(
                {domain: [(table, table in selected) for table in tables]
                 for domain, tables in grouped.items()},
                retrieved=len(selected), total=TABLE_COUNT,
                destination=destination,
            )
        if WORKSPACE_VIEW == "Ask":
            _schema_browser(ranked)
        with st.expander("Model settings"):
            _render_key_control()
        with st.expander("Voice settings"):
            _render_voice_control()


def _clear_model_key() -> None:
    """Widget callback: runs before Streamlit recreates the keyed input."""
    st.session_state["byok"] = ""
    st.session_state["byok_input"] = ""
    st.session_state.pop("_assistant", None)
    st.session_state.pop("_assistant_key", None)
    st.session_state.turns = []
    st.session_state.transcript = []


def _render_key_control() -> None:
    """Bring your own key.

    The deployment holds no key of its own and should not: a key on a public URL
    is an unmetered spend surface, and the alternative this repo chose instead
    was to ship the keyless engine that now drives the chat box. But that left
    the model path unreachable for everyone, including anyone who has a key and
    wants to see the two engines answer the same question.

    So the visitor can supply one. It is stored in `st.session_state`, which is
    per-browser-session, in memory, and gone when the tab closes: it is never
    written to disk, never logged, and never sent anywhere except Anthropic's
    API by the same client `engine/assistant.py` already used. The input is
    `type="password"` so it does not sit on screen in a screen-share.
    """
    if providers.local_provider_configured() and not _session_key():
        st.caption(
            "Model: server-configured OpenAI-compatible local provider "
            f"(`{providers.configured_provider_name()}`)."
        )
        st.caption(
            "Questions and retrieved schema stay on the configured model endpoint; "
            "the same verifier, data policy, SQL guard, and executor remain in force."
        )
        return
    if demo_mode.has_api_key():
        st.caption("Model: configured by the server environment.")
        return
    st.markdown("**Use your own API key**")
    st.caption(
        "Optional. With a key the model writes the SQL instead of the compiler, "
        "and the same guard, verifier and executor run on what it writes. "
        "Held in this browser's Streamlit session in server memory, sent to "
        "Anthropic for model calls, and never written or logged by this app."
    )
    typed = st.text_input(
        "Anthropic API key", type="password", key="byok_input",
        placeholder="sk-ant-...", label_visibility="collapsed",
    )
    if st.session_state.get("byok"):
        st.button("Switch back to the keyless compiler", on_click=_clear_model_key)
        return
    if typed and typed != st.session_state.get("byok"):
        st.session_state["byok"] = typed
        # The transcript is cleared rather than carried across. The two engines
        # produce differently shaped entries and are drawn by different
        # renderers, so a mixed transcript would ask render_entry to draw a
        # planner turn -- and every field it reads for the attempt ledger and
        # the token line would be missing.
        st.session_state.turns = []
        st.session_state.transcript = []
        st.rerun()


VOICE_LANGUAGES = {
    "Auto detect": "",
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Spanish": "es",
    "French": "fr",
}
VOICE_OPTIONS = {
    "Cedar · composed": "cedar",
    "Marin · warm": "marin",
    "Coral · bright": "coral",
    "Sage · measured": "sage",
    "Onyx · grounded": "onyx",
    "Nova · clear": "nova",
}
LOCAL_VOICE_OPTIONS = {
    "Heart · warm": "af_heart",
    "Bella · expressive": "af_bella",
    "Sky · clear": "af_sky",
    "Adam · grounded": "am_adam",
}


# The in-process Piper model IS one voice, so offering a menu of four would be
# a control that does nothing — the worst kind, because it invites the reader
# to believe the other three exist.
IN_PROCESS_VOICE_LABELS = {
    "en_US-joe-medium": "Joe · grounded male",
    # Kept for operators upgrading an existing deployment that deliberately
    # pinned the previous voice. The public/default path is Joe.
    "en_US-lessac-low": "Lessac · neutral",
}


def _voice_options() -> dict[str, str]:
    label = _voice_label()
    if label == "local":
        _stt, model = _voice_models()
        voice_id = str(model).removeprefix("piper ").strip()
        friendly = IN_PROCESS_VOICE_LABELS.get(voice_id, f"{voice_id} · local")
        return {friendly: voice_id}
    return LOCAL_VOICE_OPTIONS if label == "self-hosted" else VOICE_OPTIONS


def _voice_ready() -> bool:
    return _voice_client() is not None


def _clear_voice_key() -> None:
    st.session_state["voice_byok"] = ""
    st.session_state["voice_key_input"] = ""
    st.session_state.pop("_voice_client", None)
    st.session_state.pop("_voice_key_fingerprint", None)


def local_voice_pinned() -> bool:
    """Whether the running speech models are the checksum-pinned ones.

    Asked of engine/local_voice.py rather than recomputed here: the interface
    must not be able to claim a control the engine is not applying, which is
    exactly the drift this caption had — it told every visitor both models were
    checksum-pinned while only the voice was.
    """
    try:
        from engine import local_voice

        return bool(local_voice.stt_is_pinned())
    except Exception:  # noqa: BLE001 - a caption must never break the sidebar
        return False


def _render_voice_control() -> None:
    """Voice provider settings, deliberately separate from the SQL model key."""
    st.markdown("**Voice input & playback**")
    if st.session_state.get("_voice_error"):
        st.error(st.session_state["_voice_error"])
    if _voice_label() == "local":
        stt_model, tts_model = _voice_models()
        stt_model_label = stt_model.replace("faster-whisper ", "")
        st.caption(
            "Voice runs **in this process** on open models — faster-whisper "
            f"`{stt_model_label}` transcribes and `{tts_model}` supplies the male "
            "voice. No API key, no account, and no second service: the recording "
            "and the answer text never leave the server. "
            + (
                "Both download once on first use and are verified against a "
                "checksum pinned in this release — the speech model to an "
                "immutable Hub revision, the voice to its two artifacts."
                if local_voice_pinned()
                else "The voice is checksum-pinned; the speech model has been "
                     "overridden with ASK_LOCAL_STT_MODEL, so it is fetched "
                     "unpinned and this release cannot vouch for its weights."
            )
        )
    elif voice.local_endpoint_configured():
        st.caption(
            "Free self-hosted voice is enabled: faster-whisper transcribes and "
            "Kokoro generates speech. Audio and answer text go only to the "
            "configured local Speaches service. No paid API key is required."
        )
        st.caption(f"Endpoint `{voice.base_url()}`")
    else:
        server_key = voice.server_api_key()
        if server_key:
            st.caption(
                "OpenAI voice is configured by the server. Recordings are sent to "
                "OpenAI for transcription; answer text is sent only when Listen is pressed."
            )
        else:
            st.caption(
                "Optional cloud fallback. Your recording and API key are sent to OpenAI "
                "for transcription; answer text is sent when you press Listen. This app "
                "does not write the key, recording, or generated audio to disk."
            )
            typed = st.text_input(
                "OpenAI API key", type="password", key="voice_key_input",
                placeholder="sk-...", label_visibility="collapsed",
            )
            if typed and typed != st.session_state.get("voice_byok"):
                st.session_state["voice_byok"] = typed
                st.session_state.pop("_voice_client", None)
                st.session_state.pop("_voice_key_fingerprint", None)
                st.rerun()
            if st.session_state.get("voice_byok"):
                st.button("Disable voice", on_click=_clear_voice_key)

    if not _voice_ready():
        return
    st.selectbox("Transcription language", tuple(VOICE_LANGUAGES),
                 key="voice_language", index=0)
    options = _voice_options()
    # A browser session can outlive a deployment. If its widget still holds the
    # previous Lessac label, Streamlit must not try to render a selection that
    # the newly deployed Joe-only option set no longer contains.
    if st.session_state.get("voice_name") not in options:
        st.session_state["voice_name"] = next(iter(options))
    st.selectbox("Answer voice", tuple(options),
                 key="voice_name", index=0)
    # On by default, because an assistant that has to be asked to speak every
    # time is not really talking. Off is a real option and has to be reachable:
    # the first spoken answer takes the process from 330 MB to 465 MB, and a
    # reader on a shared machine may simply not want sound.
    st.toggle("Speak answers automatically", key="voice_autospeak",
              value=_autospeak_on(),
              help="Answers are read aloud as they arrive. Turn this off to "
                   "keep a Listen control instead.")
    # Mirrored into a plain key so the choice survives the widget being
    # unrendered. See the note in _autospeak_on.
    st.session_state["_autospeak_pref"] = bool(st.session_state["voice_autospeak"])
    stt_model, tts_model = _voice_models()
    st.caption(f"STT `{stt_model}` · TTS `{tts_model}` · playback is AI-generated speech.")


def _voice_models() -> tuple[str, str]:
    """(stt, tts) as the RESOLVED engine names them.

    voice.configured_*_model() reads the environment, which describes the
    remote seam. When speech is running in this process those functions
    describe a provider nobody is talking to, so the readouts asked the client
    that actually answered.
    """
    client = _voice_client()
    if client is None:
        return voice.configured_stt_model(), voice.configured_tts_model()
    return (
        getattr(client, "stt_model", voice.configured_stt_model()),
        getattr(client, "tts_model", voice.configured_tts_model()),
    )


def _voice_language() -> str:
    return VOICE_LANGUAGES.get(st.session_state.get("voice_language", "Auto detect"), "")


def _voice_name() -> str:
    options = _voice_options()
    default = next(iter(options.values()))
    return options.get(st.session_state.get("voice_name", ""), default)


def _voice_question() -> str:
    """Record, transcribe, and confirm one question; never auto-run a transcript."""
    ready = _voice_ready()
    stt_model, tts_model = _voice_models()
    ui.voice_dock(ready=ready, stt_model=stt_model, tts_model=tts_model)
    if not ready:
        st.caption(
            "Speech is unavailable in this deployment. Install the voice extras "
            "for in-process open models, or configure an endpoint or key under "
            "Voice input & playback."
        )
        return ""

    if st.session_state.pop("_voice_reset_pending", False):
        old_revision = int(st.session_state.get("_voice_revision", 0))
        st.session_state.pop(f"voice_recording_{old_revision}", None)
        st.session_state["_voice_revision"] = old_revision + 1
        for key in ("voice_draft", "voice_last_digest", "voice_error", "voice_meta"):
            st.session_state.pop(key, None)

    revision = int(st.session_state.get("_voice_revision", 0))
    recording = st.audio_input(
        "Record a data question", sample_rate=16000,
        key=f"voice_recording_{revision}",
    )
    if recording is not None:
        raw = recording.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != st.session_state.get("voice_last_digest"):
            # Mark it before the network call. A failed provider request should
            # not repeat on every unrelated Streamlit rerun and spend again.
            st.session_state["voice_last_digest"] = digest
            st.session_state["voice_error"] = ""
            with st.spinner("Transcribing with the speech model..."):
                try:
                    transcript = _voice_client().transcribe(
                        raw,
                        filename=getattr(recording, "name", "question.wav"),
                        mime_type=getattr(recording, "type", "audio/wav"),
                        language=_voice_language(),
                    )
                    st.session_state["voice_draft"] = transcript.text[:MAX_QUESTION_CHARS]
                    st.session_state["voice_meta"] = (
                        f"{transcript.model} · {transcript.bytes_received / 1024:,.0f} KB"
                    )
                except voice.VoiceUnavailable as exc:
                    st.session_state["voice_error"] = str(exc)

    if st.session_state.get("voice_error"):
        st.error(st.session_state["voice_error"])
        if st.button("Retry transcription", key=f"voice_retry_{revision}"):
            st.session_state.pop("voice_last_digest", None)
            st.rerun()

    if not st.session_state.get("voice_draft"):
        st.caption("Record a short question. You will review the transcript before it runs.")
        return ""

    with st.form("voice_confirm"):
        st.text_input(
            "Review the transcript", key="voice_draft", max_chars=MAX_QUESTION_CHARS,
        )
        st.caption(st.session_state.get("voice_meta", "") +
                   " · Nothing reaches the query engine until you confirm.")
        submitted = st.form_submit_button("Ask this question", width="stretch")
    if submitted:
        question = str(st.session_state.get("voice_draft", "")).strip()[:MAX_QUESTION_CHARS]
        if question:
            st.session_state["_voice_reset_pending"] = True
            return question
    return ""


def _autospeak_on() -> bool:
    """Whether answers should speak themselves. On by default; a real toggle.

    Default-on is a deliberate cost decision, not an oversight. Speech is lazy
    — a session that never gets an answer never loads the model — but the first
    spoken answer takes the process from 330 MB to 465 MB and pays a one-time
    model load. Every visitor who asks anything now pays that, so the control
    to turn it off has to exist and has to be findable.
    """
    # Read from the MIRROR, not from the widget key.
    #
    # Streamlit drops a widget's session_state entry on any run where that
    # widget is not rendered, and the toggle lives in the sidebar's voice
    # panel — which several paths do not draw. Probed on a live session, the
    # key alternated ABSENT / True / ABSENT between runs. Reading the widget
    # key directly therefore meant a reader who turned speech OFF would have
    # it turn itself back on the moment the panel was not on screen, which is
    # the one thing a mute control must never do.
    return bool(st.session_state.get("_autospeak_pref", True))


def _is_latest_turn(index: int) -> bool:
    """Whether this transcript entry is the one the reader just created.

    Streamlit re-runs the whole script on every interaction, so every previous
    answer is re-rendered each time. Without this, touching any control would
    replay the entire conversation.
    """
    return index == len(st.session_state.get("transcript", [])) - 1


def _render_answer_audio(answer: str, index: int, *, namespace: str = "turn",
                         autospeak: bool = False) -> None:
    """Speak an answer, and keep the audio only for this browser session.

    `autospeak` is passed by the CALLER rather than decided here, and that is
    the whole design. Streamlit re-runs the entire script on every interaction,
    so a renderer that decided for itself would re-synthesize and re-play every
    previous answer each time the reader touched anything — and the worked-
    examples panel, which renders its default question even while collapsed,
    would make the page start talking on load with nobody having asked it to.

    So only the turn the reader just created asks to be spoken, and it is
    spoken exactly ONCE: `voice_played_` records that this audio has already
    autoplayed, and every later rerun renders the same player inert. The
    control stays on screen so it can be replayed deliberately.

    Autoplay is a request, not a guarantee. Browsers block audio that starts
    without a user gesture, so a click or a typed question plays, and a cold
    `?q=` deep link renders a player that waits. Nothing is claimed on screen
    that the browser might not do.
    """
    if not answer or not _voice_ready():
        return
    selected = _voice_name()
    _stt_model, resolved_tts_model = _voice_models()
    identity = hashlib.sha256(
        f"{resolved_tts_model}\0{selected}\0{answer}".encode("utf-8")
    ).hexdigest()[:16]
    audio_key = f"voice_audio_{identity}"
    error_key = f"voice_audio_error_{identity}"
    button_key = f"listen_{namespace}_{index}_{identity}"
    mime_key = f"voice_audio_mime_{identity}"
    played_key = f"voice_played_{identity}"

    def synthesize() -> None:
        with st.spinner("Reading the answer aloud…"):
            try:
                speech = _voice_client().synthesize(answer, voice=selected)
                st.session_state[audio_key] = speech.audio
                # The remote provider returns MP3 and the in-process voice
                # returns WAV. Hard-coding audio/mpeg played a RIFF file as
                # an MP3, which some browsers refuse outright and others
                # render as a player that never starts.
                st.session_state[mime_key] = speech.mime_type
                cached = [
                    key for key in st.session_state
                    if key.startswith("voice_audio_")
                    and not key.startswith("voice_audio_error_")
                ]
                for old_key in cached[:-3]:
                    st.session_state.pop(old_key, None)
                st.session_state.pop(error_key, None)
            except voice.VoiceUnavailable as exc:
                st.session_state[error_key] = str(exc)

    wants_speech = autospeak and _autospeak_on()
    if audio_key not in st.session_state and wants_speech:
        synthesize()
    if audio_key not in st.session_state:
        # Either speaking is switched off, or synthesis failed. Either way the
        # answer is on screen and listening stays one click away.
        if st.button("Listen to answer", key=button_key,
                     icon=":material/volume_up:"):
            synthesize()
    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
    if st.session_state.get(audio_key):
        play_now = wants_speech and not st.session_state.get(played_key)
        if play_now:
            st.session_state[played_key] = True
        st.audio(st.session_state[audio_key],
                 format=st.session_state.get(mime_key, "audio/mpeg"),
                 autoplay=play_now)
        client = _voice_client()
        model = getattr(client, "tts_model", voice.configured_tts_model())
        where = ("Synthesized in this process — the answer text never left the "
                 "server." if _voice_label() == "local" else
                 "Generated on demand and kept in session memory only.")
        st.caption(f"AI-generated voice · {model}. {where}")


def _audit_turn(entry: dict, *, engine: str, coverage=None, metric: str = "") -> None:
    """Persist one turn.

    Called from every path that answers, including the one where a person wrote
    the SQL themselves — a trail with a hole in it where the interesting queries
    go is not a trail. Wrapped so that an audit failure can never take an answer
    down: engine/audit.record already swallows its own I/O errors, and this
    catches everything else for the same reason.
    """
    if entry.get("policy_denied"):
        outcome = "blocked"
    elif entry.get("timed_out"):
        outcome = "timeout"
    elif entry.get("refused"):
        outcome = "blocked" if entry.get("guard_ok") is False else "refused"
    elif entry.get("error"):
        outcome = "error"
    else:
        outcome = "answered"
    timings = {name: entry[key] for name, key in (
        ("retrieve", "retrieve_ms"), ("plan", "plan_ms"), ("guard", "guard_ms"),
        ("verify", "verify_ms"), ("execute", "exec_ms")) if entry.get(key) is not None}
    usage = entry.get("usage") or {}
    try:
        audit.record(
            actor=_actor(),
            role=",".join(sorted(PRINCIPAL.roles)) or "none",
            engine=engine,
            question=entry.get("question", ""),
            sql=entry.get("sql", ""),
            outcome=outcome,
            refusal_kind=entry.get("refusal_kind", ""),
            reason=entry.get("reason", ""),
            tables=list(entry.get("tables") or [
                h.table for h in (entry.get("bundle") or {}).get("hits", [])
            ])[:20],
            row_count=0 if entry.get("rows") is None else int(len(entry["rows"])),
            truncated=bool(entry.get("truncated")),
            attempts=int(entry.get("attempts", 1) or 1),
            corrections=list(entry.get("corrections") or []),
            guard_ok=bool(entry.get("guard_ok", True)),
            guard_reason=entry.get("guard_reason", ""),
            verifier=[{"rule": getattr(f, "rule", ""), "severity": getattr(f, "severity", "")}
                      for f in (entry.get("findings") or [])],
            metric=metric,
            coverage=coverage,
            tokens_in=int(usage.get("input_tokens", 0) or 0),
            tokens_out=int(usage.get("output_tokens", 0) or 0),
            elapsed_ms=float(entry.get("elapsed_ms") or sum(timings.values())),
            timings=timings,
            timeout_stage=entry.get("timeout_stage", ""),
        )
    except Exception:  # noqa: BLE001 - an observer must never break the observed
        pass


def _timeout_turn(
    question: str,
    engine: str,
    exc: deadline.DeadlineExpired,
    *,
    entry: dict | None = None,
    coverage=None,
    metric: str = "",
) -> dict:
    """Turn a stage overrun into the same visible, auditable refusal shape."""
    timed_out = dict(entry or {})
    timed_out.update({
        "question": question,
        "engine": engine,
        "refused": True,
        "reason": str(exc),
        "refusal_kind": "request timeout",
        "answer": "",
        "timed_out": True,
        "error": str(exc),
        "timeout_stage": exc.stage,
        "attempts": int(timed_out.get("attempts", 1) or 1),
        "corrections": list(timed_out.get("corrections") or []),
        "findings": list(timed_out.get("findings") or []),
        "ran": bool(timed_out.get("ran", False)),
        "rows": None,
        "truncated": False,
        "usage": dict(timed_out.get("usage") or {}),
        "sql": str(timed_out.get("sql", "") or ""),
        "trace": dict(timed_out.get("trace") or {}),
    })
    _audit_turn(timed_out, engine=engine, coverage=coverage, metric=metric)
    return timed_out


def _metric_turn(question: str, metric, match_ms: float,
                 request_deadline=None) -> dict:
    """Run one exact governed definition through the ordinary safety boundary."""
    request_deadline = request_deadline or deadline.RequestDeadline.configured()
    entry = {
        "question": question,
        "engine": "metric",
        "metric": metric,
        "refused": False,
        "reason": "",
        "refusal_kind": "",
        "answer": "",
        "sql": metric.sql,
        "attempts": 1,
        "corrections": [],
        "findings": [],
        "ran": False,
        "rows": None,
        "truncated": False,
        "error": "",
        "usage": {},
        "metric_ms": match_ms,
        "bundle": None,
        "trace": {},
        "contract_match": False,
    }
    guard_started = time.perf_counter()
    ok, reason = validate_sql(metric.sql)
    entry["guard_ms"] = 1000 * (time.perf_counter() - guard_started)
    entry["guard_ok"] = ok
    entry["guard_reason"] = reason
    try:
        request_deadline.require("SQL guard")
    except deadline.DeadlineExpired as exc:
        return _timeout_turn(
            question, "metric", exc, entry=entry, metric=metric.name
        )
    if not ok:
        entry["refused"] = True
        entry["reason"] = f"{GUARD_BLOCK_PREFIX}: {reason}"
        entry["refusal_kind"] = "blocked by the guard"
        _audit_turn(entry, engine="metric", metric=metric.name)
        return entry

    exec_started = time.perf_counter()
    ran = run_query(con, metric.sql, access=ACCESS, deadline=request_deadline)
    entry["exec_ms"] = 1000 * (time.perf_counter() - exec_started)
    findings, verify_ms = _verify_now(metric.sql, ran, question)
    entry["findings"] = findings
    entry["verify_ms"] = verify_ms
    entry["ran"] = bool(ran.ok)
    entry["error"] = "" if ran.ok else ran.error
    try:
        request_deadline.require("verification")
    except deadline.DeadlineExpired as exc:
        return _timeout_turn(
            question, "metric", exc, entry=entry, metric=metric.name
        )
    entry["timed_out"] = bool(getattr(ran, "timed_out", False))
    entry["policy_denied"] = bool(getattr(ran, "policy_denied", False))
    certified = metric_layer.MetricAnswer(metric=metric, result=ran)
    if ran.ok and ran.rows:
        entry["rows"] = pd.DataFrame(ran.rows, columns=ran.columns)
        entry["truncated"] = bool(ran.truncated)
        entry["answer"] = certified.sentence
        entry["contract_match"] = certified.matches_contract
    elif ran.ok:
        entry["answer"] = "No rows matched."
    _audit_turn(entry, engine="metric", metric=metric.name)
    return entry


def _keyless_turn(question: str) -> dict:
    """Prefer an explicitly named certified metric; otherwise compile the schema."""
    request_deadline = deadline.RequestDeadline.configured()
    question = str(question or "").strip()
    if len(question) > MAX_QUESTION_CHARS:
        entry = {
            "question": question[:MAX_QUESTION_CHARS],
            "engine": "planner",
            "refused": True,
            "reason": (f"question exceeds the {MAX_QUESTION_CHARS}-character limit; "
                       "shorten it and ask again"),
            "refusal_kind": "question too long",
            "answer": "",
            "sql": "",
            "attempts": 1,
            "corrections": [],
            "findings": [],
            "ran": False,
            "rows": None,
            "truncated": False,
            "error": "",
            "usage": {},
            "bundle": None,
            "trace": {},
        }
        _audit_turn(entry, engine="plan")
        return entry
    started = time.perf_counter()
    metric = metric_layer.match_metric(question, metric_registry)
    match_ms = 1000 * (time.perf_counter() - started)
    if metric is not None:
        return _metric_turn(question, metric, match_ms, request_deadline)
    return _plan_turn(question, request_deadline)


def _plan_turn(question: str, request_deadline=None) -> dict:
    """Compile a question without a model, run it, and record what happened.

    Deliberately shaped like the live path's transcript entry, because the same
    renderer draws both. The planner gets no privileges for being
    deterministic: its SQL goes through the same `validate_sql`, the same
    `run_query`, and the same `Verifier` the model's SQL goes through, and the
    only difference the page shows is which cell of the pipeline strip lights.
    """
    request_deadline = request_deadline or deadline.RequestDeadline.configured()
    # Retrieval and planning are timed SEPARATELY, because they are separate
    # stages and the strip draws a cell for each. Timed together, the first
    # question in a container reported PLAN 501ms -- almost all of it the MiniLM
    # forward pass behind the schema index, none of it spent compiling anything.
    # A stage that bills another stage's clock is the same class of error as a
    # cell that lights on the wrong evidence.
    bundle = _retrieval_bundle(question)
    try:
        request_deadline.require("schema retrieval")
    except deadline.DeadlineExpired as exc:
        return _timeout_turn(
            question,
            "plan",
            exc,
            entry={
                "bundle": bundle,
                "retrieve_ms": bundle["ms"] if bundle else None,
            },
        )
    hits = [hit.table for hit in bundle["hits"]] if bundle else []
    started = time.perf_counter()
    result = planner.plan_question(question, layer, retrieved=hits)
    plan_ms = 1000 * (time.perf_counter() - started)

    entry = {
        "question": question,
        "engine": "planner",
        "refused": result.refused,
        "reason": result.reason,
        "refusal_kind": result.kind or "not compiled",
        "answer": "",
        "sql": result.sql,
        "attempts": 1,
        "corrections": [],
        "findings": [],
        "ran": False,
        "rows": None,
        "truncated": False,
        "error": "",
        "usage": {},
        "plan_ms": plan_ms,
        "retrieve_ms": bundle["ms"] if bundle else None,
        "bundle": bundle,
        "trace": _plan_trace_payload(result),
    }
    try:
        request_deadline.require("planning")
    except deadline.DeadlineExpired as exc:
        return _timeout_turn(
            question,
            "plan",
            exc,
            entry=entry,
            coverage=getattr(result.plan, "coverage", None),
        )
    if result.refused:
        _audit_turn(entry, engine="plan",
                    coverage=getattr(result.plan, "coverage", None))
        return entry

    guard_started = time.perf_counter()
    ok, reason = validate_sql(result.sql)
    entry["guard_ms"] = 1000 * (time.perf_counter() - guard_started)
    try:
        request_deadline.require("SQL guard")
    except deadline.DeadlineExpired as exc:
        return _timeout_turn(
            question, "plan", exc, entry=entry, coverage=result.plan.coverage
        )
    if not ok:
        # Has never happened: the compiler emits SELECT from a grammar that has
        # no way to write anything else. The branch exists because "it cannot
        # happen" is not a thing this repo lets a boundary assert about itself,
        # and the guard is the boundary whether or not the thing behind it is
        # trusted.
        entry["refused"] = True
        entry["guard_ok"] = False
        entry["guard_reason"] = reason
        entry["reason"] = f"{GUARD_BLOCK_PREFIX}: {reason}"
        entry["corrections"] = [f"{GUARD_BLOCK_PREFIX}: {reason}"]
        _audit_turn(entry, engine="plan", coverage=result.plan.coverage)
        return entry

    exec_started = time.perf_counter()
    ran = run_query(con, result.sql, access=ACCESS, deadline=request_deadline)
    entry["exec_ms"] = 1000 * (time.perf_counter() - exec_started)
    findings, verify_ms = _verify_now(result.sql, ran, question)
    entry["findings"] = findings
    entry["verify_ms"] = verify_ms
    entry["ran"] = bool(ran.ok)
    entry["error"] = "" if ran.ok else ran.error
    try:
        request_deadline.require("verification")
    except deadline.DeadlineExpired as exc:
        return _timeout_turn(
            question, "plan", exc, entry=entry, coverage=result.plan.coverage
        )
    if ran.ok and ran.rows:
        entry["rows"] = pd.DataFrame(ran.rows, columns=ran.columns)
        entry["truncated"] = bool(ran.truncated)
        entry["answer"] = _plan_headline(result.plan, ran)
    elif ran.ok:
        entry["answer"] = "No rows matched."
    entry["timed_out"] = bool(getattr(ran, "timed_out", False))
    entry["policy_denied"] = bool(getattr(ran, "policy_denied", False))
    _audit_turn(entry, engine="plan", coverage=result.plan.coverage)
    return entry


def _manual_turn(sql: str, origin: str = "") -> dict:
    """Run SQL a person typed, through exactly the same boundary.

    This is the feature that turns the page from a demonstration into a tool.
    The app's whole argument is "the SQL is shown next to the answer so anyone
    can audit it" — and until now auditing it was all you could do. If you read
    the query and saw that the compiler had picked `submitted_amount` where you
    wanted `paid_amount`, the interface's answer was: leave.

    Nothing is relaxed to allow it. A human is exactly as untrusted as the model
    and as the compiler: the same `validate_sql`, the same `Verifier`, the same
    capped, cursor-isolated executor. That is the point — the boundary was
    never about who was writing, so it costs nothing to let you write.
    """
    request_deadline = deadline.RequestDeadline.configured()
    entry = {
        "question": origin or "Edited query",
        "engine": "manual",
        "refused": False,
        "reason": "",
        "answer": "",
        "sql": sql.strip(),
        "findings": [],
        "ran": False,
        "rows": None,
        "truncated": False,
        "error": "",
        "trace": {},
    }
    started = time.perf_counter()
    ok, reason = validate_sql(entry["sql"])
    entry["guard_ms"] = 1000 * (time.perf_counter() - started)
    entry["guard_ok"] = ok
    entry["guard_reason"] = reason
    try:
        request_deadline.require("SQL guard")
    except deadline.DeadlineExpired as exc:
        return _timeout_turn(entry["question"], "manual", exc, entry=entry)
    if not ok:
        entry["refused"] = True
        entry["reason"] = f"{GUARD_BLOCK_PREFIX}: {reason}"
        entry["refusal_kind"] = "blocked by the guard"
        _audit_turn(entry, engine="manual")
        return entry

    exec_started = time.perf_counter()
    ran = run_query(con, entry["sql"], access=ACCESS, deadline=request_deadline)
    entry["exec_ms"] = 1000 * (time.perf_counter() - exec_started)
    findings, verify_ms = _verify_now(entry["sql"], ran, origin or "")
    entry["findings"] = findings
    entry["verify_ms"] = verify_ms
    entry["ran"] = bool(ran.ok)
    entry["error"] = "" if ran.ok else ran.error
    try:
        request_deadline.require("verification")
    except deadline.DeadlineExpired as exc:
        return _timeout_turn(entry["question"], "manual", exc, entry=entry)
    if ran.ok and ran.rows:
        entry["rows"] = pd.DataFrame(ran.rows, columns=ran.columns)
        entry["truncated"] = bool(ran.truncated)
        value = ran.rows[0][0]
        entry["answer"] = (_fmt(value) if len(ran.rows) == 1 and len(ran.rows[0]) == 1
                           else f"{ran.row_count} rows returned.")
    elif ran.ok:
        entry["answer"] = "No rows matched."
    entry["timed_out"] = bool(getattr(ran, "timed_out", False))
    entry["policy_denied"] = bool(getattr(ran, "policy_denied", False))
    _audit_turn(entry, engine="manual")
    return entry


def _sql_editor(entry, index: int) -> None:
    """The editor attached to one answer.

    Keyed on the transcript index so several turns can each carry their own
    draft without sharing state. Rendered inside an expander because it is an
    invitation, not an instruction — the answer above it is complete.
    """
    with st.expander("Edit this SQL and run it yourself"):
        st.caption(
            "Your query goes through the same read-only guard, the same "
            "structural verifier and the same 200-row cap as everything else on "
            "this page. Nothing is relaxed because a person typed it."
        )
        # A FORM, not a loose text_area plus a button. Streamlit commits a bare
        # text_area to session state on blur, and a button click in the same
        # interaction is processed against the value the server last saw — so
        # editing the SQL and hitting Run submitted the ORIGINAL query and the
        # handler correctly answered "that is the same query". A form batches
        # the field with its submit button, which is the whole reason forms
        # exist and the only version of this that is not a race.
        with st.form(key=f"sqlform_{index}", clear_on_submit=False):
            draft = st.text_area(
                "SQL", value=entry["sql"], height=170,
                key=f"sqlbox_{index}", label_visibility="collapsed",
                max_chars=MAX_SQL_CHARS,
            )
            submitted = st.form_submit_button("Run it", type="primary")
        if submitted:
            if not draft.strip():
                st.caption("Nothing to run.")
            else:
                st.session_state.transcript.append(
                    _manual_turn(draft, origin=entry["question"]))
                st.rerun()


def _plan_trace_payload(result) -> dict:
    """What ui.plan_trace needs, split out so the renderer stays presentational.

    The three word groups are the coverage arithmetic made visible: `bound` is
    what the plan used, `loose` is what nothing in 71 tables contains (excused
    from the denominator, not silently dropped), and `missed` is what the
    warehouse knows and this plan did not use — the only group that is a debt.
    """
    plan = result.plan
    return {
        "rationale": plan.rationale() if plan else [],
        "coverage": plan.coverage if plan else 0.0,
        "considered": result.considered,
        "bound": sorted(plan.explained) if plan else [],
        "missed": sorted(plan.unexplained - result.unbound) if plan else [],
        "loose": sorted(result.unbound),
        "refused": result.refused,
    }


def _humanise(name: str) -> str:
    """`customer_name` -> "customer", `avg_base_salary` -> "base salary".

    Column identifiers leak into the headline otherwise. "5 groups by
    customer_name, ordered desc" is the query restated in the query's own
    vocabulary, which is exactly the register a headline should not be in.
    """
    words = [w for w in re.split(r"[^a-zA-Z0-9]+", name) if w]
    drop = {"id", "name", "key", "code", "avg", "sum", "total", "n", "pct"}
    kept = [w for w in words if w.lower() not in drop] or words
    return " ".join(kept).lower()


def _plural(label: str) -> str:
    """"customer" -> "customers". Naive, and that is the right amount.

    The alternative was appending the word "groups" to whatever the dimension
    was called, which produced "the highest of 5 customer groups" — accurate
    about the SQL and a slightly wrong sentence about the data. These are
    customers.
    """
    if not label or label.endswith("s"):
        return label
    if label.endswith("y") and label[-2:-1] not in "aeiou":
        return label[:-1] + "ies"
    if label.endswith(("ch", "sh", "x", "z")):
        return label + "es"
    return label + "s"


def _fmt(value) -> str:
    """A number a person reads, not a float repr.

    The result grid was printing 100281.5534 next to 93041.791 — different
    precision on the same column, no thousands separators, in a headline whose
    job is to be read at a glance.
    """
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        if float(value).is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def _plan_headline(plan, ran) -> str:
    """One sentence, built from the plan rather than written about it.

    The composition moved to `engine/narrate.py`; this is the seam. That module
    is Streamlit-free and depends only on engine.planner, so every sentence it
    can produce is testable without rendering a page — which is what let the
    shapes a typed question cannot reach (a truncated breakdown, a NULL group
    label, a fan-out COUNT) be driven from mutated plans rather than argued
    about.

    What it replaced fell off a cliff at exactly the most common shape. Grouped
    results read reasonably; every scalar — COUNT, SUM, AVG, MIN, MAX, SHARE —
    fell through to `return _fmt(value)`, so "how many denied claims are there?"
    was answered with the four characters `876`. Correct, and not an answer a
    person would give.

    Three defects came out of writing the replacement, all of them shipping:
      * "the highest of 5 departments" for `top 5 departments by headcount`,
        over a warehouse with TWELVE. A LIMIT tells you how many rows came
        back and nothing about how many groups exist.
      * `_plural("hour_of_day")` pluralised the tail — "24 hour of days".
      * `_plural("status")` short-circuited on the trailing s, and any NULL in
        a breakdown bailed to "3 status. The full breakdown is below." — the
        case where a sentence is most useful is the one it gave up on.
    """
    return narrate.answer_sentence(plan, ran)


def _numeric_format(frame):
    """Column config that formats float columns, and nothing else.

    Only floats: an integer count needs no decimals, and a key column that
    happens to be numeric must not gain thousands separators and stop matching
    the value in the SQL above it.
    """
    config = {}
    for column in frame.columns:
        if str(frame[column].dtype).startswith("float"):
            config[str(column)] = st.column_config.NumberColumn(format="localized")
    return config or None



EXAMPLES = [
    "Which payer type collects the least of what it bills?",
    "How many active employees do we have, and how many left voluntarily?",
    "What's the overall order fill rate?",
    "Who is the top wholesale customer by revenue?",
    "How many migration artifacts passed parallel-run validation?",
]

# Separate from EXAMPLES on purpose. Those five were chosen to show the MODEL
# off, and four of the five are questions the compiler correctly refuses —
# offering them in keyless mode would be handing a visitor a row of buttons that
# mostly answer "I can't". These five are inside the grammar, and every one of
# them is in evals/planner_questions.yaml with reference SQL under test, so this
# list cannot drift into promising something the planner stopped doing.
PLANNER_EXAMPLES = [
    "How many denied claims are there?",
    "What is the average salary by department?",
    "Total paid amount by payer type",
    "Which department has the highest average tenure?",
    "What percentage of transactions are cash?",
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
        key="contract_choice",
        help="Runs the reference SQL for this question against the warehouse now.",
    )
    active = labels[choice]
    # Speak this one only when the reader CHOSE it. This panel renders its
    # default question on every run — including while it is still collapsed and
    # nobody has opened it — so autospeaking whatever `index=0` happens to be
    # would make the page start talking on load. Comparing against the previous
    # run's selection is what separates "the reader picked this" from "Streamlit
    # re-ran the script".
    # The FIRST render records the default without speaking it — an absent key
    # means "this panel has not been looked at", not "the reader chose this".
    seen_before = "_contract_spoken" in st.session_state
    picked = seen_before and st.session_state["_contract_spoken"] != active["id"]
    st.session_state["_contract_spoken"] = active["id"]
    bundle = _retrieval_bundle(active["question"])

    # Timed separately from execution so each pipeline cell reports its own
    # stage. The guard is pure and re-running it costs nothing, which is what
    # makes an honest measurement of it possible at all.
    guard_started = time.perf_counter()
    validate_sql(active["sql"])
    guard_ms = 1000 * (time.perf_counter() - guard_started)

    exec_started = time.perf_counter()
    result = demo_mode.answer(connection, active, access=ACCESS)
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
            # The SENTENCE, not the bare scalar. `result.headline` is still the
            # value CI asserts and it is still what the amber badge vouches
            # for — it is now substituted into a sentence committed beside the
            # reference SQL, so a worked example reads like an answer instead
            # of like a cell. The value is never typed into the template.
            ui.answer(
                result.sentence,
                verified=result.matches_contract,
                verified_note=("Asserted in evals/golden_questions.yaml "
                               "and re-checked by CI on every push."),
            )
            # Worked examples get the same voice every other answer has. They
            # are the questions a first-time visitor actually clicks, so they
            # were the worst place in the app to have speech missing.
            _render_answer_audio(result.sentence, abs(hash(active["id"])) % 10_000,
                                 namespace="contract", autospeak=picked)
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
            st.code(result.sql, language="sql", wrap_lines=True)
            frame = pd.DataFrame(result.result.rows, columns=result.result.columns)
            _result_readout(result.sql, frame, truncated=result.result.truncated)
            st.dataframe(frame, width="stretch", hide_index=True)
            st.caption(
                "Reference SQL, executed live — not written by the model. "
                "The model-authored path is what the API key unlocks."
            )

    return active["question"]


@st.cache_data(show_spinner=False, max_entries=256)
def _nearest_vocabulary(words: tuple[str, ...]) -> list[tuple[str, str]]:
    """For words the compiler could not place, what the warehouse DOES have.

    A refusal that names the words it could not bind is honest and, on its own,
    a dead end: the reader is told `lift` meant nothing here and left to guess
    what would. The semantic layer already holds every column name and every
    indexed value, so the nearest few are a real answer to "then what CAN I
    ask", and they cost one pass over a dict.

    Matched on shared prefixes rather than by edit distance. The failures worth
    catching are "revenue" against `total_revenue` and "employee" against
    `employee_id` — same stem, different word — and a prefix test finds those
    without pulling in a similarity library for one panel.
    """
    if layer is None or not words:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for word in words:
        if len(word) < 4:
            continue
        stem = word[:4]
        for table in layer.tables.values():
            for column in table.columns:
                if column.name in seen:
                    continue
                if any(part.startswith(stem) for part in column.words):
                    seen.add(column.name)
                    out.append((f"{table.name}.{column.name}", column.role))
                    break
            if len(out) >= 6:
                return out
    return out


def _refusal_help(entry) -> None:
    """Give a refused question somewhere to go.

    Three things a person can act on: the columns whose names are closest to
    the words that failed, a reminder that the SQL box below takes anything,
    and the questions this compiler is known to answer.
    """
    trace = entry.get("trace") or {}
    stuck = tuple(sorted(set(trace.get("missed", []) + trace.get("loose", []))))[:4]
    near = _nearest_vocabulary(stuck)
    if near:
        st.markdown("**Closest things this warehouse actually has**")
        st.markdown("\n".join(
            f"- `{name}` — {role}" for name, role in near))
    # Only when there IS an answered turn to open. On a first-question refusal
    # this told the reader to use an editor on "any answered question above",
    # and there were none — advice that cannot be followed is worse than none.
    answered = any(not e.get("refused") for e in st.session_state.get("transcript", []))
    if answered:
        st.markdown(
            "You can also **write the SQL yourself** — open any answered "
            "question above and use its editor."
        )
    else:
        st.markdown(
            "Ask one of the example questions and every answer comes with an "
            "editor, so you can start from working SQL and change it."
        )


CHART_MAX_BARS = 20


def _chart_shape(frame):
    """Whether this result IS a shape, and which one — or None.

    Deliberately conservative, and the conservatism is the design. A chart is
    the most prominent thing on a turn, so it has to be a restatement of the
    result rather than an interpretation of it: one label column, one numeric
    column, more than one row. Anything else — two measures, three dimensions,
    a single scalar — is a grid, and the grid was never the problem.

    A single row gets nothing. `ui.answer` above already says the number in a
    sentence, and one bar at 100% of itself is a rectangle, not a comparison.
    """
    if frame is None or frame.empty or len(frame.columns) != 2 or len(frame) < 2:
        return None
    label_col, value_col = frame.columns[0], frame.columns[1]
    values = pd.to_numeric(frame[value_col], errors="coerce")
    if values.isna().any():
        return None
    if pd.api.types.is_numeric_dtype(frame[label_col]):
        return None
    # A date axis is a fact about the column, not a preference: a series over
    # time reads as a line and a ranking reads as bars, and drawing a time
    # series as a sorted bar chart hides the one thing it is for.
    kind = "line" if pd.api.types.is_datetime64_any_dtype(frame[label_col]) else "bar"
    return label_col, value_col, kind


def _result_chart(entry) -> None:
    shape = _chart_shape(entry["rows"])
    if shape is None:
        return
    frame = entry["rows"]
    label_col, value_col, kind = shape
    # The query's own ORDER BY is the ranking the reader asked for, so a query
    # that HAS one is drawn exactly as it came back: re-sorting a top-N would
    # draw the answer to a query nobody ran, and re-sorting a time series would
    # scramble the axis outright.
    #
    # A GROUP BY with no ORDER BY has no order at all — DuckDB returns those
    # groups however it likes, and this repo has already been bitten once by
    # treating that arbitrary order as meaningful. Drawing it unsorted puts
    # Grocery above Electronics for no reason a reader can see. Sorting is
    # therefore not an interpretation here; it is the only defensible reading
    # of a result set the query left unordered.
    ordered = "order by" in " ".join(entry.get("sql", "").lower().split())
    if not ordered and kind != "line":
        frame = frame.sort_values(value_col, ascending=False)
    head = frame.head(CHART_MAX_BARS)
    pairs = [(str(a), float(b)) for a, b in zip(head[label_col], head[value_col],
                                                strict=False)]
    ui.result_chart(pairs, label=_humanise(str(label_col)),
                    measure=_humanise(str(value_col)), kind=kind,
                    truncated_from=len(frame) if len(frame) > CHART_MAX_BARS else 0)


def _result_block(entry, index: int) -> None:
    """The rows, how to take them away, and how to change the query.

    Export is not a nicety here. An answer you cannot leave with is a
    demonstration of an answer; the CSV is what makes this a place you get work
    done rather than a place you watch work being done.
    """
    if entry["rows"] is None:
        if entry.get("timed_out"):
            # A different fact from a broken query, and it leads somewhere
            # different: narrow the question rather than fix a word.
            st.warning(entry["error"], icon=":material/timer_off:")
        elif entry["error"]:
            st.error(f"Query error: {entry['error']}")
        return
    _result_chart(entry)
    st.dataframe(entry["rows"], width="stretch", hide_index=True,
                 column_config=_numeric_format(entry["rows"]))
    left, right = st.columns([1, 3])
    left.download_button(
        "Download CSV",
        data=entry["rows"].to_csv(index=False).encode("utf-8"),
        file_name="ask-your-data.csv", mime="text/csv",
        key=f"csv_{index}", width="stretch",
    )
    if entry.get("engine") != "manual":
        right.caption(
            "Share this exact question: append "
            f"`?q={quote_plus(entry['question'])}` to the app URL."
        )


def _evidence_block(entry, trace: dict) -> None:
    """Everything that argues the answer, one click below it.

    These panels used to be open on every turn, and together they ran about
    1,400px — so reading a number meant scrolling past the proof of the number,
    every time, including the twentieth time. They are the reason to trust the
    app and they are not the reason to open it.

    The pipeline strip, the answer, the binding trace and the SQL stay above:
    the strip is one line, and the other three ARE the product. Retrieval, the
    guard, the verifier and the physical plan move in here.
    """
    with st.expander("Evidence — retrieval, guard, verifier, query plan"):
        _show_grounding(entry.get("bundle"), tokens=False)
        _guard_readout(entry["sql"])
        _verification_readout(entry["findings"], verify_ms=entry.get("verify_ms"))
        if entry["rows"] is not None:
            _result_readout(entry["sql"], entry["rows"],
                            truncated=bool(entry["truncated"]))


def render_manual_entry(entry, index: int) -> None:
    """A query a person wrote, reported exactly as strictly as a compiled one."""
    st.chat_message("user").write(f"↳ {entry['sql'].splitlines()[0][:90]}…"
                                  if len(entry["sql"].splitlines()) > 1
                                  else f"↳ {entry['sql']}")
    timings = {k: entry[v] for k, v in
               (("guard", "guard_ms"), ("verify", "verify_ms"), ("execute", "exec_ms"))
               if entry.get(v) is not None}
    with st.chat_message("assistant"):
        if entry["refused"]:
            # GUARD lights in the alert colour and EXECUTE stays dark, which is
            # the true shape of this turn: the statement reached the boundary
            # and stopped there.
            ui.pipeline(retrieved=False, planned=False, guarded="fail",
                        executed=False, timings=timings)
            ui.refusal(entry["reason"], kind=entry.get("refusal_kind", "blocked"))
            st.caption("Your query was checked before the database saw it, which "
                       "is the same thing that happens to the model's SQL.")
            return
        ui.pipeline(retrieved=False, planned=False, verified=True, guarded=True,
                    executed=True if entry["ran"] else "fail", timings=timings)
        if entry["answer"]:
            ui.answer(entry["answer"])
            _render_answer_audio(entry["answer"], index, namespace="manual",
                                 autospeak=_is_latest_turn(index))
        st.code(entry["sql"], language="sql", wrap_lines=True)
        _evidence_block(entry, {})
        _result_block(entry, index)
        _sql_editor(entry, index)
        st.caption("You wrote this one. Same guard, same verifier, same row cap.")


def render_metric_entry(entry, index: int) -> None:
    """A policy-owned definition, visually and operationally distinct from inference."""
    metric = entry["metric"]
    st.chat_message("user").write(entry["question"])
    timings = {label: entry[key] for label, key in (
        ("metric", "metric_ms"), ("guard", "guard_ms"),
        ("verify", "verify_ms"), ("execute", "exec_ms"),
    ) if entry.get(key) is not None}
    with st.chat_message("assistant"):
        if entry["refused"]:
            ui.pipeline(certified="fail", verified=False, guarded="fail",
                        executed=False, timings=timings)
            ui.refusal(entry["reason"], kind=entry.get("refusal_kind", "blocked"))
            return
        ui.pipeline(certified=True, verified=True, guarded=True,
                    executed=True if entry["ran"] else "fail", timings=timings)
        if entry["answer"]:
            ui.answer(
                entry["answer"], verified=bool(entry.get("contract_match")),
                verified_note=(f"Expected {metric.expect} in metrics.yaml; the committed "
                               "definition produced the same value live."),
            )
            _render_answer_audio(entry["answer"], index, namespace="metric",
                                 autospeak=_is_latest_turn(index))
        if entry["ran"] and not entry.get("contract_match"):
            st.error(
                f"Definition drift: metrics.yaml expects {metric.expect}, but the live "
                "warehouse returned a different value. The SQL is shown below."
            )
        ui.metric_definition(
            label=metric.label, owner=metric.owner, definition=metric.definition,
            derived_value=metric.derived_value, derived_why=metric.derived_why,
        )
        st.code(entry["sql"], language="sql", wrap_lines=True)
        _evidence_block(entry, {})
        _result_block(entry, index)
        _sql_editor(entry, index)
        st.caption(
            f"Certified definition `{metric.name}` · owner: {metric.owner}. Exact phrase "
            "matching only; qualified questions fall back to the compiler rather than "
            "silently dropping a filter or breakdown."
        )


def render_plan_entry(entry, index: int) -> None:
    """One keyless turn: what the compiler bound, and what the database returned.

    A separate renderer from the live path rather than a flag on it. The two
    turns genuinely differ in what exists to show — there is no retry ledger
    because there is no retry, no token spend because no tokens were spent, and
    a binding trace that the model path cannot produce — and a single renderer
    threading `if planner` through six branches would be harder to read than
    two, while quietly inviting exactly the kind of cell that lights on the
    wrong evidence.
    """
    st.chat_message("user").write(entry["question"])
    trace = entry.get("trace") or {}
    timings = {}
    for label, key in (("retrieve", "retrieve_ms"), ("plan", "plan_ms"),
                       ("guard", "guard_ms"), ("verify", "verify_ms"),
                       ("execute", "exec_ms")):
        if entry.get(key) is not None:
            timings[label] = entry[key]

    with st.chat_message("assistant"):
        if entry["refused"]:
            ui.pipeline(retrieved=True, planned="fail", guarded=False,
                        executed=False, timings=timings)
            # The kind comes from engine.planner, which knows why it stopped.
            # Deriving it here from "did a Plan object get built" labelled the
            # median refusal "nothing to bind", when the warehouse binds salary
            # fine and it is the grammar that has no median in it.
            ui.refusal(entry["reason"], kind=entry.get("refusal_kind", "not compiled"))
            if trace.get("rationale") or trace.get("loose") or trace.get("missed"):
                ui.plan_trace(trace["rationale"], coverage=trace["coverage"],
                              considered=trace["considered"], bound=trace["bound"],
                              missed=trace["missed"], loose=trace["loose"],
                              plan_ms=entry.get("plan_ms"), refused=True)
            _refusal_help(entry)
            st.caption(
                "A refusal here is the compiler working, not failing. It answers "
                "what it can bind to columns and values, and says so when it "
                "cannot — an API key puts the model on this box, and the model "
                "is what resolves the questions this grammar cannot."
            )
            return

        ui.pipeline(retrieved=True, planned=True, verified=True, guarded=True,
                    executed=True if entry["ran"] else "fail", timings=timings)
        if entry["answer"]:
            ui.answer(entry["answer"])
            _render_answer_audio(entry["answer"], index, namespace="plan",
                                 autospeak=_is_latest_turn(index))
        elif entry["error"]:
            st.error(f"Query error: {entry['error']}")
        ui.plan_trace(trace["rationale"], coverage=trace["coverage"],
                      considered=trace["considered"], bound=trace["bound"],
                      missed=trace["missed"], loose=trace["loose"],
                      plan_ms=entry.get("plan_ms"))
        # wrap_lines, not the default horizontal scroll. Measured at 1280 with
        # the sidebar open: the content column is 752px and a joined query is
        # 930px, so 178px of SQL sat off-screen behind a scrollbar. "The SQL is
        # shown next to the answer so anyone can audit it" is this project's
        # central claim, and a claim you have to drag sideways to check is a
        # weaker version of it.
        st.code(entry["sql"], language="sql", wrap_lines=True)
        _evidence_block(entry, trace)
        _result_block(entry, index)
        _sql_editor(entry, index)
        st.caption(
            "Compiled from the schema by engine/planner.py — no model, no API "
            "key, no cost. Every clause above traces to a word in the question."
        )


@st.cache_data(show_spinner=False, max_entries=256)
def _layer_cells():
    """The semantic layer's counts, read off the layer itself.

    Cached because it is constant for the container, and derived rather than
    written down: change what engine/semantics.py infers and this panel moves
    with it. If the layer failed to build there is nothing true to show, so it
    shows nothing.
    """
    if layer is None:
        return None
    summary = layer.summary()
    return [
        (f"{summary['role_measure']:,}", "measures",
         "numeric columns it can aggregate"),
        (f"{summary['role_dimension']:,}", "dimensions",
         "columns it can group by"),
        (f"{summary['joins']:,}", "join edges",
         "inferred keys, none crossing a domain"),
        (f"{summary['value_phrases']:,}", "value phrases",
         "words from the data itself"),
    ]


def _show_layer_summary() -> None:
    cells = _layer_cells()
    if not cells:
        return
    ui.layer_summary(
        cells,
        footnote=("All four are probed from DuckDB at startup. The separate certified "
                  "metric registry applies only when a question explicitly names one "
                  "of its governed phrases; the compiler is still evaluated with that "
                  "registry switched off."),
    )


def _operations_panel() -> None:
    """The service view, one expander down from the answers.

    This is the panel a buyer asks for and a demo never has: not "what did this
    turn do" — the pipeline strip has said that on every turn since the first
    version — but "what has this process been doing". It could not exist before
    engine/audit.py, because there was nothing durable to aggregate: every
    number the app measured was rendered once and dropped.

    It is deliberately not on by default. The evidence is the reason to trust
    the app; it is not the reason to open it.
    """
    with st.expander("Operations — the audit record of this session"):
        st.caption(
            "Every number here is computed from `engine/audit.py` records that "
            "were really written by the turns above. Nothing on this panel is a "
            "sample or a placeholder, and the limits of the trail are listed "
            "with it rather than left for a reader to discover."
        )
        ui.operations(
            audit.summarise(actor=_actor()),
            limits=audit.describe_limits(authenticated=PRINCIPAL.authenticated),
        )


def _reset_conversation() -> None:
    st.session_state.turns = []
    st.session_state.transcript = []
    # A ?q= link put the first question there. Without clearing the guard the
    # deep link would fire again on the very next run and re-ask the question
    # the reader just cleared.
    st.session_state["_link_used"] = True
    try:
        st.query_params.clear()
    except Exception:
        pass


def _transcript_header() -> None:
    """One line above the answers: how many, and how to get back.

    The only route back to the example questions used to be "Start a new
    conversation", at the BOTTOM of the sidebar, under the schema map, the
    schema browser and the API-key box. Streamlit collapses that sidebar by
    default on a phone, so on the device most people open a shared link with,
    there was no way back at all — the examples vanished on the first click and
    never returned.
    """
    count = len(st.session_state.transcript)
    left, right = st.columns([3, 1])
    left.caption(f"{count} question{'' if count == 1 else 's'} this session")
    if right.button("Start over", key="reset_top", width="stretch",
                    help="Clear these answers and show the example questions again"):
        _reset_conversation()
        st.rerun()


KEYLESS_NOTICE = (
    "**Local compiler active — no API key required.** Questions are bound to the "
    "governed schema, checked for meaning, gated read-only, and executed with the "
    "same limits as model-authored SQL."
)

KEYLESS_DETAILS = (
    "`engine/planner.py` reads column roles, the join graph, and approved dimension "
    "values directly from the warehouse. It binds the words in a question to that "
    "semantic layer and emits SQL without sending data to a model. Explicitly named "
    "governed metrics use committed definitions whose values CI re-checks. If the "
    "compiler cannot bind a request safely, it refuses and identifies the unmatched "
    "terms. Run `python scripts/run_planner_eval.py` to reproduce its measured contract."
)


def _link_question() -> str:
    """A question passed in the URL as `?q=...`, asked once on first load.

    Deep-linking a question is worth having on its own -- "here is the exact
    thing I meant, click it" is how anyone shares a result -- and it is also
    what makes a screenshot of this app reproducible instead of a thing someone
    typed by hand once.

    Asked ONCE per session, not once per rerun: Streamlit re-executes the whole
    script on every widget interaction, and a query param that survives that
    would re-ask its question after every click, appending a duplicate turn to
    the transcript each time.
    """
    if st.session_state.get("_link_used"):
        return ""
    try:
        raw = st.query_params.get("q", "")
    except Exception:
        return ""
    st.session_state["_link_used"] = True
    return str(raw or "").strip()[:400]


def _catalog_workspace() -> None:
    """Searchable, policy-filtered catalog for understanding what can be asked."""
    ui.view_header(
        "Governed data",
        "Find the table before you ask the question.",
        "Search table names, descriptions, columns, and approved sample values. "
        "Everything shown here is filtered through this session's access policy.",
    )
    schema = _searchable_schema()
    if not schema:
        st.warning("No catalog entries are visible to this access scope.")
        return

    search_col, domain_col = st.columns([3, 1])
    needle = search_col.text_input(
        "Search the data catalog",
        placeholder="Try: denied, salary, channel, supplier…",
        key="catalog_workspace_search",
    ).strip().lower()
    domains = ("All domains",) + tuple(sorted({row["domain"] for row in schema}))
    chosen_domain = domain_col.selectbox("Domain", domains, key="catalog_workspace_domain")
    matches = [
        row for row in schema
        if (not needle or needle in row["haystack"])
        and (chosen_domain == "All domains" or row["domain"] == chosen_domain)
    ]

    st.caption(
        f"{len(matches)} of {len(schema)} authorized tables"
        + (f" matching “{needle}”" if needle else "")
    )
    frame = pd.DataFrame(
        [
            {
                "Table": row["table"],
                "Domain": row["domain"],
                "Rows": row["rows"],
                "Columns": len(row["columns"]),
                "Description": row["description"],
            }
            for row in matches
        ]
    )
    if frame.empty:
        st.info(
            "No authorized table matches those filters. Clear the search or "
            "choose another domain."
        )
        return
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Rows": st.column_config.NumberColumn(format="localized"),
            "Columns": st.column_config.NumberColumn(format="%d"),
        },
    )

    selected = st.selectbox(
        "Inspect columns and approved values",
        tuple(row["table"] for row in matches),
        key="catalog_workspace_table",
    )
    detail = next(row for row in matches if row["table"] == selected)
    st.caption(
        f"{detail['domain']} domain · {detail['rows']:,} rows"
        + (f" · {detail['description']}" if detail["description"] else "")
    )
    ui.column_list(detail["columns"], highlight=needle)


def _trust_workspace(connection) -> str | None:
    """Identity, controls, operations, and reproducible accuracy in one place."""
    ui.view_header(
        "Trust center",
        "Inspect the controls behind every answer.",
        "Current access scope, execution boundaries, live session telemetry, and "
        "the committed accuracy contract are collected here for review.",
    )
    protected = sum(len(columns) for columns in ACCESS.denied_by_table.values())
    ui.layer_summary(
        [
            (
                str(len(PRINCIPAL.roles)),
                "active roles",
                ", ".join(sorted(PRINCIPAL.roles)) or "none",
            ),
            (f"{TABLE_COUNT:,}", "authorized tables", f"policy {ACCESS.policy_version}"),
            (f"{protected:,}", "masked columns", "removed before prompting and execution"),
            (f"{MAX_ROWS:,}", "row ceiling", "enforced for every query"),
        ],
        footnote=(
            f"Identity: {PRINCIPAL.subject} · authentication: "
            + (
                "verified OIDC"
                if PRINCIPAL.authenticated
                else "explicit synthetic demo session"
            )
            + " · "
            f"request deadline: {deadline.configured_timeout_s():g}s."
        ),
    )
    if PRINCIPAL.authenticated:
        st.success("OIDC identity verified and the role policy is active for this session.")
    else:
        st.info(
            "This local demo uses synthetic data and an explicitly unauthenticated demo role. "
            "Set ASK_AUTH_MODE=oidc and a policy file before serving real organizational data."
        )

    st.subheader("Session operations")
    st.caption(
        (
            "Turns for this verified subject in the current process appear here. "
            if PRINCIPAL.authenticated
            else "Only turns executed in this browser session appear here. "
        )
        + "Production deployments should export the same structured events to durable "
        "observability storage."
    )
    ui.operations(
        audit.summarise(actor=_actor()),
        limits=audit.describe_limits(authenticated=PRINCIPAL.authenticated),
    )

    st.subheader("Reproducible accuracy")
    st.caption(
        "These reference questions bypass both the model and compiler. Their committed SQL "
        "runs live, and CI checks that the expected result has not drifted."
    )
    with st.expander("Open the 39-question accuracy contract"):
        return render_demo_mode(connection)
    return None


def render_keyless(connection) -> None:
    """The keyless app: a working chat box, then the accuracy contract below it.

    This used to be a dropdown of 39 pre-registered questions with the chat
    input rendered `disabled=True` underneath. That was honest and it was also
    the wrong product: the public deployment is the only version most people
    ever touch, and it could not answer a question nobody had thought of in
    advance — the exact failure this project accuses dashboards of.
    """
    # No `icon=`. Streamlit renders a Material Symbols ligature, and the glyph
    # only appears once that font has loaded -- until then the icon's NAME sits
    # in the layout as literal text ("function"), overlapping the first line of
    # the notice. A banner whose job is to explain the engine should not open
    # with a stray word, and the icon was carrying no information anyway.
    if not PLANNER_READY:
        # Profiling the warehouse failed. Say so plainly and fall back to the
        # contract browser, which needs nothing but DuckDB.
        st.warning(
            "The compiler could not profile the warehouse on this container, so "
            "the chat box is unavailable. The accuracy contract below still "
            "runs: every question there executes committed reference SQL live "
            "against DuckDB.",
            icon=":material/warning:",
        )
        contract_question = render_demo_mode(connection)
        _render_sidebar(contract_question)
        return

    # The notice explains which engine is answering and why there is no key. It
    # is the right thing to lead with on arrival and the wrong thing to keep
    # leading with: measured at 1280, it is 140px of prose sitting between the
    # masthead and the answer on every turn after the first, and the answer was
    # already 973px down the page. Once a question has been asked the reader
    # has both read it and seen the engine named on the pipeline strip, so it
    # folds into an expander and stops being in the way.
    st.info(KEYLESS_NOTICE)
    if not st.session_state.transcript:
        with st.expander("How keyless mode works"):
            st.markdown(KEYLESS_DETAILS)

    link = _link_question()
    clicked = None
    if not st.session_state.transcript and not link:
        st.write("Try one of these, or type your own:")
        # The container carries .ayd-examples so app/ui.py can stretch the five
        # buttons to one height. Without it each sizes to its own text and the
        # row is ragged by up to two lines.
        with st.container(key="ayd-examples"):
            for column, example in zip(st.columns(len(PLANNER_EXAMPLES)),
                                       PLANNER_EXAMPLES, strict=True):
                if column.button(example, width="stretch"):
                    clicked = example
        _show_layer_summary()

    if st.session_state.transcript:
        _transcript_header()

    for index, entry in enumerate(st.session_state.transcript):
        if entry.get("engine") == "manual":
            render_manual_entry(entry, index)
        elif entry.get("engine") == "metric":
            render_metric_entry(entry, index)
        else:
            render_plan_entry(entry, index)

    voice_question = _voice_question()
    question = (st.chat_input(
                    "Ask a question about the data...", max_chars=MAX_QUESTION_CHARS)
                or voice_question or clicked or link)
    if question and _session_budget_exhausted():
        # The ceiling is per browser session and a new tab clears it. That is
        # said out loud rather than hidden, because a limit presented as a
        # security control is worse than no limit — somebody stops looking for
        # the real one, which is a spend cap on the key itself.
        st.warning(
            f"This session has reached its {MAX_QUESTIONS_PER_SESSION}-question "
            "ceiling. Reload the page to start a new session. The bound exists "
            "because a deployment with a key behind it is an unmetered spend "
            "surface — but it counts per browser session, so it is a courtesy "
            "limit and not a security control. The real ceiling is a spend cap "
            "on the key.")
        question = None
    if question:
        with st.spinner("Resolving the definition and compiling SQL..."):
            entry = _keyless_turn(question)
        _count_question()
        st.session_state.transcript.append(entry)
        st.rerun()

    st.divider()
    # The 39 pre-registered questions belong HERE as well as in Trust center,
    # and the workspace split is what took them away. They sat under the chat
    # box for most of this app's life; moving evidence out of the answer flow
    # was right for the guard readouts and the audit ledger, and wrong for
    # these, because they are not evidence about a turn — they are the fastest
    # way for a visitor to see the thing work on a question with a known
    # answer. Behind a pointer in another workspace they read as missing, and
    # a reader who has to be told where the worked examples went has already
    # lost the thread.
    #
    # Collapsed, so the answer flow above still owns the page, and rendered in
    # exactly one workspace per run — Ask and Trust center are mutually
    # exclusive, so the two copies never share a widget key.
    with st.expander(
        "Worked examples — 39 questions whose SQL is committed and re-run by CI"
    ):
        st.caption(
            "A different kind of evidence from the box above, and labelled as such: "
            "nothing here is written by the model OR by the compiler. Picking a "
            "question runs its committed reference SQL live and shows whether the "
            "result still matches the value CI asserts."
        )
        render_demo_mode(connection)
    st.caption(
        "Control evidence, session telemetry and access scope live under "
        "**Trust center** above."
    )
    last = (st.session_state.transcript[-1]["question"]
            if st.session_state.transcript else None)
    _render_sidebar(last)


if WORKSPACE_VIEW == "Data catalog":
    _catalog_workspace()
    _render_sidebar(None)
    st.stop()

if WORKSPACE_VIEW == "Trust center":
    trust_question = _trust_workspace(con)
    _render_sidebar(trust_question)
    st.stop()

if not LIVE_MODE:
    render_keyless(con)
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
        if col.button(ex, width="stretch"):
            clicked = ex


def render_entry(entry, index: int):
    bundle = _model_retrieval_bundle(entry)
    st.chat_message("user").write(entry["question"])
    with st.chat_message("assistant"):
        findings = entry.get("findings") or []
        if entry.get("provider_error"):
            ui.pipeline(
                retrieved=bool(bundle), generated="fail", verified=False,
                guarded=False, executed=False,
                timings={"retrieve": bundle["ms"]} if bundle else None,
            )
            _show_grounding(bundle)
            st.error(
                "The language-model provider was unavailable, so no SQL was generated "
                "and nothing ran. Check the configured endpoint, credentials, quota, "
                "and model availability, then retry."
            )
            if entry.get("elapsed_ms"):
                ui.note(f"Provider failed after {entry['elapsed_ms']:,.0f} ms.")
            return
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
                st.code(entry["sql"], language="sql", wrap_lines=True)
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
            _render_answer_audio(entry["answer"], index, namespace="model",
                                 autospeak=_is_latest_turn(index))
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
            st.code(entry["sql"], language="sql", wrap_lines=True)
            if entry["rows"] is not None:
                _result_readout(entry["sql"], entry["rows"],
                                truncated=bool(entry["truncated"]))
                st.dataframe(entry["rows"], width="stretch", hide_index=True)
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


for index, entry in enumerate(st.session_state.transcript):
    render_entry(entry, index)

voice_question = _voice_question()
question = (st.chat_input(
    "Ask a question about the data...", max_chars=MAX_QUESTION_CHARS,
) or voice_question or clicked)

if question and _session_budget_exhausted():
    st.warning(
        f"This session has reached its {MAX_QUESTIONS_PER_SESSION}-question ceiling. "
        "Start a new browser session to continue. This is a courtesy bound; the "
        "real cost control is a spend cap on the provider keys."
    )
    question = None

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Writing SQL and running it..."):
            started = time.perf_counter()
            try:
                result = assistant.ask(question, history=st.session_state.turns)
            except AssistantUnavailable:
                elapsed_ms = 1000 * (time.perf_counter() - started)
                entry = {
                    "question": question,
                    "refused": False,
                    "reason": "language-model provider unavailable",
                    "answer": "",
                    "sql": "",
                    "attempts": 1,
                    "corrections": [],
                    "findings": [],
                    "ran": False,
                    "rows": None,
                    "truncated": False,
                    "error": "language-model provider unavailable",
                    "provider_error": True,
                    "usage": {},
                    "elapsed_ms": elapsed_ms,
                    "tables": [],
                    "retrieval_context": question,
                    "schema_tokens": 0,
                }
                _audit_turn(entry, engine="model")
                _count_question()
                st.session_state.transcript.append(entry)
                st.rerun()
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
        "policy_denied": bool(result.result and result.result.policy_denied),
        "timed_out": bool(result.timed_out),
        "timeout_stage": result.timeout_stage,
        "usage": result.usage,
        "elapsed_ms": elapsed_ms,
        "tables": result.tables,
        "retrieval_context": result.retrieval_context,
        "schema_tokens": result.schema_tokens,
    }
    _audit_turn(entry, engine="model")
    _count_question()
    st.session_state.transcript.append(entry)
    if result.ok:
        st.session_state.turns.append(result.as_turn())
    st.rerun()

_render_sidebar(
    st.session_state.transcript[-1]["question"] if st.session_state.transcript else None
)
