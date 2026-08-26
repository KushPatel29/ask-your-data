"""
The instrument panel.

This app is not a chatbot and styling it as one would misrepresent it. A chat
bubble says "a personality is talking to you"; what is actually happening is a
question being grounded against a schema, turned into SQL, checked by a guard,
and executed against a warehouse. So the interface is built as a readout of that
pipeline: every turn shows which stages ran, what the retriever was allowed to
see, and what the guard decided.

TWO ACCENTS, AND THEY MEAN DIFFERENT THINGS
  cyan  (#22D3EE) - the machine's own work: retrieval, generated SQL, token counts.
  amber (#FBBF24) - verified against something committed: a golden-question
                    contract value that CI re-checks on every push.
Nothing is amber unless a file in the repo backs it. That distinction is the
whole argument of this project, so it gets the colour system rather than a
footnote.

Type is IBM Plex - Condensed for display, Sans for prose, Mono for anything the
machine produced. Plex was drawn for technical products and it reads as
instrumentation rather than as a brand; it is also not the typeface every other
LLM demo reaches for.

The signature element is `grounding()`: the retrieval readout. It shows the
tables the vector index selected out of the full catalogue, their cosine
similarity, and what that saved against pasting every table into the prompt.
It exists because engine/retrieval.py made those numbers real - see
scripts/run_retrieval_eval.py for how they were measured.
"""

from __future__ import annotations

import html

import streamlit as st

CYAN = "#22D3EE"
AMBER = "#FBBF24"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');

:root{
  --ayd-ground:#0A0C14; --ayd-panel:#12151F; --ayd-panel-2:#171B28;
  --ayd-line:#232838; --ayd-ink:#E6E9F2; --ayd-muted:#7C859C;
  --ayd-machine:#22D3EE; --ayd-verified:#FBBF24; --ayd-alert:#FB7185;
  --ayd-mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --ayd-sans:'IBM Plex Sans',system-ui,-apple-system,sans-serif;
  --ayd-cond:'IBM Plex Sans Condensed','IBM Plex Sans',system-ui,sans-serif;
}

html, body, [class*="st-"]{ font-family:var(--ayd-sans) !important; }
.stApp{ background:
  radial-gradient(1100px 520px at 78% -12%, rgba(34,211,238,.07), transparent 60%),
  var(--ayd-ground); }

/* ---- masthead ---------------------------------------------------------- */
.ayd-mast{ border-bottom:1px solid var(--ayd-line); padding:.2rem 0 1.1rem; margin-bottom:1.4rem; }
.ayd-kicker{ font-family:var(--ayd-mono) !important; font-size:.68rem; letter-spacing:.19em;
  text-transform:uppercase; color:var(--ayd-machine); display:flex; gap:.6rem; align-items:center; }
.ayd-kicker::after{ content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--ayd-line),transparent); }
/* Streamlit sets font-family on h1 with !important of its own, so a bare
   class loses even when it also declares !important. Element+class wins. */
.ayd-mast h1.ayd-title{ font-family:var(--ayd-cond) !important; font-weight:700; font-size:2.7rem; line-height:1.02;
  letter-spacing:-.015em; margin:.5rem 0 .4rem; color:var(--ayd-ink); }
.ayd-sub{ color:var(--ayd-muted); max-width:62ch; font-size:.95rem; line-height:1.55; margin:0; }
.ayd-stats{ display:flex; gap:1.6rem; flex-wrap:wrap; margin-top:1rem;
  font-family:var(--ayd-mono) !important; font-size:.72rem; color:var(--ayd-muted); }
.ayd-stats b{ color:var(--ayd-ink); font-weight:600; }

/* ---- pipeline strip ---------------------------------------------------- */
.ayd-pipe{ display:flex; gap:.4rem; flex-wrap:wrap; align-items:center;
  font-family:var(--ayd-mono) !important; font-size:.66rem; letter-spacing:.13em;
  text-transform:uppercase; margin:.1rem 0 .7rem; }
.ayd-step{ border:1px solid var(--ayd-line); border-radius:2px; padding:.2rem .5rem;
  color:var(--ayd-muted); background:var(--ayd-panel); }
.ayd-step[data-on="1"]{ color:var(--ayd-machine); border-color:rgba(34,211,238,.42);
  background:rgba(34,211,238,.07); }
.ayd-step[data-on="fail"]{ color:var(--ayd-alert); border-color:rgba(251,113,133,.45);
  background:rgba(251,113,133,.08); }
.ayd-arrow{ color:var(--ayd-line); }

/* ---- grounding readout (the signature) --------------------------------- */
.ayd-ground-panel{ border:1px solid var(--ayd-line); border-left:2px solid var(--ayd-machine);
  border-radius:3px; background:var(--ayd-panel); padding:.8rem .95rem; margin:.2rem 0 .9rem; }
.ayd-ground-head{ display:flex; justify-content:space-between; gap:1rem; align-items:baseline;
  font-family:var(--ayd-mono) !important; font-size:.68rem; letter-spacing:.13em; text-transform:uppercase;
  color:var(--ayd-muted); margin-bottom:.65rem; }
.ayd-ground-head b{ color:var(--ayd-ink); }
.ayd-row{ display:grid; grid-template-columns:minmax(0,1fr) 68px; gap:.7rem; align-items:center;
  font-family:var(--ayd-mono) !important; font-size:.75rem; padding:.16rem 0; }
.ayd-tbl{ color:var(--ayd-ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ayd-tbl i{ color:var(--ayd-muted); font-style:normal; }
.ayd-bar{ position:relative; height:5px; background:var(--ayd-panel-2); border-radius:1px; overflow:hidden; }
.ayd-bar span{ position:absolute; inset:0 auto 0 0; background:var(--ayd-machine); opacity:.75; }
.ayd-score{ color:var(--ayd-muted); text-align:right; font-size:.7rem; }
.ayd-saving{ margin-top:.6rem; padding-top:.55rem; border-top:1px solid var(--ayd-line);
  font-family:var(--ayd-mono) !important; font-size:.7rem; color:var(--ayd-muted); }
.ayd-saving b{ color:var(--ayd-machine); }

/* ---- answer ------------------------------------------------------------ */
.ayd-answer{ font-family:var(--ayd-cond) !important; font-weight:600; font-size:1.75rem; line-height:1.22;
  color:var(--ayd-ink); margin:.15rem 0 .45rem; letter-spacing:-.01em; }
.ayd-verified{ display:inline-flex; align-items:center; gap:.45rem; font-family:var(--ayd-mono) !important;
  font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--ayd-verified);
  border:1px solid rgba(251,191,36,.35); background:rgba(251,191,36,.07);
  border-radius:2px; padding:.2rem .5rem; }
.ayd-note{ font-family:var(--ayd-mono) !important; font-size:.72rem; color:var(--ayd-muted); }

/* ---- sidebar ----------------------------------------------------------- */
[data-testid="stSidebar"]{ border-right:1px solid var(--ayd-line); }
.ayd-dom{ border-bottom:1px solid var(--ayd-line); padding:.5rem 0; }
.ayd-dom b{ font-family:var(--ayd-mono) !important; font-size:.74rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ayd-machine); }
.ayd-dom p{ margin:.22rem 0 0; font-size:.8rem; color:var(--ayd-muted); line-height:1.45; }

/* Dense SQL and dataframes stay legible - the point of the app is reading them. */
.stCode, pre, code{ font-family:var(--ayd-mono) !important; font-size:.82rem !important; }
.stCode{ border:1px solid var(--ayd-line) !important; border-radius:3px !important; }

@media (prefers-reduced-motion: no-preference){
  .ayd-step[data-on="1"]{ animation:aydIn .28s ease-out both; }
  @keyframes aydIn{ from{ opacity:.35; transform:translateY(1px);} to{ opacity:1; transform:none;} }
}
@media (max-width:640px){
  .ayd-mast h1.ayd-title{ font-size:2rem; }
  .ayd-row{ grid-template-columns:minmax(0,1fr) 54px; }
}
</style>
"""


def inject() -> None:
    """Load the stylesheet once per session."""
    st.markdown(_CSS, unsafe_allow_html=True)


def masthead(*, tables: int, domains: int, live: bool) -> None:
    mode = "live · model-authored SQL" if live else "demo · committed reference SQL"
    st.markdown(
        f"""
<div class="ayd-mast">
  <div class="ayd-kicker">Ask your data</div>
  <h1 class="ayd-title">A question goes in.<br>SQL comes out, and you see all of it.</h1>
  <p class="ayd-sub">Natural language over the analytics warehouses from my portfolio projects.
  It retrieves the schema the question needs, writes DuckDB SQL, checks it read-only before running it,
  and shows the query next to the number.</p>
  <div class="ayd-stats">
    <span><b>{tables}</b> tables</span>
    <span><b>{domains}</b> domains</span>
    <span>{mode}</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def pipeline(*, retrieved: bool = False, generated: bool = False,
             guarded: bool | str = False, executed: bool = False,
             attempts: int = 1) -> None:
    """The stages this turn actually went through.

    Not decoration: each cell is lit from what really happened, so a refusal
    shows GENERATE lit and EXECUTE dark, and a guard rejection shows GUARD in
    the alert colour with the retry count beside it.
    """
    def cell(label: str, state) -> str:
        on = "fail" if state == "fail" else ("1" if state else "0")
        return f'<span class="ayd-step" data-on="{on}">{label}</span>'

    parts = [
        cell("retrieve", retrieved), '<span class="ayd-arrow">→</span>',
        cell("generate", generated), '<span class="ayd-arrow">→</span>',
        cell("guard", guarded), '<span class="ayd-arrow">→</span>',
        cell("execute", executed),
    ]
    if attempts > 1:
        parts.append(f'<span class="ayd-step" data-on="fail">retry ×{attempts - 1}</span>')
    st.markdown(f'<div class="ayd-pipe">{"".join(parts)}</div>', unsafe_allow_html=True)


def grounding(hits, *, total_tables: int, tokens_used: int, tokens_full: int) -> None:
    """What the model was allowed to see, and what that cost.

    The signature panel. `hits` are RetrievedTable records from engine.retrieval;
    the bar is cosine similarity, normalised to the top hit so the shape of the
    ranking is readable rather than every bar sitting near the middle.
    """
    if not hits:
        return
    top = max((h.score for h in hits), default=1.0) or 1.0
    rows = []
    for h in hits:
        width = max(4, min(100, round(100 * h.score / top)))
        table = html.escape(h.table)
        domain = html.escape(h.domain)
        rows.append(
            f'<div class="ayd-row">'
            f'<div class="ayd-tbl">{table} <i>· {domain}</i></div>'
            f'<div class="ayd-score">{h.score:.3f}</div>'
            f'</div>'
            f'<div class="ayd-bar"><span style="width:{width}%"></span></div>'
        )
    saved = max(0, tokens_full - tokens_used)
    pct = round(100 * saved / tokens_full) if tokens_full else 0
    st.markdown(
        f"""
<div class="ayd-ground-panel">
  <div class="ayd-ground-head">
    <span>grounding · <b>{len(hits)}</b> of {total_tables} tables retrieved</span>
    <span>cosine</span>
  </div>
  {''.join(rows)}
  <div class="ayd-saving">~<b>{tokens_used:,}</b> tokens of schema in the prompt,
  against ~{tokens_full:,} for the whole catalogue — <b>{pct}%</b> smaller.</div>
</div>""",
        unsafe_allow_html=True,
    )


def answer(text: str, *, verified: bool = False, verified_note: str = "") -> None:
    st.markdown(f'<div class="ayd-answer">{html.escape(str(text))}</div>', unsafe_allow_html=True)
    if verified:
        st.markdown(
            f'<span class="ayd-verified">✓ matches the committed contract</span>'
            f'<div class="ayd-note" style="margin-top:.4rem">{html.escape(verified_note)}</div>',
            unsafe_allow_html=True,
        )


def domain_card(name: str, blurb: str) -> None:
    st.markdown(
        f'<div class="ayd-dom"><b>{html.escape(name)}</b><p>{html.escape(blurb)}</p></div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="ayd-note">{html.escape(text)}</div>', unsafe_allow_html=True)
