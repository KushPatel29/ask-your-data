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
footnote. Where two machine-produced things have to be told apart - the vector
ranking against the keyword ranking - they are separated by FORM (filled tick
against hollow tick), never by inventing a third hue. A third accent would cost
the two real ones their meaning.

Type is IBM Plex - Condensed for display, Sans for prose, Mono for anything the
machine produced. Plex was drawn for technical products and it reads as
instrumentation rather than as a brand; it is also not the typeface every other
LLM demo reaches for.

THE SIGNATURE ELEMENT: `grounding()`, THE FUSION READOUT
It shows which tables the retriever selected out of the full catalogue and -
this is the part that matters - WHY each one is there, by drawing the two
rankings that reciprocal rank fusion actually reads.

It was rebuilt. The first version drew one bar per table, normalised to the top
hit, under a column header reading "cosine". Two things were wrong with that:

  * The number was not cosine. app/streamlit_app.py feeds this panel
    retrieve_hybrid(), whose score is a fused RRF value - measured range across
    the 39 golden questions, 0.011 to 0.033. The label described a different
    function's output.
  * Normalising an RRF score to the top hit produces a step, not a ranking.
    Measured over all 39 golden questions, the fourteen bars occupied a median
    span of 52 points out of 100, and the shape was bimodal: tables found by
    BOTH retrievers sat near 100%, tables found by only one fell off a cliff to
    ~50%. That cliff is the single most interesting fact in the panel - and it
    was rendered as an unexplained discontinuity.

So the bar is gone. Each row now shows the table's rank in the vector ranking
and its rank in the keyword ranking, plotted on one track. RRF only reads ranks,
so a track of ranks is the mechanism itself rather than a picture of it. The
example engine/retrieval.py cites in its own docstring - retail_customer_analytics
at vector rank 17, keyword rank 3, for "top wholesale customer by revenue" - is
now something you can see happen instead of something the source code claims.

See scripts/run_retrieval_eval.py for how the retrieval numbers were measured.
"""

from __future__ import annotations

import html
from pathlib import Path

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

/* Every readout in here is a column of numbers meant to be compared down the
   page. Proportional digits make a 1 narrower than a 7 and the column stops
   lining up, which is the difference between a table and an instrument. */
.ayd-mono, .ayd-stats, .ayd-pipe, .ayd-rail, .ayd-ground-panel, .ayd-map,
.ayd-guard, .ayd-note, .ayd-verified{
  font-variant-numeric:tabular-nums; font-feature-settings:'tnum' 1; }

/* A reticle rather than a box: two corners, not four sides. Enough to read as
   an instrument frame without adding a second border weight next to the real
   one. Purely decorative, so it is hidden from assistive tech by being a
   pseudo-element and is never the only thing marking a boundary. */
.ayd-hud{ position:relative; }
.ayd-hud::before, .ayd-hud::after{
  content:''; position:absolute; width:9px; height:9px; pointer-events:none;
  border-style:solid; border-color:var(--ayd-machine); border-width:0; opacity:.5; }
.ayd-hud::before{ top:-1px; left:-1px; border-top-width:1px; border-left-width:1px; }
.ayd-hud::after{ bottom:-1px; right:-1px; border-bottom-width:1px; border-right-width:1px; }

/* ---- masthead ---------------------------------------------------------- */
.ayd-mast{ border-bottom:1px solid var(--ayd-line); padding:.2rem 0 1.1rem; margin-bottom:1rem; }
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

/* ---- status rail ------------------------------------------------------- */
/* Machine state that is true for the whole session, not for one turn: what is
   loaded, which retriever is wired up, what the guard is set to refuse. It sits
   under the masthead on every view so the pipeline strip below it never has to
   re-explain the constants it runs under. */
/* The dividers are the grid's own 1px gaps with the line colour showing through
   from behind, not borders on the cells. Borders were wrong: the rail wraps to
   two rows under 700px or so, and `:first-child` can only clear the border on
   the first cell of the FIRST row - measured at 640px, the first cell of the
   second row drew a 1px stub hard against the container edge. A gap cannot
   produce an edge stub because there is no gap at an edge. */
.ayd-rail{ display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
  gap:1px; border:1px solid var(--ayd-line); border-radius:3px;
  background:var(--ayd-line); margin:0 0 1.3rem; overflow:hidden; }
.ayd-cellgrp{ padding:.5rem .7rem .55rem; min-width:0; background:var(--ayd-panel); }
.ayd-cellgrp .k{ display:block; font-family:var(--ayd-mono) !important; font-size:.6rem;
  letter-spacing:.17em; text-transform:uppercase; color:var(--ayd-muted); margin-bottom:.26rem; }
.ayd-cellgrp .v{ display:block; font-family:var(--ayd-mono) !important; font-size:.76rem;
  color:var(--ayd-ink); line-height:1.3; overflow-wrap:anywhere; }
.ayd-cellgrp .v em{ font-style:normal; color:var(--ayd-muted); }
.ayd-cellgrp .v s{ text-decoration:none; color:var(--ayd-machine); }
.ayd-cellgrp .v u{ text-decoration:none; color:var(--ayd-alert); }

/* ---- pipeline strip ---------------------------------------------------- */
.ayd-pipe{ display:flex; gap:.4rem; flex-wrap:wrap; align-items:center;
  font-family:var(--ayd-mono) !important; font-size:.66rem; letter-spacing:.13em;
  text-transform:uppercase; margin:.1rem 0 .7rem; }
.ayd-step{ border:1px solid var(--ayd-line); border-radius:2px; padding:.2rem .5rem;
  color:var(--ayd-muted); background:var(--ayd-panel); white-space:nowrap; }
.ayd-step[data-on="1"]{ color:var(--ayd-machine); border-color:rgba(34,211,238,.42);
  background:rgba(34,211,238,.07); }
.ayd-step[data-on="fail"]{ color:var(--ayd-alert); border-color:rgba(251,113,133,.45);
  background:rgba(251,113,133,.08); }
/* Measured wall-clock for the stage, when there is one to show. Letter-spacing
   is reset because a spaced-out "196ms" reads as three tokens, not a duration. */
.ayd-step .t{ margin-left:.45rem; letter-spacing:0; text-transform:none; opacity:.72; }
.ayd-arrow{ color:var(--ayd-line); }

/* ---- grounding readout (the signature) --------------------------------- */
.ayd-ground-panel{ border:1px solid var(--ayd-line); border-left:2px solid var(--ayd-machine);
  border-radius:3px; background:var(--ayd-panel); padding:.8rem .95rem; margin:.2rem 0 .9rem; }
.ayd-ground-head{ display:flex; justify-content:space-between; gap:1rem; align-items:baseline;
  flex-wrap:wrap; font-family:var(--ayd-mono) !important; font-size:.68rem; letter-spacing:.13em;
  text-transform:uppercase; color:var(--ayd-muted); margin-bottom:.2rem; }
.ayd-ground-head b{ color:var(--ayd-ink); }

/* Column header for the fusion table. The track's two ends are labelled so the
   axis is not something the reader has to infer from the tick positions. */
.ayd-cols{ display:grid; grid-template-columns:minmax(0,1fr) 34px 34px 150px 62px; gap:.55rem;
  align-items:end; font-family:var(--ayd-mono) !important; font-size:.6rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ayd-muted); padding:.5rem 0 .3rem;
  border-bottom:1px solid var(--ayd-line); }
.ayd-cols .axis{ display:flex; justify-content:space-between; letter-spacing:.06em; }
.ayd-cols .num{ text-align:right; }

.ayd-row{ display:grid; grid-template-columns:minmax(0,1fr) 34px 34px 150px 62px; gap:.55rem;
  align-items:center; font-family:var(--ayd-mono) !important; font-size:.75rem;
  padding:.3rem 0; border-bottom:1px solid rgba(35,40,56,.55); }
.ayd-row:last-of-type{ border-bottom:0; }
.ayd-tbl{ color:var(--ayd-ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ayd-tbl i{ color:var(--ayd-muted); font-style:normal; }

/* Rank badges. V is the vector ranking, K the keyword ranking. Both are the
   machine's own work so both are cyan; they are told apart by weight, and a
   table missing from one ranking shows a dash rather than a zero - it was not
   ranked last, it was not ranked. */
.ayd-rk{ font-size:.68rem; text-align:right; color:var(--ayd-machine); }
.ayd-rk.dim{ color:var(--ayd-muted); }
.ayd-rk.none{ color:var(--ayd-line); }

/* The track. Rank 1 at the left, the last rank in the fusion pool at the right.
   The filled tick is the vector rank, the hollow tick the keyword rank, and the
   bar between them is how far apart the two retrievers put this table - which
   is the whole reason reciprocal rank fusion is here. */
.ayd-track{ position:relative; height:14px; }
.ayd-track::before{ content:''; position:absolute; left:0; right:0; top:6px; height:1px;
  background:var(--ayd-line); }
.ayd-track i{ position:absolute; display:block; }
.ayd-gap{ top:6px; height:1px; background:var(--ayd-machine); opacity:.42; }
.ayd-tv{ top:2px; width:3px; height:9px; margin-left:-1.5px; background:var(--ayd-machine); }
.ayd-tk{ top:2px; width:7px; height:9px; margin-left:-3.5px;
  border:1px solid var(--ayd-machine); opacity:.85; }
.ayd-score{ color:var(--ayd-muted); text-align:right; font-size:.7rem; }

.ayd-saving{ margin-top:.6rem; padding-top:.55rem; border-top:1px solid var(--ayd-line);
  font-family:var(--ayd-mono) !important; font-size:.7rem; color:var(--ayd-muted); line-height:1.6; }
.ayd-saving b{ color:var(--ayd-machine); }

/* ---- schema map -------------------------------------------------------- */
/* One cell per table in the warehouse, grouped by domain. It answers "14 of 71"
   with the actual 71, so the scale of what was NOT sent to the model is visible
   rather than asserted. Hovering a cell names its table. */
.ayd-map{ border:1px solid var(--ayd-line); border-radius:3px; background:var(--ayd-panel);
  padding:.75rem .95rem .8rem; margin:.2rem 0 .9rem; }
.ayd-map-head{ font-family:var(--ayd-mono) !important; font-size:.62rem; letter-spacing:.15em;
  text-transform:uppercase; color:var(--ayd-muted); margin-bottom:.6rem;
  display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
.ayd-map-head b{ color:var(--ayd-ink); }
.ayd-map-row{ display:grid; grid-template-columns:104px minmax(0,1fr); gap:.7rem;
  align-items:center; padding:.11rem 0; }
.ayd-map-dom{ font-family:var(--ayd-mono) !important; font-size:.64rem; letter-spacing:.08em;
  color:var(--ayd-muted); text-align:right; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.ayd-map-dom[data-hit="1"]{ color:var(--ayd-ink); }
.ayd-cells{ display:flex; flex-wrap:wrap; gap:3px; }
.ayd-cell{ width:9px; height:9px; border-radius:1px; background:var(--ayd-panel-2);
  border:1px solid var(--ayd-line); }
.ayd-cell[data-hit="1"]{ background:var(--ayd-machine); border-color:var(--ayd-machine); }

/* ---- guard verdict ----------------------------------------------------- */
/* The guard is the safety boundary the assistant leans on, and until now the UI
   only ever showed that it had run. This shows what it checked and what it
   decided, which is the difference between a badge and a readout. */
.ayd-guard{ border:1px solid var(--ayd-line); border-radius:3px; background:var(--ayd-panel);
  padding:.6rem .8rem; margin:.2rem 0 .9rem; font-family:var(--ayd-mono) !important;
  font-size:.7rem; color:var(--ayd-muted); }
.ayd-guard[data-ok="0"]{ border-left:2px solid var(--ayd-alert); }
.ayd-guard[data-ok="1"]{ border-left:2px solid var(--ayd-machine); }
.ayd-guard-head{ font-size:.62rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--ayd-muted); margin-bottom:.45rem; }
.ayd-guard-head b[data-ok="1"]{ color:var(--ayd-machine); }
.ayd-guard-head b[data-ok="0"]{ color:var(--ayd-alert); }
.ayd-checks{ display:flex; flex-wrap:wrap; gap:.35rem .9rem; }
.ayd-check{ color:var(--ayd-muted); }
.ayd-check::before{ content:'✓'; color:var(--ayd-machine); margin-right:.35rem; }
.ayd-check[data-pass="0"]{ color:var(--ayd-alert); }
.ayd-check[data-pass="0"]::before{ content:'✕'; color:var(--ayd-alert); }

/* ---- answer ------------------------------------------------------------ */
.ayd-answer{ font-family:var(--ayd-cond) !important; font-weight:600; font-size:1.75rem; line-height:1.22;
  color:var(--ayd-ink); margin:.15rem 0 .45rem; letter-spacing:-.01em;
  font-variant-numeric:tabular-nums; }
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

/* Motion is about the subject or it does not run. The pipeline cell settles
   when its stage completes; the retrieved cells in the schema map light left to
   right, because that is a scan of the catalogue. Nothing loops, nothing
   pulses, and the whole sweep is capped at ~340ms so a Streamlit rerun that
   replays it does not become a tic.

   NO ANIMATION HERE CARRIES CONTENT, and that is a rule with a scar behind it.
   The first version wrote these as `animation:... both`, which paints the
   `from` keyframe whenever the animation has not started - including forever,
   if it never starts. Measured in a headless viewport that never composited:
   the keyword tick and the fusion gap bar computed to opacity 0, and every lit
   cell in the schema map computed to the UNLIT background, contrast ratio 1.00
   against its neighbours. The map was silently reporting that nothing had been
   retrieved.

   So fill-mode is gone. Every element's resting style is its CORRECT style, the
   keyframes only travel toward it, and the two that carry a stagger start from
   a state that is still visibly right - a lit cell begins lit and merely small,
   never unlit. A stalled animation now costs the flourish and nothing else. */
@media (prefers-reduced-motion: no-preference){
  .ayd-step[data-on="1"]{ animation:aydIn .28s ease-out; }
  @keyframes aydIn{ from{ opacity:.35; transform:translateY(1px);} to{ opacity:1; transform:none;} }
  /* backwards fill only so the stagger has something to hold during its delay;
     that held state is scale(.45) of an already-lit cell, not an unlit one. */
  .ayd-cell[data-hit="1"]{ animation:aydLit .3s ease-out backwards; animation-delay:var(--d,0ms); }
  @keyframes aydLit{ from{ transform:scale(.45); } to{ transform:none; } }
  /* Transform, never opacity. Dropping fill-mode is not sufficient on its own:
     an animation that has STARTED and never advances still paints its `from`
     keyframe, so a fade-in from opacity 0 leaves an invisible tick for as long
     as frames are not being produced - which is exactly what was measured. A
     tick that grows out of the rank line is legible at every frame of its own
     animation, including the first one. */
  .ayd-tv, .ayd-tk{ animation:aydTick .26s ease-out; transform-origin:50% 50%; }
  @keyframes aydTick{ from{ transform:scaleY(.25); } to{ transform:none; } }
  /* The connector is emphasis, not content - the two ticks and the two rank
     numbers already say everything it says - so this one may sweep from zero. */
  .ayd-gap{ animation:aydGap .26s ease-out; transform-origin:left center; }
  @keyframes aydGap{ from{ transform:scaleX(.02); } to{ transform:none; } }
}

@media (max-width:900px){
  .ayd-cols, .ayd-row{ grid-template-columns:minmax(0,1fr) 30px 30px 96px 56px; }
}
@media (max-width:640px){
  .ayd-mast h1.ayd-title{ font-size:2rem; }
  /* The track is the first thing to go: at this width it cannot resolve 28
     rank positions, and a track that lies about precision is worse than the
     two numbers on their own. The V and K badges carry the same information.

     The row then stacks, because keeping it on one line was measured at 375px
     to leave 105px for the table name and ellipsize all fourteen of them - and
     a schema readout whose table names are cut to "healthcare_f…" has stopped
     being a readout. The name takes a full line; the ranks and the fused score
     take the next one, and each rank grows its own label since the column
     header that used to name them is gone at this width. */
  .ayd-cols{ display:none; }
  .ayd-track{ display:none; }
  .ayd-row{ grid-template-columns:auto auto minmax(0,1fr);
    row-gap:.2rem; column-gap:.9rem; padding:.42rem 0; }
  .ayd-tbl{ grid-column:1 / -1; white-space:normal; overflow:visible;
    overflow-wrap:anywhere; }
  .ayd-rk{ text-align:left; }
  .ayd-rk::before{ content:'v '; color:var(--ayd-muted); }
  .ayd-rk.dim::before{ content:'k '; }
  .ayd-map-row{ grid-template-columns:74px minmax(0,1fr); }
}
</style>
"""


def inject() -> None:
    """Load the stylesheet once per session."""
    st.markdown(_CSS, unsafe_allow_html=True)


def build_marker() -> str:
    """A short fingerprint of the code actually running.

    Streamlit Cloud rebuilds on push but caches aggressively, and there was no
    way to tell from the page whether you were looking at the new build or a
    warm container still serving the old one. Hashing the source is more honest
    than a hand-bumped version string, which only ever tells you what someone
    remembered to change: this moves whenever the UI, the retriever or the
    manifest actually moves.

    Not git-derived on purpose - the deployed checkout may have no .git, and a
    marker that silently degrades to "unknown" answers nothing.
    """
    import hashlib

    root = Path(__file__).resolve().parents[1]
    watched = [
        root / "app" / "ui.py",
        root / "app" / "streamlit_app.py",
        root / "engine" / "retrieval.py",
        root / "data_manifest.py",
    ]
    digest = hashlib.sha256()
    for path in watched:
        try:
            # Normalise line endings before hashing. Git checks these files out
            # with CRLF on Windows and LF on the Linux box that runs the deploy,
            # so hashing raw bytes gave identical source two different markers -
            # which defeats the entire point of comparing local against live.
            raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            digest.update(raw)
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()[:7]


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
    <span>build <b>{build_marker()}</b></span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def status_rail(cells: list[tuple[str, str]]) -> None:
    """Session-constant machine state, rendered on every view.

    `cells` is a list of (label, value) pairs; the value may carry three inline
    tags, which exist so the caller can mark up a value without knowing any
    class names: <em> for a de-emphasised qualifier, <s> for the machine accent,
    <u> for the alert colour. Everything else the caller passes is escaped by
    the caller - this function does not escape, because the whole point of the
    tags is that they survive.

    The rail is deliberately NOT position:sticky. Streamlit's own fixed header
    is the only thing that could tell it what `top` to stick to, and its height
    is a per-release detail of an emotion class name - exactly the dependency
    this file is not allowed to take. A rail that overlaps the toolbar on the
    next Streamlit release is worse than one that scrolls.
    """
    groups = "".join(
        f'<div class="ayd-cellgrp"><span class="k">{html.escape(label)}</span>'
        f'<span class="v">{value}</span></div>'
        for label, value in cells
    )
    st.markdown(f'<div class="ayd-rail ayd-hud">{groups}</div>', unsafe_allow_html=True)


def pipeline(*, retrieved: bool = False, generated: bool = False,
             guarded: bool | str = False, executed: bool = False,
             attempts: int = 1, timings: dict[str, float] | None = None) -> None:
    """The stages this turn actually went through.

    Not decoration: each cell is lit from what really happened, so a refusal
    shows GENERATE lit and EXECUTE dark, and a guard rejection shows GUARD in
    the alert colour with the retry count beside it.

    `timings` maps a stage name to measured wall-clock milliseconds. A stage
    with no entry shows no number rather than a zero, because "not measured"
    and "took no time" are different claims and this panel does not make the
    second one on the first one's behalf.
    """
    timings = timings or {}

    def cell(label: str, state) -> str:
        on = "fail" if state == "fail" else ("1" if state else "0")
        ms = timings.get(label)
        if ms is None:
            stamp = ""
        else:
            # The guard is a handful of regexes and genuinely lands under a
            # millisecond. Rounding that to "0ms" reads as "not measured", which
            # is the one thing it is not - so anything faster than the clock can
            # resolve is reported as an upper bound instead.
            shown = f"{ms:,.0f}ms" if ms >= 1 else "&lt;1ms"
            stamp = f'<span class="t">{shown}</span>'
        return f'<span class="ayd-step" data-on="{on}">{label}{stamp}</span>'

    parts = [
        cell("retrieve", retrieved), '<span class="ayd-arrow">→</span>',
        cell("generate", generated), '<span class="ayd-arrow">→</span>',
        cell("guard", guarded), '<span class="ayd-arrow">→</span>',
        cell("execute", executed),
    ]
    if attempts > 1:
        parts.append(f'<span class="ayd-step" data-on="fail">retry ×{attempts - 1}</span>')
    st.markdown(f'<div class="ayd-pipe">{"".join(parts)}</div>', unsafe_allow_html=True)


def _tick_pos(rank: int, pool: int) -> float:
    """Map a 1-based rank onto the track, inset so an end tick is not clipped.

    Ticks are up to 7px wide and centred on their position, so a rank-1 tick at
    a literal 0% would lose its left half to the container. 2%..98% keeps both
    ends whole at every width the panel is used at.
    """
    if pool <= 1:
        return 50.0
    return round(2.0 + (rank - 1) * 96.0 / (pool - 1), 2)


def grounding(hits, *, total_tables: int, tokens_used: int, tokens_full: int,
              vector_ranks: dict[str, int] | None = None,
              keyword_ranks: dict[str, int] | None = None,
              pool: int | None = None) -> None:
    """What the model was allowed to see, why each table is there, and the cost.

    `hits` are RetrievedTable records from engine.retrieval.retrieve_hybrid, in
    fused order. `vector_ranks` and `keyword_ranks` are that table's position in
    each of the two input rankings - the only two inputs RRF has - and `pool` is
    how deep each ranking was read (retrieve_hybrid uses max(k*2, 12)).

    Ranks are optional. If the caller cannot produce them the panel degrades to
    the table list and the token accounting, which is still true; it does not
    invent a rank to keep the layout tidy.
    """
    if not hits:
        return
    vector_ranks = vector_ranks or {}
    keyword_ranks = keyword_ranks or {}
    have_ranks = bool(vector_ranks or keyword_ranks)
    pool = pool or max(
        [total_tables]
        + list(vector_ranks.values())
        + list(keyword_ranks.values())
    )

    rows = []
    both = 0
    for index, hit in enumerate(hits):
        vector = vector_ranks.get(hit.table)
        keyword = keyword_ranks.get(hit.table)
        if vector and keyword:
            both += 1

        marks = ""
        if vector and keyword:
            left, right = sorted((_tick_pos(vector, pool), _tick_pos(keyword, pool)))
            marks += f'<i class="ayd-gap" style="left:{left}%;width:{right - left:.2f}%"></i>'
        if vector:
            marks += f'<i class="ayd-tv" style="left:{_tick_pos(vector, pool)}%"></i>'
        if keyword:
            marks += f'<i class="ayd-tk" style="left:{_tick_pos(keyword, pool)}%"></i>'

        def badge(rank: int | None, dim: bool = False) -> str:
            if rank is None:
                return '<div class="ayd-rk none">—</div>'
            return f'<div class="ayd-rk{" dim" if dim else ""}">{rank}</div>'

        rows.append(
            f'<div class="ayd-row">'
            f'<div class="ayd-tbl">{html.escape(hit.table)} <i>· {html.escape(hit.domain)}</i></div>'
            f'{badge(vector)}{badge(keyword, dim=True)}'
            f'<div class="ayd-track">{marks}</div>'
            f'<div class="ayd-score">{hit.score:.5f}</div>'
            f'</div>'
        )

    header = (
        '<div class="ayd-cols">'
        '<div>table</div><div class="num">v</div><div class="num">k</div>'
        f'<div class="axis"><span>rank 1</span><span>{pool}</span></div>'
        '<div class="num">rrf</div>'
        '</div>'
    ) if have_ranks else ""

    saved = max(0, tokens_full - tokens_used)
    pct = round(100 * saved / tokens_full) if tokens_full else 0
    fusion_note = (
        f'<br><b>{both}</b> of {len(hits)} were ranked by both retrievers; '
        f'the rest were found by only one — which is the reason both are run.'
    ) if have_ranks else ""

    st.markdown(
        f"""
<div class="ayd-ground-panel ayd-hud">
  <div class="ayd-ground-head">
    <span>grounding · <b>{len(hits)}</b> of {total_tables} tables retrieved</span>
    <span>reciprocal rank fusion</span>
  </div>
  {header}
  {''.join(rows)}
  <div class="ayd-saving">~<b>{tokens_used:,}</b> tokens of schema in the prompt,
  against ~{tokens_full:,} for the whole catalogue — <b>{pct}%</b> smaller.{fusion_note}</div>
</div>""",
        unsafe_allow_html=True,
    )


def schema_map(by_domain: dict[str, list[tuple[str, bool]]], *,
               retrieved: int, total: int) -> None:
    """One cell per table in the warehouse, lit where the retriever selected it.

    The grounding panel says "14 of 71". This draws the 71. The claim the whole
    retrieval module rests on is that most of a warehouse is irrelevant to any
    one question, and a wall of unlit cells argues that better than a ratio.
    """
    if not by_domain:
        return
    lit_index = 0
    rows = []
    for domain, tables in by_domain.items():
        cells = []
        for table, hit in tables:
            if hit:
                # Stagger the sweep by lit cell, not by position, so the delay
                # never runs past the ~340ms cap however sparse the hits are.
                delay = min(lit_index * 22, 320)
                lit_index += 1
                style = f' style="--d:{delay}ms"'
            else:
                style = ""
            cells.append(
                f'<span class="ayd-cell" data-hit="{1 if hit else 0}"'
                f' title="{html.escape(table)}"{style}></span>'
            )
        any_hit = any(hit for _table, hit in tables)
        rows.append(
            f'<div class="ayd-map-row">'
            f'<div class="ayd-map-dom" data-hit="{1 if any_hit else 0}">{html.escape(domain)}</div>'
            f'<div class="ayd-cells">{"".join(cells)}</div>'
            f'</div>'
        )
    st.markdown(
        f"""
<div class="ayd-map ayd-hud">
  <div class="ayd-map-head">
    <span>catalogue · <b>{retrieved}</b> of {total} tables sent to the model</span>
    <span>{len(by_domain)} domains</span>
  </div>
  {''.join(rows)}
</div>""",
        unsafe_allow_html=True,
    )


def guard_verdict(*, ok: bool, reason: str, checks: list[tuple[str, bool]]) -> None:
    """What engine.sql_guard.validate_sql checked, and what it returned.

    `checks` are (label, passed) pairs describing the individual conditions.
    They are recomputed by the caller from the same module the guard uses, so
    this is a readout of the boundary rather than a decorative reassurance that
    one exists.
    """
    items = "".join(
        f'<span class="ayd-check" data-pass="{1 if passed else 0}">{html.escape(label)}</span>'
        for label, passed in checks
    )
    verdict = "pass" if ok else "blocked"
    st.markdown(
        f"""
<div class="ayd-guard ayd-hud" data-ok="{1 if ok else 0}">
  <div class="ayd-guard-head">read-only guard · <b data-ok="{1 if ok else 0}">{verdict}</b>
  — {html.escape(reason)}</div>
  <div class="ayd-checks">{items}</div>
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
