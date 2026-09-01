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

Type uses an operating-system-native control-room stack: condensed faces for
display, humanist sans for prose, and mono for anything the machine produced.
It keeps the instrumentation hierarchy without making the governed application
phone a third-party font CDN or weaken a strict production CSP.

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

Its foot then closes the funnel. "10 of 71 tables" names the two ends and hides
the middle: the two rankings are each read 20 deep and between them propose
21..35 distinct candidates (median 28 over the 39 golden questions, 30-49% of
the catalogue), of which the fusion keeps ten. The eighteen-odd tables that were
ranked and then dropped are what the fusion is FOR, and they were invisible.

SIX MORE READOUTS, ON THE SAME ARGUMENT
Every panel added since has to answer the same question the fusion readout
answers: is there real machine state here that the UI was only asserting?

  `query_plan()`   DuckDB's physical operator tree, read from
                   EXPLAIN (FORMAT JSON) — the plan the engine actually chose,
                   with the optimiser's estimated cardinality per operator and
                   the root estimate set against the rows the query really
                   returned. Measured over the 39 golden queries: EXPLAIN costs
                   0.20–1.70ms (median 0.44) and the tree is 2–36 operators
                   (median 5), which is why the panel scrolls at 21rem rather
                   than growing without limit.
  `guard_verdict()` now draws all of sql_guard.FORBIDDEN. "None of 26 forbidden
                   verbs" was a claim; the 26 are the evidence, and on a block
                   the one the guard named lights.
  `attempt_ledger()` the self-correction loop as a ledger — one row per attempt
                   with the error that ended it, which is the text the loop fed
                   back to the model. It used to be a sentence.
  `result_shape()` rows × columns in DuckDB's own type names (DESCRIBE, 0.18–
                   0.61ms), and how much of the row cap the answer used.
  `verification()` the layer that catches SQL which runs and is WRONG. It ran on
                   every turn and rendered nothing at all — engine/verify.py put
                   its findings on AskResult and app/streamlit_app.py dropped
                   them building the transcript entry, so the cost of checking
                   was paid and none of the benefit collected. Severity is form,
                   not a third hue: filled chip / hollow chip / bare, matching
                   the error-warn-note ladder that module documents. The rule
                   board is the forbidden-verb board's argument moved one stage
                   downstream. Measured on the 39 golden queries, both halves
                   cost 0.30–2.37ms together (median 0.66) and produce ZERO
                   findings — which is what a check that fires on good SQL
                   would not do.
  `exemplars()`    which solved question/SQL pairs the few-shot selector put
                   nearest this question, out of the 39 the accuracy contract
                   asserts — including the leave-one-out rule dropping the
                   question's own pair, which is what makes an eval over this
                   corpus mean anything and had never been visible. Nothing here
                   is amber even though the pairs are committed: amber marks a
                   value CI re-checks, and widening it to "anything from a
                   committed file" would cost the badge on the answer the only
                   thing it means.

A REFUSAL IS A READOUT TOO
engine/assistant.py has two ways to refuse and the UI was narrating them as
one sentence: the model calling cannot_answer, and the verifier still holding a
blocking finding on the final attempt, where the loop declines to execute SQL it
cannot stand behind. Only the first is "I can't answer that from the loaded
data". The second wrote a query and refused to run it — a far more interesting
thing to say — and it is now said by `verification(refused=True)` with the
refused SQL shown unrun beneath it. It also closed a leak: on that path the
refusal reason is verify.correction_message(), whose closing lines are addressed
to the MODEL, and they were being pasted into a warning a human reads.

And the pipeline strip's connectors are drawn rather than typed, carrying state:
a segment lights only where the signal crossed it, so a turn that stopped at the
guard shows a dark link at the point it stopped instead of four identical
arrows.

See scripts/run_retrieval_eval.py for how the retrieval numbers were measured,
and scripts/audit_ui.py for the contrast and markup audit this file is held to.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

CYAN = "#22D3EE"
AMBER = "#FBBF24"

_CSS = """
<style>
:root{
  --ayd-ground:#0A0C14; --ayd-panel:#12151F; --ayd-panel-2:#171B28;
  --ayd-line:#232838; --ayd-ink:#E6E9F2; --ayd-muted:#7C859C;
  --ayd-machine:#22D3EE; --ayd-verified:#FBBF24; --ayd-alert:#FB7185;
  --ayd-mono:'Cascadia Code','SFMono-Regular',Consolas,'Liberation Mono',monospace;
  --ayd-sans:Aptos,'Segoe UI Variable Text','Segoe UI',system-ui,-apple-system,sans-serif;
  --ayd-cond:'Bahnschrift SemiCondensed','Arial Narrow',Aptos,'Segoe UI',system-ui,sans-serif;
}

html, body, [class*="st-"]{ font-family:var(--ayd-sans) !important; }

/* Primary interactive surfaces meet the 44px mobile touch recommendation.
   Streamlit's password reveal and sidebar controls otherwise render as tiny
   icon-only targets even when the surrounding input is comfortably sized. */
.stButton > button,
.stFormSubmitButton > button,
[data-testid="stTextInput"] input,
[data-testid="stChatInput"] textarea,
[data-testid="stTextInput"] button,
[data-testid="stSidebarCollapseButton"] button{
  min-height:44px !important;
}
[data-testid="stTextInput"] button,
[data-testid="stSidebarCollapseButton"] button{
  min-width:44px !important;
}

/* ...except the icons, and this exception is not a nicety - without it the app
   renders every icon as its own NAME.

   Streamlit draws icons as ligatures: a <span data-testid="stIconMaterial">
   containing the literal text `keyboard_arrow_right`, which the Material
   Symbols font composes into one glyph. The selector above matches
   `[class*="st-"]`, every one of those spans carries an `st-emotion-cache-…`
   class, and `!important` beat Streamlit's own font-family. So the glyph font
   never applied and the ligature fell back to the prose face, which has no ligature
   for it and simply drew the letters.

   Every expander read "keyboard_arrow_right The accuracy contract…", the
   sidebar toggle read "keyboard_double_arrow_left", the password field read
   "visibility", and the chat avatars read "face" and "smart_toy" - overlapping
   whatever sat beside them. I twice dismissed this as an artifact of headless
   Chrome not loading Google Fonts. It was not: it was this line, in the real
   browser, on the deployed app, from the first version of this file.

   The family is NAMED rather than `unset`. font-family is an inherited
   property, so `unset` computes to `inherit` — and the parent is also matched
   by `[class*="st-"]`, so unsetting the override merely inherited the same
   wrong font. Measured: still the prose face on all fifteen icons.

   `font-feature-settings:'liga'` is belt and braces: these glyphs ARE
   ligatures, and anything that disables them puts the letters back. */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
.material-symbols-rounded,
.material-symbols-outlined{
  font-family:'Material Symbols Rounded','Material Symbols Outlined',
              'Material Icons' !important;
  font-feature-settings:'liga' !important;
  font-variant-ligatures:normal !important;
  letter-spacing:normal !important; }
.stApp{ background:
  radial-gradient(1100px 520px at 78% -12%, rgba(34,211,238,.07), transparent 60%),
  var(--ayd-ground); }

/* Every readout in here is a column of numbers meant to be compared down the
   page. Proportional digits make a 1 narrower than a 7 and the column stops
   lining up, which is the difference between a table and an instrument. */
/* Adding a panel and forgetting to name it here is a silent regression: the
   panel looks almost right, and only a column of digits that will not line up
   says otherwise. It has already happened once - the plan, ledger and shape
   readouts all shipped computing `font-variant-numeric: normal`, caught by
   reading getComputedStyle off the running app rather than by looking at it. */
.ayd-mono, .ayd-stats, .ayd-pipe, .ayd-rail, .ayd-ground-panel, .ayd-map,
.ayd-guard, .ayd-note, .ayd-verified, .ayd-plan, .ayd-att, .ayd-shape,
.ayd-cols-list, .ayd-ver, .ayd-ex, .ayd-ops, .ayd-layer, .ayd-voice, .ayd-metric{
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
.ayd-kicker::after{ content:''; flex:1; height:1px;
  background:linear-gradient(90deg,var(--ayd-line),transparent); }
/* Streamlit sets font-family on h1 with !important of its own, so a bare
   class loses even when it also declares !important. Element+class wins. */
.ayd-mast h1.ayd-title{ font-family:var(--ayd-cond) !important; font-weight:700;
  font-size:2.7rem; line-height:1.02;
  letter-spacing:-.015em; margin:.5rem 0 .4rem; color:var(--ayd-ink); }
.ayd-sub{ color:var(--ayd-muted); max-width:62ch; font-size:.95rem; line-height:1.55; margin:0; }
.ayd-stats{ display:flex; gap:1.6rem; flex-wrap:wrap; margin-top:1rem;
  font-family:var(--ayd-mono) !important; font-size:.72rem; color:var(--ayd-muted); }
.ayd-stats b{ color:var(--ayd-ink); font-weight:600; }
.ayd-mast[data-compact="1"]{ padding-bottom:.65rem; margin-bottom:.7rem; }
.ayd-mast[data-compact="1"] .ayd-kicker{ font-size:.6rem; }
.ayd-mast[data-compact="1"] h1.ayd-title{ font-size:1.5rem; margin:.28rem 0 0; }
.ayd-mast[data-compact="1"] h1.ayd-title br{ display:none; }
.ayd-mast[data-compact="1"] .ayd-title-tail::before{ content:' '; }
.ayd-mast[data-compact="1"] .ayd-sub{ display:none; }
.ayd-mast[data-compact="1"] .ayd-stats{ margin-top:.5rem; }

/* ---- workspace navigation --------------------------------------------- */
/* Three product jobs, three destinations: ask, inspect the governed catalog,
   and verify the controls. The active item reads as a selected instrument tab,
   not a marketing-site pill. The Streamlit control retains native keyboard and
   screen-reader behaviour; these rules only establish hierarchy. */
.st-key-workspace-nav{ margin:-.2rem 0 1rem; }
.st-key-workspace-nav [data-testid="stSegmentedControl"]{ width:100%; }
.st-key-workspace-nav [role="radiogroup"]{ width:100%; padding:3px;
  border:1px solid var(--ayd-line); border-radius:4px; background:var(--ayd-panel); }
.st-key-workspace-nav [role="radiogroup"] > label{ flex:1 1 0; justify-content:center;
  min-height:40px; font-family:var(--ayd-mono) !important; font-size:.68rem;
  letter-spacing:.08em; text-transform:uppercase; }
.st-key-workspace-nav [aria-checked="true"]{ color:var(--ayd-machine) !important;
  background:rgba(34,211,238,.08) !important; }

.ayd-viewhead{ margin:.1rem 0 1rem; max-width:72ch; }
.ayd-viewhead .k{ display:block; font-family:var(--ayd-mono) !important;
  color:var(--ayd-machine); font-size:.62rem; letter-spacing:.16em;
  text-transform:uppercase; margin-bottom:.32rem; }
.ayd-viewhead h2{ font-family:var(--ayd-cond) !important; color:var(--ayd-ink);
  font-size:1.55rem; line-height:1.1; margin:0 0 .32rem; }
.ayd-viewhead p{ color:var(--ayd-muted); font-size:.88rem; line-height:1.5; margin:0; }

/* The product's signature: every answer has a visible, governed route. This is
   a real sequence (unlike decorative numbered cards), so order carries meaning. */
.ayd-life{ display:grid; grid-template-columns:repeat(7,minmax(0,1fr));
  border:1px solid var(--ayd-line); border-radius:3px; overflow:hidden;
  background:var(--ayd-panel); margin:-.35rem 0 1.15rem; }
.ayd-life-step{ position:relative; padding:.52rem .58rem .58rem;
  min-width:0; border-right:1px solid var(--ayd-line); }
.ayd-life-step:last-child{ border-right:0; }
.ayd-life-step::after{ content:''; position:absolute; width:5px; height:5px;
  border-top:1px solid var(--ayd-machine); border-right:1px solid var(--ayd-machine);
  right:-3px; top:50%; transform:translateY(-50%) rotate(45deg);
  background:var(--ayd-panel); z-index:1; }
.ayd-life-step:last-child::after{ display:none; }
.ayd-life-step b{ display:block; font-family:var(--ayd-mono) !important;
  font-size:.63rem; letter-spacing:.1em; text-transform:uppercase;
  color:var(--ayd-ink); font-weight:500; white-space:nowrap; }
.ayd-life-step span{ display:block; color:var(--ayd-muted); font-size:.66rem;
  line-height:1.25; margin-top:.2rem; }
.ayd-life-step[data-stage="control"] b{ color:var(--ayd-machine); }

/* The example-question row. st.columns lays five buttons side by side and each
   sizes to its own text, so "Which department has the highest average tenure?"
   wrapped to four lines while "Total paid amount by payer type" took two — a
   ragged row of five different heights, which is the first thing on the page a
   visitor sees. Stretching the column and letting the button fill it makes them
   one band. st.container(key="ayd-examples") is what emits .st-key-ayd-examples,
   so the rule is scoped to that row and no other button on the page moves. */
.st-key-ayd-examples [data-testid="stColumn"]{ display:flex; align-items:stretch; }
.st-key-ayd-examples [data-testid="stColumn"] > div{ width:100%; height:100%; }
/* stElementContainer is the link that broke the first attempt: it sits between
   the column and the button as a display:block box sized to its own content,
   so stretching the column moved the COLUMN to 65px and left three of the five
   buttons at 46. Every rung from the column down to the button has to pass the
   height along, which means flex on each of them and `flex:1` on the child
   that must absorb the slack. Measured on the deployed app before and after:
   46/65/46/65/65 -> 65/65/65/65/65. */
.st-key-ayd-examples [data-testid="stElementContainer"]{ display:flex; flex:1 1 auto; }
.st-key-ayd-examples .stButton{ width:100%; display:flex; flex:1 1 auto; }
.st-key-ayd-examples .stButton > button{ width:100%; flex:1 1 auto;
  white-space:normal; line-height:1.3; }

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
/* The separators are drawn ON the cells, not by letting a container background
   show through 1px gaps. The gap trick is tidier to write and it has one bad
   failure: seven cells into a six-column grid leaves an orphan on row two, and
   the five empty column-slots beside it paint the CONTAINER colour. Measured at
   1280 with the sidebar open, that was a 700px block of #232838 sitting a full
   step lighter than the panel it borders — reading as a broken cell rather than
   as empty space. Drawing the rules on the cells makes leftover space panel
   coloured, so an orphan row simply ends. */
.ayd-rail{ display:grid; grid-template-columns:repeat(auto-fit,minmax(124px,1fr));
  gap:0; border:1px solid var(--ayd-line); border-radius:3px;
  background:var(--ayd-panel); margin:0 0 1.3rem; overflow:hidden; }
.ayd-cellgrp{ padding:.5rem .7rem .55rem; min-width:0; background:var(--ayd-panel);
  border-right:1px solid var(--ayd-line); border-bottom:1px solid var(--ayd-line); }
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
.ayd-step[data-on="cert"]{ color:var(--ayd-verified); border-color:rgba(251,191,36,.42);
  background:rgba(251,191,36,.07); }
.ayd-step[data-on="fail"]{ color:var(--ayd-alert); border-color:rgba(251,113,133,.45);
  background:rgba(251,113,133,.08); }
/* Measured wall-clock for the stage, when there is one to show. Letter-spacing
   is reset because a spaced-out "196ms" reads as three tokens, not a duration. */
.ayd-step .t{ margin-left:.45rem; letter-spacing:0; text-transform:none; opacity:.72; }
/* The link between two stages is drawn rather than typed, and it carries state:
   it lights only where the signal really passed OUT of one lit stage and INTO
   the next, so a turn that stopped at the guard shows a dark segment at exactly
   the point it stopped. A row of identical "→" glyphs said the pipeline had
   four stages; this says how far down it the turn got. Decorative in the
   accessibility sense - the stage cells already carry every state it encodes -
   so it is an empty element with no text to announce. */
.ayd-arrow{ flex:0 0 auto; width:14px; height:1px; background:var(--ayd-line); }
.ayd-arrow[data-on="1"]{ background:var(--ayd-machine); opacity:.5; }
.ayd-arrow[data-on="fail"]{ background:var(--ayd-alert); opacity:.55; }

/* ---- voice dock ------------------------------------------------------- */
.ayd-voice{ display:grid; grid-template-columns:44px minmax(0,1fr) auto; gap:.8rem;
  align-items:center; border:1px solid var(--ayd-line); border-left:2px solid var(--ayd-machine);
  border-radius:3px; background:linear-gradient(90deg,rgba(34,211,238,.055),transparent 62%),
  var(--ayd-panel); padding:.7rem .85rem; margin:.45rem 0 .55rem; }
.ayd-voice-orb{ width:38px; height:38px; border-radius:50%; position:relative;
  border:1px solid rgba(34,211,238,.52); background:rgba(34,211,238,.08);
  box-shadow:inset 0 0 0 7px rgba(34,211,238,.025); }
.ayd-voice-orb::before{ content:''; position:absolute; inset:11px 13px 10px;
  border:2px solid var(--ayd-machine); border-top:0; border-radius:0 0 9px 9px; opacity:.9; }
.ayd-voice-orb::after{ content:''; position:absolute; width:2px; height:6px;
  left:17px; bottom:5px; background:var(--ayd-machine);
  box-shadow:-5px 5px 0 -1px var(--ayd-machine),
  5px 5px 0 -1px var(--ayd-machine); opacity:.9; }
.ayd-voice-copy b{ display:block; font-family:var(--ayd-cond) !important; font-size:.92rem;
  color:var(--ayd-ink); letter-spacing:.01em; }
.ayd-voice-copy span{ display:block; margin-top:.12rem; font-family:var(--ayd-mono) !important;
  color:var(--ayd-muted); font-size:.66rem; line-height:1.45; }
.ayd-voice-state{ font-family:var(--ayd-mono) !important; font-size:.62rem; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ayd-machine); white-space:nowrap; }
.ayd-voice[data-ready="0"]{ border-left-color:var(--ayd-muted); }
.ayd-voice[data-ready="0"] .ayd-voice-orb{ border-color:var(--ayd-line);
  filter:grayscale(1); opacity:.65; }
.ayd-voice[data-ready="0"] .ayd-voice-state{ color:var(--ayd-muted); }

/* ---- certified metric ------------------------------------------------ */
.ayd-metric{ border:1px solid rgba(251,191,36,.25); border-left:2px solid var(--ayd-verified);
  border-radius:3px; background:linear-gradient(90deg,rgba(251,191,36,.055),transparent 68%),
  var(--ayd-panel); padding:.8rem .95rem; margin:.55rem 0 .85rem; }
.ayd-metric-head{ display:flex; justify-content:space-between; gap:1rem; align-items:baseline;
  flex-wrap:wrap; font-family:var(--ayd-mono) !important; font-size:.64rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ayd-verified); }
.ayd-metric-head span{ color:var(--ayd-muted); letter-spacing:.05em; text-transform:none; }
.ayd-metric-def{ color:var(--ayd-ink); font-size:.82rem; line-height:1.55; margin:.55rem 0; }
.ayd-metric-compare{ border-top:1px solid var(--ayd-line); padding-top:.55rem;
  font-family:var(--ayd-mono) !important; font-size:.68rem; line-height:1.55;
  color:var(--ayd-muted); }
.ayd-metric-compare b{ color:var(--ayd-verified); font-weight:500; }

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
  font-family:var(--ayd-mono) !important; font-size:.7rem;
  color:var(--ayd-muted); line-height:1.6; }
.ayd-saving b{ color:var(--ayd-machine); }

/* ---- schema map -------------------------------------------------------- */
/* One cell per table in the warehouse, grouped by domain. It answers "10 of 71"
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

/* Sidebar expander labels. Table names here run to 34 characters
   (`wholesale_labor_department_month`) inside a 300px rail, and the default
   break put "healthcare_ar_yield_predictio / ns" on two lines mid-word. A
   smaller mono face fits more of the name before it has to break at all. */
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] details summary p{
  font-family:var(--ayd-mono) !important; font-size:.7rem;
  overflow-wrap:anywhere; line-height:1.35; }

/* One column of a table, with the role engine/semantics.py inferred for it.

   The sidebar's schema browser is the only place in the app that shows what the
   layer concluded at COLUMN level, which was a strange gap: every answer the
   keyless engine gives is downstream of these five verdicts, and none of them
   were visible anywhere. Roles carry the machine accent, values are quoted in
   muted text because they are data rather than schema, and a search hit is
   marked so you can see WHY a table matched. */
.ayd-cols-browse{ font-family:var(--ayd-mono) !important; font-size:.64rem;
  line-height:1.55; }
.ayd-colrow{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:.5rem;
  padding:.1rem 0; align-items:baseline; }
.ayd-colname{ color:var(--ayd-ink); overflow-wrap:anywhere; }
.ayd-colname mark{ background:rgba(34,211,238,.18); color:var(--ayd-machine);
  padding:0 .1rem; border-radius:2px; }
.ayd-colrole{ color:var(--ayd-muted); font-size:.58rem; letter-spacing:.1em;
  text-transform:uppercase; white-space:nowrap; }
.ayd-colvals{ grid-column:1 / -1; color:var(--ayd-muted); font-size:.6rem;
  overflow-wrap:anywhere; margin:0 0 .18rem; }

/* What the compiler read out of the warehouse, on the empty state.

   The landing page used to end at the example buttons and leave roughly 700px
   of nothing between them and the chat input — over half the first screen at
   768x1900. The honest thing to put there is not decoration: it is the four
   numbers that make the keyless engine possible at all, every one of them
   probed from DuckDB at startup rather than typed into this file. A visitor who
   reads nothing else learns that the thing answering them derived its own
   understanding of the schema. */
.ayd-layer{ display:grid; grid-template-columns:repeat(auto-fit,minmax(184px,1fr));
  gap:0; border:1px solid var(--ayd-line); border-radius:3px;
  background:var(--ayd-panel); margin:.2rem 0 1rem; overflow:hidden; }
.ayd-layer-cell{ padding:.7rem .85rem .75rem; min-width:0;
  border-right:1px solid var(--ayd-line); border-bottom:1px solid var(--ayd-line); }
.ayd-layer-n{ display:block; font-family:var(--ayd-cond) !important; font-weight:700;
  font-size:1.5rem; line-height:1.1; color:var(--ayd-machine);
  font-variant-numeric:tabular-nums; }
.ayd-layer-k{ display:block; font-family:var(--ayd-mono) !important; font-size:.6rem;
  letter-spacing:.16em; text-transform:uppercase; color:var(--ayd-muted);
  margin-top:.3rem; }
.ayd-layer-w{ display:block; font-family:var(--ayd-mono) !important; font-size:.62rem;
  color:var(--ayd-muted); line-height:1.45; margin-top:.2rem; }
.ayd-layer-foot{ grid-column:1 / -1; padding:.55rem .85rem .6rem;
  border-bottom:1px solid transparent; font-family:var(--ayd-mono) !important;
  font-size:.62rem; color:var(--ayd-muted); line-height:1.5; }

/* ---- the result, drawn -------------------------------------------------- */
/* Ten departments and their revenue is a SHAPE, and the app was printing it as
   ten rows of digits and asking the reader to do the comparison in their head.
   The grid stays — it is the auditable artifact and the thing the CSV matches —
   but the chart goes above it, because "which is biggest and by how much" is
   the question a breakdown was asked in order to answer.

   Hand-drawn SVG rather than a plotting library, and not for the dependency:
   every other readout in this file is drawn from the same tokens, and a Vega
   chart would arrive with its own palette, its own type stack and its own idea
   of a gridline. One accent, tabular figures, the value printed at the end of
   the bar so the length never has to be estimated. */
.ayd-chart{ border:1px solid var(--ayd-line); border-radius:3px;
  background:var(--ayd-panel); padding:.7rem .85rem .5rem; margin:.2rem 0 .6rem; }
.ayd-chart-head{ display:flex; justify-content:space-between; gap:1rem;
  font-family:var(--ayd-mono) !important; font-size:.6rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ayd-muted); margin-bottom:.5rem; }
.ayd-chart svg{ display:block; width:100%; height:auto; overflow:visible; }
.ayd-chart .bar{ fill:var(--ayd-machine); opacity:.72; }
.ayd-chart .bar-top{ opacity:1; }
.ayd-chart .track{ fill:var(--ayd-panel-2); }
.ayd-chart .cat{ fill:var(--ayd-ink); font-family:var(--ayd-mono); font-size:10px; }
.ayd-chart .val{ fill:var(--ayd-muted); font-family:var(--ayd-mono); font-size:10px;
  font-variant-numeric:tabular-nums; text-anchor:end; }
.ayd-chart .line{ fill:none; stroke:var(--ayd-machine); stroke-width:1.5; }
.ayd-chart .dot{ fill:var(--ayd-machine); }
.ayd-chart .axis{ stroke:var(--ayd-line); stroke-width:1; }
.ayd-chart-foot{ font-family:var(--ayd-mono) !important; font-size:.6rem;
  color:var(--ayd-muted); margin-top:.35rem; }
/* SVG text scales with the viewBox, so a chart squeezed to a phone shrinks its
   own 10px labels to about 6px. Below 560 the chart keeps a legible width and
   scrolls inside its own box instead — the page itself never scrolls
   sideways, which is the rule the rest of this file follows. */
@media (max-width:560px){
  .ayd-chart{ overflow-x:auto; }
  .ayd-chart svg{ min-width:460px; } }

/* ---- operations ledger ------------------------------------------------- */
/* Observability OF A REQUEST is what the pipeline strip has always shown.
   Observability OF A SERVICE is a different thing and needs something durable
   to aggregate over, which is why this panel could not exist until
   engine/audit.py did. Every number here is computed from records that were
   really written this session; none is a placeholder and none is a rate the
   app would like to have.

   No amber anywhere in it. These are the machine's own measurements of itself,
   which is exactly what cyan means; amber would claim CI re-checks them. */
.ayd-ops{ border:1px solid var(--ayd-line); border-radius:3px;
  background:var(--ayd-panel); margin:.2rem 0 1rem; overflow:hidden; }
.ayd-ops-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(122px,1fr)); gap:0; }
.ayd-ops-cell{ padding:.6rem .8rem .65rem; min-width:0; background:var(--ayd-panel);
  border-right:1px solid var(--ayd-line); border-bottom:1px solid var(--ayd-line); }
.ayd-ops-n{ display:block; font-family:var(--ayd-cond) !important; font-weight:700;
  font-size:1.32rem; line-height:1.1; color:var(--ayd-machine); }
.ayd-ops-k{ display:block; font-family:var(--ayd-mono) !important; font-size:.58rem;
  letter-spacing:.16em; text-transform:uppercase; color:var(--ayd-muted); margin-top:.28rem; }
/* The stage bars are the one place a length carries meaning, so they are
   normalised to the slowest stage and the number is printed beside the bar —
   a bar you cannot read a value off is decoration. */
.ayd-ops-rows{ padding:.55rem .8rem .7rem; border-bottom:1px solid var(--ayd-line); }
.ayd-ops-row{ display:grid; grid-template-columns:5.6rem 1fr 4.2rem; gap:.6rem;
  align-items:center; font-family:var(--ayd-mono) !important; font-size:.66rem;
  color:var(--ayd-muted); padding:.16rem 0; }
.ayd-ops-row b{ color:var(--ayd-ink); font-weight:500; }
.ayd-ops-track{ height:5px; background:var(--ayd-panel-2); border-radius:1px; overflow:hidden; }
.ayd-ops-fill{ height:100%; background:var(--ayd-machine); opacity:.62; }
.ayd-ops-num{ text-align:right; color:var(--ayd-ink); }
.ayd-ops-head{ font-family:var(--ayd-mono) !important; font-size:.6rem;
  letter-spacing:.16em; text-transform:uppercase; color:var(--ayd-muted);
  padding:.55rem .8rem .1rem; }
.ayd-ops-foot{ padding:.55rem .8rem .65rem; font-family:var(--ayd-mono) !important;
  font-size:.62rem; color:var(--ayd-muted); line-height:1.6; }
.ayd-ops-foot li{ margin-left:.9rem; }
@media (max-width:640px){
  .ayd-ops-row{ grid-template-columns:4.8rem 1fr 3.6rem; font-size:.6rem; } }

/* A refusal, in this palette rather than Streamlit's.

   st.warning paints an olive box with a black-on-yellow icon, which on this
   near-black instrument panel is the single loudest thing on the page — louder
   than the answer, in a hue the design system does not otherwise use. And a
   refusal is not a warning: nothing went wrong. The compiler declined to guess,
   which is the behaviour this whole project argues for, so it gets the amber
   the app already reserves for "this claim is backed", not a colour that reads
   as an error.

   NEITHER accent, though, and the repo's own test is what settled it: amber
   means "a committed file backs this number" and cyan means "a machine derived
   this", and a refusal is neither. `--ayd-alert` is the failure state and a
   refusal is not a failure. So the panel is neutral, and the mono heading does
   the work of marking it as a different KIND of result rather than a louder
   one.

   The heading names WHICH kind of refusal, because they are different facts:
   the grammar cannot express the question, or the question cannot be bound to
   this warehouse. */
.ayd-refusal{ border:1px solid var(--ayd-line); border-left:2px solid var(--ayd-muted);
  border-radius:3px; background:var(--ayd-panel); padding:.7rem .9rem .75rem;
  margin:.2rem 0 .9rem; }
.ayd-refusal-head{ font-family:var(--ayd-mono) !important; font-size:.62rem;
  letter-spacing:.15em; text-transform:uppercase; color:var(--ayd-muted);
  margin-bottom:.4rem; }
.ayd-refusal-body{ font-size:.9rem; line-height:1.5; color:var(--ayd-ink); }

/* The compiler trace. One row per binding decision, each carrying the words in
   the question that bought it. This panel exists because the deterministic
   planner can do something the model cannot: name the evidence for every clause
   it wrote. A model can be asked to explain itself and will produce plausible
   prose; engine/planner.py returns the actual bindings it used, so this is a
   readout rather than a rationalisation. */
.ayd-trace{ border:1px solid var(--ayd-line); border-radius:3px; background:var(--ayd-panel);
  padding:.6rem .8rem; margin:.2rem 0 .9rem; font-family:var(--ayd-mono) !important;
  font-size:.7rem; color:var(--ayd-muted); border-left:2px solid var(--ayd-machine); }
.ayd-trace-head{ font-size:.62rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--ayd-muted); margin-bottom:.5rem; }
.ayd-trace-head b{ color:var(--ayd-machine); }
.ayd-trace-row{ display:flex; gap:.6rem; padding:.16rem 0; align-items:baseline; }
.ayd-trace-key{ min-width:5.6rem; color:var(--ayd-muted); font-size:.62rem;
  letter-spacing:.09em; text-transform:uppercase; flex:none; }
.ayd-trace-val{ color:var(--ayd-ink); word-break:break-word; }
.ayd-words{ display:flex; flex-wrap:wrap; gap:.16rem .3rem; margin-top:.5rem;
  padding-top:.5rem; border-top:1px solid var(--ayd-line); }
.ayd-word{ font-size:.6rem; letter-spacing:.06em; border:1px solid transparent;
  border-radius:2px; padding:0 .22rem; }
/* Bound: the planner used this word. Loose: nothing in 71 tables contains it,
   so it was excused from the coverage denominator rather than silently dropped.
   Missed: the warehouse HAS this word somewhere and this plan did not use it —
   the only one of the three that is a debt. */
.ayd-word[data-w="bound"]{ color:var(--ayd-machine); border-color:rgba(34,211,238,.35);
  background:rgba(34,211,238,.08); }
.ayd-word[data-w="loose"]{ color:var(--ayd-muted); border-color:var(--ayd-line); }
.ayd-word[data-w="missed"]{ color:var(--ayd-alert); border-color:rgba(251,113,133,.4);
  background:rgba(251,113,133,.08); }

/* The forbidden-verb array. Same argument as the schema map, applied to the
   safety boundary: "none of 26 forbidden verbs" is a claim, and the 26 drawn out
   is the evidence. On a passing query every token is muted and the row reads as
   a cleared board; on a block the one verb the guard named lights in the alert
   colour, so the reader sees WHICH boundary was crossed rather than that one
   exists. Read from sql_guard.FORBIDDEN by the caller, never re-typed here.

   Muted is used rather than the line colour even though these are the "off"
   state: 27 labels nobody can read is not restraint, it is a wall of noise.
   Measured 5.05:1 on the panel background. */
.ayd-verbs{ display:flex; flex-wrap:wrap; gap:.16rem .3rem; margin-top:.5rem;
  padding-top:.5rem; border-top:1px solid var(--ayd-line); }
.ayd-verb{ font-size:.6rem; letter-spacing:.09em; color:var(--ayd-muted);
  border:1px solid transparent; border-radius:2px; padding:0 .18rem; }
.ayd-verb[data-hit="1"]{ color:var(--ayd-alert); border-color:rgba(251,113,133,.45);
  background:rgba(251,113,133,.1); }

/* ---- verifier ---------------------------------------------------------- */
/* The guard answers "is this allowed to run". The verifier answers the harder
   one: "did what ran mean anything". It is the only stage that can reject SQL
   which parses, binds, executes in three milliseconds and returns rows — and
   until now it ran on every turn and put nothing on the page at all.

   Severity is told apart by FORM, not by a third hue, the same rule the fusion
   ticks follow. error is a filled chip, warn is the same colour hollow, note is
   muted with no chip at all. That is exactly the ladder engine/verify.py
   documents: error never reaches the reader, warn buys one correction attempt,
   note only annotates — so the visual weight tracks the cost of being wrong.

   The rule board underneath is the forbidden-verb board's argument applied to
   this boundary: "nothing structural was found" is a claim, and the checks that
   could have fired are the evidence for it. It is deliberately silent about
   whether a check was APPLICABLE — Verifier.check_sql short-circuits on a
   single-table query and does not report that it did, and a UI that inferred
   applicability would be re-deriving structure the verifier owns. A quiet rule
   means "did not fire", which is true either way. */
.ayd-ver{ border:1px solid var(--ayd-line); border-radius:3px; background:var(--ayd-panel);
  padding:.6rem .8rem; margin:.2rem 0 .9rem; font-family:var(--ayd-mono) !important;
  font-size:.7rem; color:var(--ayd-muted); }
.ayd-ver[data-worst="clean"]{ border-left:2px solid var(--ayd-machine); }
.ayd-ver[data-worst="note"]{ border-left:2px solid var(--ayd-line); }
.ayd-ver[data-worst="warn"], .ayd-ver[data-worst="error"]{ border-left:2px solid var(--ayd-alert); }
.ayd-ver-head{ font-size:.62rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--ayd-muted); margin-bottom:.45rem; display:flex; justify-content:space-between;
  gap:1rem; flex-wrap:wrap; }
.ayd-ver-head b{ color:var(--ayd-machine); }
.ayd-ver-head b[data-alert="1"]{ color:var(--ayd-alert); }
/* The refusal line. It is the loudest thing this panel can say, so it says it
   in words rather than by shouting in colour: the turn ended here, and the
   query that was written is on screen below unrun. */
.ayd-ver-refused{ color:var(--ayd-alert); letter-spacing:.02em; text-transform:none;
  font-size:.7rem; line-height:1.5; margin:0 0 .5rem; }
.ayd-ver-row{ display:grid; grid-template-columns:52px 148px minmax(0,1fr); gap:.6rem;
  align-items:baseline; padding:.3rem 0; border-top:1px solid rgba(35,40,56,.55); }
.ayd-ver-row:first-of-type{ border-top:0; }
/* filled / hollow / bare — one shape per rung of the severity ladder. */
.ayd-sev{ font-size:.58rem; letter-spacing:.11em; text-transform:uppercase;
  text-align:center; border-radius:2px; padding:.05rem .2rem; border:1px solid transparent; }
.ayd-sev[data-sev="error"]{ color:var(--ayd-alert); border-color:rgba(251,113,133,.45);
  background:rgba(251,113,133,.1); }
.ayd-sev[data-sev="warn"]{ color:var(--ayd-alert); border-color:rgba(251,113,133,.45); }
.ayd-sev[data-sev="note"]{ color:var(--ayd-muted); }
.ayd-ver-check{ color:var(--ayd-machine); overflow-wrap:anywhere; }
.ayd-ver-msg{ color:var(--ayd-ink); line-height:1.45; overflow-wrap:anywhere; }
/* Same board as the forbidden verbs, same reason. Muted rather than the line
   colour: a rule nobody can read is not restraint. Measured 4.94:1 on panel. */
.ayd-rules{ display:flex; flex-wrap:wrap; gap:.16rem .3rem; margin-top:.5rem;
  padding-top:.5rem; border-top:1px solid var(--ayd-line); }
.ayd-rule{ font-size:.6rem; letter-spacing:.06em; color:var(--ayd-muted);
  border:1px solid transparent; border-radius:2px; padding:0 .18rem; }
.ayd-rule[data-hit="error"], .ayd-rule[data-hit="warn"]{ color:var(--ayd-alert);
  border-color:rgba(251,113,133,.45); background:rgba(251,113,133,.1); }
.ayd-rule[data-hit="note"]{ color:var(--ayd-ink); border-color:var(--ayd-line); }

/* ---- exemplar bank ----------------------------------------------------- */
/* Which solved questions were put in front of the model. engine/exemplars.py
   picks the k nearest verified question/SQL pairs out of the same 39 the
   accuracy contract asserts, and the reader had no way to know any of it
   happened.

   Nothing here is amber, and that was a decision rather than an oversight. The
   pairs do come out of a committed file, which is nearly the amber rule — but
   amber marks a VALUE this app computed and CI re-checks, not source material
   the prompt quoted. Widening it to "anything committed" would cost the badge
   on the answer the only thing it means. The SELECTION is the machine's own
   work (one MiniLM embedding, then reciprocal rank fusion against the retrieved
   tables), so the selection is cyan and the provenance is stated in words. */
.ayd-ex{ border:1px solid var(--ayd-line); border-left:2px solid var(--ayd-machine);
  border-radius:3px; background:var(--ayd-panel); padding:.6rem .8rem .65rem;
  margin:.2rem 0 .9rem; font-family:var(--ayd-mono) !important; font-size:.7rem;
  color:var(--ayd-muted); }
.ayd-ex-head{ font-size:.62rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--ayd-muted); margin-bottom:.5rem; display:flex; justify-content:space-between;
  gap:1rem; flex-wrap:wrap; }
.ayd-ex-head b{ color:var(--ayd-ink); }
.ayd-ex-row{ padding:.34rem 0; border-top:1px solid rgba(35,40,56,.55); }
.ayd-ex-row:first-of-type{ border-top:0; }
.ayd-ex-top{ display:grid; grid-template-columns:20px minmax(0,1fr) auto; gap:.6rem;
  align-items:baseline; }
.ayd-ex-n{ color:var(--ayd-muted); font-size:.66rem; }
.ayd-ex-q{ color:var(--ayd-ink); line-height:1.4; overflow-wrap:anywhere; }
.ayd-ex-meta{ color:var(--ayd-muted); font-size:.66rem; white-space:nowrap; }
.ayd-ex-meta b{ color:var(--ayd-machine); font-weight:400; }
/* The reference SQL is the substance of an exemplar, so it is shown exactly and
   never re-wrapped. A long one scrolls in its own track rather than reflowing
   into a paragraph or pushing the page sideways. */
.ayd-ex-sql{ margin-top:.2rem; margin-left:calc(20px + .6rem); font-size:.68rem;
  color:var(--ayd-ink); white-space:pre; overflow-x:auto; padding-bottom:.15rem; }
.ayd-ex-foot{ margin-top:.5rem; padding-top:.45rem; border-top:1px solid var(--ayd-line);
  font-size:.68rem; color:var(--ayd-muted); line-height:1.55; }
.ayd-ex-foot b{ color:var(--ayd-machine); }

/* ---- attempt ledger ---------------------------------------------------- */
/* The self-correction loop is the most machine-like thing the assistant does
   and it used to be a sentence. One row per attempt, in order, each carrying
   the error that ended it - which is exactly what the loop fed back to the
   model - and the surviving attempt marked as the one that ran.

   The row count is bounded by engine.assistant.MAX_ATTEMPTS, passed in rather
   than assumed, so the denominator moves if that constant does. */
.ayd-att{ border:1px solid var(--ayd-line); border-left:2px solid var(--ayd-alert);
  border-radius:3px; background:var(--ayd-panel); padding:.6rem .8rem; margin:.2rem 0 .9rem;
  font-family:var(--ayd-mono) !important; font-size:.7rem; color:var(--ayd-muted); }
.ayd-att-head{ font-size:.62rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--ayd-muted); margin-bottom:.5rem; }
.ayd-att-head b{ color:var(--ayd-alert); }
.ayd-att-row{ display:grid; grid-template-columns:64px 46px minmax(0,1fr); gap:.6rem;
  align-items:baseline; padding:.22rem 0; border-top:1px solid rgba(35,40,56,.55); }
.ayd-att-row:first-of-type{ border-top:0; }
.ayd-att-n{ color:var(--ayd-muted); }
.ayd-att-v{ color:var(--ayd-alert); }
.ayd-att-row[data-ok="1"] .ayd-att-v{ color:var(--ayd-machine); }
.ayd-att-why{ color:var(--ayd-muted); overflow-wrap:anywhere; line-height:1.45; }
.ayd-att-row[data-ok="1"] .ayd-att-why{ color:var(--ayd-ink); }

/* ---- query plan -------------------------------------------------------- */
/* DuckDB's own physical operator tree for the SQL that is about to run, read
   from EXPLAIN (FORMAT JSON) - not a redrawing of the query, the plan the
   engine actually chose. Operator names are the machine's work so they are
   cyan; PROJECTION is plumbing and recedes to muted, because a plan where 108
   of 240 nodes shout equally is not a readout.

   The right-hand column is the optimiser's ESTIMATE, which is the interesting
   part: it is a guess, it is labelled as one, and the panel foot puts the root
   estimate next to the row count the query really returned. An instrument that
   shows a prediction is only useful if it also shows the outturn.

   The tree guides are box-drawing characters rather than CSS borders on purpose:
   the rows are a flat DFS list, and the vertical continuation line of a
   two-child operator cannot be drawn from a row that does not know its
   ancestors' sibling counts.

   But they are NOT laid out by the monospace grid, because coverage of the Box
   Drawing block varies across operating-system mono faces. Measured at .72rem:
   space, M and i all advance 6.913px, while │ └ ├ ─ advance 6.334px — they are
   coming from a fallback face, and a tree drawn with them drifts about half a
   pixel per character, which at depth 10 is a branch line three pixels off the
   one above it. So each three-character level is boxed in an inline-block of
   exactly 3ch and the glyph sits inside it. The grid is then the box's, not the
   glyph's, and it holds whichever face ends up drawing the corner. */
.ayd-plan{ border:1px solid var(--ayd-line); border-left:2px solid var(--ayd-machine);
  border-radius:3px; background:var(--ayd-panel); padding:.7rem .9rem .75rem;
  margin:.2rem 0 .9rem; font-family:var(--ayd-mono) !important; }
.ayd-plan-head{ font-size:.62rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--ayd-muted); margin-bottom:.5rem; display:flex; justify-content:space-between;
  gap:1rem; flex-wrap:wrap; }
.ayd-plan-head b{ color:var(--ayd-ink); }
/* A deep plan scrolls inside the panel instead of pushing the result off the
   page. Measured over the 39 golden queries the tree is 2..36 operators, median
   5, so this caps the tail rather than the common case. */
.ayd-plan-body{ max-height:21rem; overflow:auto; }
.ayd-op{ display:grid; grid-template-columns:minmax(0,1fr) 92px; gap:.7rem;
  align-items:baseline; font-size:.72rem; padding:.13rem 0; }
.ayd-op-l{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ayd-guide{ color:var(--ayd-line); white-space:pre; }
.ayd-guide i{ display:inline-block; width:3ch; font-style:normal; }
.ayd-op-name{ color:var(--ayd-machine); letter-spacing:.04em; }
.ayd-op[data-plumb="1"] .ayd-op-name{ color:var(--ayd-muted); }
.ayd-op-detail{ color:var(--ayd-muted); }
.ayd-op-card{ text-align:right; color:var(--ayd-ink); font-size:.68rem; }
.ayd-op-card em{ font-style:normal; color:var(--ayd-muted); }
.ayd-op-card.none{ color:var(--ayd-line); }
.ayd-plan-foot{ margin-top:.55rem; padding-top:.5rem; border-top:1px solid var(--ayd-line);
  font-size:.7rem; color:var(--ayd-muted); line-height:1.55; }
.ayd-plan-foot b{ color:var(--ayd-machine); }

/* ---- result shape ------------------------------------------------------ */
/* What came back, in the warehouse's own type names (DESCRIBE, so the types are
   DuckDB's rather than pandas' guess at them), plus how much of the row cap the
   answer used. A dataframe with no shape line above it makes the reader count. */
.ayd-shape{ font-family:var(--ayd-mono) !important; font-size:.68rem; color:var(--ayd-muted);
  border-top:1px solid var(--ayd-line); padding-top:.45rem; margin:.55rem 0 .35rem;
  display:flex; flex-wrap:wrap; gap:.3rem .9rem; align-items:baseline; }
.ayd-shape b{ color:var(--ayd-ink); font-weight:500; }
/* The × between the two counts is read as content, not as a divider rule, so it
   stays at the muted colour that clears AA rather than dropping to the line
   colour the way a purely structural mark could. */
.ayd-shape em{ font-style:normal; color:var(--ayd-muted); }
.ayd-shape u{ text-decoration:none; color:var(--ayd-alert); }
.ayd-cols-list{ display:flex; flex-wrap:wrap; gap:.16rem .45rem;
  font-family:var(--ayd-mono) !important;
  font-size:.65rem; color:var(--ayd-muted); margin:0 0 .5rem; }
.ayd-coltype b{ color:var(--ayd-ink); font-weight:500; }
.ayd-coltype i{ font-style:normal; color:var(--ayd-machine); opacity:.8; }

/* ---- answer ------------------------------------------------------------ */
.ayd-answer{ font-family:var(--ayd-cond) !important; font-weight:600;
  font-size:1.75rem; line-height:1.22;
  color:var(--ayd-ink); margin:.15rem 0 .45rem; letter-spacing:-.01em;
  font-variant-numeric:tabular-nums; }
.ayd-verified{ display:inline-flex; align-items:center; gap:.45rem;
  font-family:var(--ayd-mono) !important;
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
  .ayd-step[data-on="cert"]{ animation:aydIn .28s ease-out; }
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
  .ayd-voice[data-ready="1"] .ayd-voice-orb{ animation:aydVoiceReady 2.8s ease-in-out infinite; }
  @keyframes aydVoiceReady{ 0%,100%{ box-shadow:0 0 0 0 rgba(34,211,238,0); }
    50%{ box-shadow:0 0 0 5px rgba(34,211,238,.08); } }
}

@media (max-width:900px){
  .ayd-cols, .ayd-row{ grid-template-columns:minmax(0,1fr) 30px 30px 96px 56px; }
}
@media (max-width:640px){
  .ayd-mast h1.ayd-title{ font-size:2rem; }
  .ayd-voice{ grid-template-columns:38px minmax(0,1fr); gap:.65rem;
    padding:.65rem .7rem; }
  .ayd-voice-orb{ width:34px; height:34px; }
  .ayd-voice-orb::after{ left:15px; }
  .ayd-voice-state{ grid-column:2; white-space:normal; margin-top:-.45rem; }
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
  /* 84px, not 74px. `supplychain` is the longest domain name in the manifest
     and it measures 77px at this size — at 74 it ellipsised to "supplychai…",
     which is a catalogue that cannot name its own contents. */
  .ayd-map-row{ grid-template-columns:84px minmax(0,1fr); }
  /* The correction that ended an attempt is a DuckDB error message and it is
     the content of that panel, so at 375px it takes its own line rather than
     being squeezed into a ~190px column and wrapped to six. */
  .ayd-att-row{ grid-template-columns:auto auto; row-gap:.18rem; column-gap:.8rem; }
  .ayd-att-why{ grid-column:1 / -1; }
  /* Same treatment for the verifier: the finding's message IS the panel, and a
     ~130px column at 375px wraps it to eight lines. The severity chip and the
     rule that fired take the first line; the sentence takes the rest. */
  .ayd-ver-row{ grid-template-columns:auto auto; row-gap:.18rem; column-gap:.7rem; }
  .ayd-ver-check{ text-align:left; }
  .ayd-ver-msg{ grid-column:1 / -1; }
  /* The compiler trace stacks, for the reason every other panel here stacks:
     a 5.6rem label column is 90px of a 375px screen, which left 159px for the
     binding and wrapped "hr_fact_employees — 1 join" onto two lines. The label
     is four characters of context; the binding is the content. */
  .ayd-trace-row{ display:grid; grid-template-columns:minmax(0,1fr);
    row-gap:.05rem; padding:.24rem 0; }
  .ayd-trace-key{ min-width:0; }
  .ayd-life{ grid-template-columns:repeat(2,minmax(0,1fr)); }
  .ayd-life-step{ border-bottom:1px solid var(--ayd-line); }
  .ayd-life-step:nth-child(2n){ border-right:0; }
  .ayd-life-step:nth-child(2n)::after{ display:none; }
  .ayd-life-step:last-child{ grid-column:1 / -1; border-bottom:0; }

  /* The domain/score pair drops under the question rather than squeezing it,
     and the SQL loses its hanging indent so the scroll track is full width. */
  .ayd-ex-top{ grid-template-columns:20px minmax(0,1fr); row-gap:.14rem; }
  .ayd-ex-meta{ grid-column:2; }
  .ayd-ex-sql{ margin-left:0; }
  .ayd-op{ grid-template-columns:minmax(0,1fr) 72px; gap:.5rem; }
  /* A ten-deep plan spends 10 × 3ch on guides before the operator name starts,
     which at 375px is most of the row. Two characters per level still carries
     the branch and gives the name back a third of the line. */
  .ayd-guide i{ width:2ch; }
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
    remembered to change.

    IT USED TO WATCH FOUR FILES — ui.py, streamlit_app.py, retrieval.py and the
    manifest — and that was not "the running source", it was a guess at which
    parts of it mattered. The guess failed the first time it was tested for
    real: the commit that added `engine/planner.py`, a 1,400-line engine that
    answers every question on the keyless path, touched none of the four and
    produced an IDENTICAL marker. So did the commit that added
    `engine/semantics.py`. Four consecutive commits, two of them entire new
    subsystems, all reported the same build — and a deploy marker that cannot
    see a new engine is worse than no marker, because it is consulted and
    believed.

    So it walks the whole tree now: every .py under app/ and engine/, the
    manifest, and the YAML contracts that drive what the app answers. The path
    is hashed alongside the bytes, so adding, renaming or deleting a file moves
    the marker too.

    Not git-derived on purpose — the deployed checkout may have no .git, and a
    marker that silently degrades to "unknown" answers nothing. Not cached
    either: hashing ~20 small files is under a millisecond, and a cached marker
    would go stale under Streamlit's hot reload, which is precisely when you are
    watching it.
    """
    import hashlib

    root = Path(__file__).resolve().parents[1]
    watched: list[Path] = []
    for folder in ("app", "engine"):
        watched.extend(sorted((root / folder).glob("*.py")))
    watched.append(root / "data_manifest.py")
    watched.append(root / "metrics.yaml")
    watched.extend(sorted((root / "evals").glob("*.yaml")))

    digest = hashlib.sha256()
    for path in sorted(watched, key=lambda p: p.relative_to(root).as_posix()):
        # The path goes in as well as the bytes: a file that appears or
        # disappears has to move the marker even if nothing else changed.
        digest.update(path.relative_to(root).as_posix().encode())
        try:
            # Normalise line endings before hashing. Git checks these files out
            # with CRLF on Windows and LF on the Linux box that runs the deploy,
            # so hashing raw bytes gave identical source two different markers —
            # which defeats the entire point of comparing local against live.
            raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            digest.update(raw)
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()[:7]


def masthead(*, tables: int, domains: int, live: bool, compact: bool = False) -> None:
    """The banner. `mode` is the one line on this page that says which engine ran.

    It used to read "demo · committed reference SQL" whenever no key was set,
    and that stopped being true the moment engine/planner.py started answering
    the chat box: keyless turns are compiled from the schema, not served from
    evals/golden_questions.yaml. A masthead that names the wrong engine is worse
    than one that names none, because it is the first claim a visitor reads and
    the SQL below it would quietly contradict it.
    """
    mode = ("live · model-authored SQL" if live
            else "keyless · compiled from the schema")
    st.markdown(
        f"""
<div class="ayd-mast" data-compact="{1 if compact else 0}">
  <div class="ayd-kicker">Ask your data</div>
  <h1 class="ayd-title">Ask governed data.<br>
    <span class="ayd-title-tail">See every step.</span></h1>
  <p class="ayd-sub">Voice or text across an enterprise analytics warehouse.
  Each question is scoped by access policy, grounded in the catalog, verified,
  executed read-only, and returned with the SQL and evidence behind the answer.</p>
  <div class="ayd-stats">
    <span><b>{tables}</b> tables</span>
    <span><b>{domains}</b> domains</span>
    <span>{mode}</span>
    <span>build <b>{build_marker()}</b></span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def view_header(kicker: str, title: str, description: str) -> None:
    """A compact, consistent orientation point for each workspace."""
    st.markdown(
        f'<div class="ayd-viewhead"><span class="k">{html.escape(kicker)}</span>'
        f'<h2>{html.escape(title)}</h2><p>{html.escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def lifecycle(*, live: bool) -> None:
    """The governed route an answer takes, visible before the first question."""
    producer = ("generate", "model writes SQL") if live else ("plan", "compiler writes SQL")
    stages = [
        ("ask", "voice or text", "input"),
        ("scope", "policy + catalog", "control"),
        (producer[0], producer[1], "work"),
        ("verify", "meaning checks", "control"),
        ("guard", "read-only gate", "control"),
        ("execute", "bounded query", "work"),
        ("evidence", "SQL + audit", "control"),
    ]
    body = "".join(
        f'<div class="ayd-life-step" data-stage="{kind}"><b>{html.escape(label)}</b>'
        f'<span>{html.escape(detail)}</span></div>'
        for label, detail, kind in stages
    )
    st.markdown(f'<div class="ayd-life ayd-hud">{body}</div>', unsafe_allow_html=True)


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
             planned: bool | str = False,
             certified: bool | str = False,
             guarded: bool | str = False, executed: bool = False,
             verified: bool | str | None = None,
             attempts: int = 1, timings: dict[str, float] | None = None) -> None:
    """The stages this turn actually went through.

    Not decoration: each cell is lit from what really happened, so a refusal
    shows GENERATE lit and EXECUTE dark, and a guard rejection shows GUARD in
    the alert colour with the retry count beside it.

    `timings` maps a stage name to measured wall-clock milliseconds. A stage
    with no entry shows no number rather than a zero, because "not measured"
    and "took no time" are different claims and this panel does not make the
    second one on the first one's behalf.

    `verified` is the VERIFY stage and it is opt-in — omitted entirely when
    None, which is not the same as dark. A dark cell claims a stage exists and
    did not run; on a path where engine.verify is genuinely not wired in, the
    honest strip is the one that never mentions it.

    It sits between GENERATE and GUARD because that is where the only BLOCKING
    half runs: Verifier.check_sql is structural and happens before run_query,
    which is what makes a refusal possible at all. The advisory half
    (check_result) runs after execution and cannot fail the turn, so it is
    reported in the verifier panel rather than given a second cell here — one
    cell for two moments would blur the order the strip exists to show.
    """
    timings = timings or {}

    def cell(label: str, state) -> str:
        on = ("fail" if state == "fail" else
              "cert" if state == "cert" else ("1" if state else "0"))
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

    # PLAN and GENERATE occupy the same position and are mutually exclusive,
    # because they are two different things that could have produced the SQL and
    # exactly one of them did. Lighting GENERATE for a compiled query would be
    # the single most misleading pixel in this app: it is the claim that a model
    # wrote something a grammar wrote. Keyless mode used to leave the cell dark
    # and disable the chat box; now it lights its own cell with its own name.
    if certified:
        stages = [("metric", "cert" if certified is True else certified)]
    elif planned:
        stages = [("retrieve", retrieved), ("plan", planned)]
    else:
        stages = [("retrieve", retrieved), ("generate", generated)]
    if verified is not None:
        stages.append(("verify", verified))
    stages += [("guard", guarded), ("execute", executed)]

    def link(left, right) -> str:
        """State of the segment BETWEEN two stages, read as "did anything cross".

        The rule is directional, and getting it wrong was measured: a first
        version marked BOTH segments touching a failed guard red, which claims
        that something crossed out of the guard and went wrong downstream.
        Nothing crossed — EXECUTE never ran.

          left failed          → fail, this is where the turn stopped
          left ran, right ran  → lit, the signal crossed
          left ran, right failed → lit; the SQL did reach the guard, and what
                                   the guard then decided is the guard cell's
                                   job to say, not this segment's
          otherwise            → dark

        Demo mode leaves GENERATE dark on purpose, so both segments touching it
        stay dark and the strip says "nothing was generated here" in the same
        breath as the cell does.
        """
        def state(value):
            return ("fail" if value == "fail" else
                    "1" if value in (True, "cert") else "0")

        a, b = state(left), state(right)
        if a == "fail":
            segment = "fail"
        elif a == "1" and b in ("1", "fail"):
            segment = "1"
        else:
            segment = "0"
        return f'<span class="ayd-arrow" data-on="{segment}"></span>'

    parts: list[str] = []
    for index, (label, state) in enumerate(stages):
        if index:
            parts.append(link(stages[index - 1][1], state))
        parts.append(cell(label, state))
    if attempts > 1:
        parts.append(f'<span class="ayd-step" data-on="fail">retry ×{attempts - 1}</span>')
    st.markdown(f'<div class="ayd-pipe">{"".join(parts)}</div>', unsafe_allow_html=True)


def column_list(columns, *, highlight: str = "") -> None:
    """Every column of one table, with its inferred role and sample values.

    `columns` is (name, role, values). The values are the ones
    engine/semantics.py indexed into the value lexicon, so what is shown here is
    literally what the compiler can bind a question to — not a sample chosen for
    display. `highlight` marks the substring that made this table match a
    search, which is what turns a list into an explanation.
    """
    rows = []
    for name, role, values in columns:
        shown = html.escape(name)
        if highlight and highlight in name.lower():
            start = name.lower().index(highlight)
            end = start + len(highlight)
            shown = (html.escape(name[:start]) + "<mark>"
                     + html.escape(name[start:end]) + "</mark>"
                     + html.escape(name[end:]))
        rows.append(
            f'<div class="ayd-colrow"><span class="ayd-colname">{shown}</span>'
            f'<span class="ayd-colrole">{html.escape(role)}</span></div>'
        )
        if values:
            joined = " · ".join(html.escape(str(v)) for v in values)
            rows.append(f'<div class="ayd-colvals">{joined}</div>')
    st.markdown(f'<div class="ayd-cols-browse">{"".join(rows)}</div>',
                unsafe_allow_html=True)


def layer_summary(cells: list[tuple[str, str, str]], *, footnote: str = "") -> None:
    """The semantic layer's own counts, for the empty state.

    `cells` is (number, label, gloss). Every value is passed in from
    engine.semantics' summary of the live warehouse — this component invents
    nothing, which is the only reason it is worth showing. It fills the space
    the landing page used to leave blank between the examples and the chat box,
    and it answers the question a visitor actually has there: what does this
    thing know before I ask it anything.
    """
    body = "".join(
        f'<div class="ayd-layer-cell"><span class="ayd-layer-n">{html.escape(n)}</span>'
        f'<span class="ayd-layer-k">{html.escape(k)}</span>'
        f'<span class="ayd-layer-w">{html.escape(w)}</span></div>'
        for n, k, w in cells
    )
    foot = (f'<div class="ayd-layer-foot">{html.escape(footnote)}</div>'
            if footnote else "")
    st.markdown(f'<div class="ayd-layer ayd-hud">{body}{foot}</div>',
                unsafe_allow_html=True)


def _fmt_compact(value: float) -> str:
    """A number short enough to sit at the end of a bar without moving it."""
    number = float(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(number) >= limit:
            scaled = number / limit
            return f"{scaled:,.1f}{suffix}" if abs(scaled) < 100 else f"{scaled:,.0f}{suffix}"
    if number == int(number):
        return f"{int(number):,}"
    return f"{number:,.2f}"


def result_chart(pairs: list[tuple[str, float]], *, label: str, measure: str,
                 kind: str = "bar", truncated_from: int = 0) -> None:
    """The answer as a shape, above the answer as a grid.

    "Total revenue by department" returns ten rows, and ten rows of digits ask
    the reader to do the comparison the question was asked in order to avoid.
    The grid stays — it is the auditable artifact, and it is what the CSV
    matches — but the ranking goes on top, because the shape IS the finding.

    Drawn here rather than delegated to a plotting library, and not to save a
    dependency. Every readout in this file is built from the same tokens, and a
    Vega or Plotly chart arrives with its own palette, its own type stack and
    its own idea of a gridline — three quiet contradictions of the design
    system, on the most prominent element of the turn. One accent, tabular
    figures, and the value printed at the end of every bar so a length never
    has to be estimated to be read.

    `pairs` is already ordered by the caller: the query's own ORDER BY is the
    ranking the reader asked for, and re-sorting here would draw a different
    query's answer.
    """
    if len(pairs) < 2:
        return
    magnitudes = [abs(float(v)) for _, v in pairs]
    widest = max(magnitudes) or 1.0
    # A chart of negatives, or of a mix, is a different chart — a baseline in
    # the middle, and a bar that means "less than nothing" rather than "less".
    # Rather than draw the bar chart wrong, this draws nothing and leaves the
    # grid, which is honest about a shape this component does not do yet.
    if any(float(v) < 0 for _, v in pairs):
        return

    row_h, gap, pad_l, pad_r = 20, 4, 148, 62
    height = len(pairs) * (row_h + gap)
    width = 620
    track = width - pad_l - pad_r

    def clip(text: str, limit: int = 22) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    body = []
    if kind == "line":
        # A series over time reads as a line; a ranking reads as bars. The
        # caller decides which, from the column's DuckDB type, because a date
        # axis is a fact about the data and not a preference.
        plot_h = max(90, min(190, height))
        step = track / max(len(pairs) - 1, 1)
        points = " ".join(
            f"{pad_l + i * step:.1f},"
            f"{plot_h - (abs(float(v)) / widest) * (plot_h - 16) - 8:.1f}"
            for i, (_, v) in enumerate(pairs))
        body.append(f'<polyline class="line" points="{points}"/>')
        for i, (name, value) in enumerate(pairs):
            x = pad_l + i * step
            y = plot_h - (abs(float(value)) / widest) * (plot_h - 16) - 8
            body.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="2"/>')
            if i in (0, len(pairs) - 1):
                anchor = "start" if i == 0 else "end"
                body.append(
                    f'<text class="cat" x="{x:.1f}" y="{plot_h + 12}" '
                    f'text-anchor="{anchor}">{html.escape(clip(name, 16))}</text>')
        body.append(f'<line class="axis" x1="{pad_l}" y1="{plot_h}" '
                    f'x2="{width - pad_r}" y2="{plot_h}"/>')
        view_h = plot_h + 18
    else:
        for i, (name, value) in enumerate(pairs):
            y = i * (row_h + gap)
            length = max(1.0, track * abs(float(value)) / widest)
            top = " bar-top" if i == 0 else ""
            body.append(
                f'<text class="cat" x="0" y="{y + 13}">{html.escape(clip(name))}</text>'
                f'<rect class="track" x="{pad_l}" y="{y + 3}" width="{track}" '
                f'height="{row_h - 6}" rx="1"/>'
                f'<rect class="bar{top}" x="{pad_l}" y="{y + 3}" width="{length:.1f}" '
                f'height="{row_h - 6}" rx="1"/>'
                f'<text class="val" x="{width}" y="{y + 13}">'
                f'{html.escape(_fmt_compact(value))}</text>')
        view_h = height

    foot = ""
    if truncated_from:
        foot = (f'<div class="ayd-chart-foot">showing {len(pairs)} of '
                f'{truncated_from:,} rows — the grid below has the rest</div>')
    st.markdown(
        f'<div class="ayd-chart ayd-hud">'
        f'<div class="ayd-chart-head"><span>{html.escape(label)}</span>'
        f'<span>{html.escape(measure)}</span></div>'
        f'<svg viewBox="0 0 {width} {view_h}" role="img" '
        f'aria-label="{html.escape(measure)} by {html.escape(label)}, '
        f'{len(pairs)} values, highest {html.escape(clip(str(pairs[0][0])))}">'
        f'{"".join(body)}</svg>{foot}</div>',
        unsafe_allow_html=True,
    )


def operations(summary: dict, *, limits: list[str] | None = None) -> None:
    """The service, rather than the request.

    Every other readout in this file describes ONE turn. This one describes the
    process: how many questions it has been asked, how many it refused, how
    long the stages really took at the median, and which engine answered. It
    exists because `engine/audit.py` now keeps a durable record — before that
    there was nothing to aggregate over, and a panel of aggregates computed
    from nothing is the exact kind of decoration this app spends its whole
    interface arguing against.

    The refusal rate is the number to watch here, and it is deliberately given
    a cell of its own. It is the price this system pays for never returning a
    wrong number, and it is the first thing that moves when the grammar or the
    confidence gate changes.

    `limits` is not optional in spirit. An audit trail is a control, and a
    control whose weaknesses are undocumented is how a reviewer ends up relying
    on something that was never load-bearing.
    """
    turns = int(summary.get("turns", 0) or 0)
    if not turns:
        st.markdown(
            '<div class="ayd-ops ayd-hud"><div class="ayd-ops-foot">'
            'No turns recorded yet this session. Ask a question and this panel '
            'fills from the audit record of what actually ran — not from a '
            'sample.</div></div>',
            unsafe_allow_html=True,
        )
        return

    def cell(number: str, label: str) -> str:
        return (f'<div class="ayd-ops-cell"><span class="ayd-ops-n">'
                f'{html.escape(number)}</span>'
                f'<span class="ayd-ops-k">{html.escape(label)}</span></div>')

    cells = [
        cell(f"{turns:,}", "turns"),
        cell(f"{100 * float(summary.get('refusal_rate', 0.0)):.0f}%", "refused"),
        cell(f"{float(summary.get('p50_ms', 0.0)):,.0f}ms", "p50"),
        cell(f"{float(summary.get('p95_ms', 0.0)):,.0f}ms", "p95"),
    ]
    blocked = int(summary.get("blocked", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    if blocked:
        cells.append(cell(f"{blocked:,}", "guard blocks"))
    if failed:
        cells.append(cell(f"{failed:,}", "errors"))
    tokens = int(summary.get("tokens_in", 0) or 0) + int(summary.get("tokens_out", 0) or 0)
    # Only shown once a model has actually been used. A keyless session that
    # reports "0 tokens" is inviting the reader to wonder what it would have
    # been; a keyless session that reports nothing is telling the truth that
    # this axis does not apply to it.
    if tokens:
        cells.append(cell(f"{tokens:,}", "tokens"))

    stages = summary.get("stage_p50_ms") or {}
    rows = ""
    if stages:
        widest = max(stages.values()) or 1.0
        rows = "".join(
            f'<div class="ayd-ops-row"><b>{html.escape(name)}</b>'
            f'<span class="ayd-ops-track"><span class="ayd-ops-fill" '
            f'style="width:{max(2.0, 100.0 * value / widest):.1f}%"></span></span>'
            f'<span class="ayd-ops-num">{value:,.1f}ms</span></div>'
            for name, value in stages.items()
        )
        rows = (f'<div class="ayd-ops-head">median per stage</div>'
                f'<div class="ayd-ops-rows">{rows}</div>')

    mix = summary.get("engines") or {}
    kinds = summary.get("refusal_kinds") or {}
    lines = []
    if mix:
        lines.append("answered by " + ", ".join(
            f"{k} × {v}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])))
    if kinds:
        lines.append("refused because " + ", ".join(
            f"{k} × {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    sink = summary.get("sink") or ""
    lines.append(f"sink: {sink}" if sink else
                 "sink: in-memory ring only — set ASK_YOUR_DATA_AUDIT to a path to persist")
    if summary.get("sink_error"):
        lines.append(f"audit sink error: {summary['sink_error']}")
    automation = summary.get("automation") or {}
    if automation.get("configured"):
        lines.append(
            "n8n operations: "
            f"{int(automation.get('delivered', 0))} delivered, "
            f"{int(automation.get('failed', 0))} failed, "
            f"{int(automation.get('dropped', 0))} dropped"
        )
    body = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    for note_line in (limits or []):
        body += f"<li>{html.escape(note_line)}</li>"

    st.markdown(
        f'<div class="ayd-ops ayd-hud">'
        f'<div class="ayd-ops-grid">{"".join(cells)}</div>'
        f'{rows}'
        f'<div class="ayd-ops-foot"><ul>{body}</ul></div></div>',
        unsafe_allow_html=True,
    )


def refusal(reason: str, *, kind: str = "not compiled") -> None:
    """The compiler declined, said why, and that is a result rather than an error.

    Deliberately not st.warning. A refusal is the designed behaviour of this
    app — the thing its README argues for — and painting it in Streamlit's
    olive alert made it the loudest element on the page, in a hue this palette
    does not otherwise use.

    It is also deliberately not amber and not red. Amber is reserved for "a
    committed file backs this number" and red for a failure; a refusal is
    neither, so the panel is neutral and its mono heading carries the meaning.
    """
    st.markdown(
        f"""
<div class="ayd-refusal ayd-hud">
  <div class="ayd-refusal-head">{html.escape(kind)}</div>
  <div class="ayd-refusal-body">{html.escape(reason)}</div>
</div>""",
        unsafe_allow_html=True,
    )


def plan_trace(rationale, *, coverage: float, considered: int,
               bound: list[str], missed: list[str], loose: list[str],
               plan_ms: float | None = None, refused: bool = False) -> None:
    """Every binding the compiler made, and the words that paid for each one.

    The panel a language model cannot honestly produce. `rationale` comes off
    `Plan.rationale()` -- the plan's own record of which column it chose for
    which role -- and the word chips underneath are the coverage arithmetic
    itself: which words were bound, which the warehouse has no vocabulary for,
    and which it knows but this plan did not use. The last group is the only one
    that is a debt, and it is drawn in the alert colour for that reason.

    On a refusal the same panel renders with whatever the nearest plan managed,
    which is the useful thing to show: "this is how far I got" beats an apology.
    """
    rows = "".join(
        f'<div class="ayd-trace-row"><span class="ayd-trace-key">{html.escape(key)}</span>'
        f'<span class="ayd-trace-val">{html.escape(str(value))}</span></div>'
        for key, value in rationale
    )
    chips = "".join(
        f'<span class="ayd-word" data-w="{kind}">{html.escape(word)}</span>'
        for kind, words in (("bound", bound), ("missed", missed), ("loose", loose))
        for word in words
    )
    stamp = f" · {plan_ms:,.0f}ms" if plan_ms is not None and plan_ms >= 1 else ""
    verdict = "no plan met the floor" if refused else f"{coverage:.0%} of the question bound"
    st.markdown(
        f"""
<div class="ayd-trace ayd-hud">
  <div class="ayd-trace-head">compiled without a model · <b>{html.escape(verdict)}</b>
  — {considered} candidate table{'' if considered == 1 else 's'} considered{stamp}</div>
  {rows}
  {f'<div class="ayd-words">{chips}</div>' if chips else ''}
</div>""",
        unsafe_allow_html=True,
    )


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
              pool: int | None = None, candidates: int | None = None) -> None:
    """What the model was allowed to see, why each table is there, and the cost.

    `hits` are RetrievedTable records from engine.retrieval.retrieve_hybrid, in
    fused order. `vector_ranks` and `keyword_ranks` are that table's position in
    each of the two input rankings - the only two inputs RRF has - and `pool` is
    how deep each ranking was read (retrieve_hybrid uses max(k*2, 12)).

    Ranks are optional. If the caller cannot produce them the panel degrades to
    the table list and the token accounting, which is still true; it does not
    invent a rank to keep the layout tidy.

    `candidates` is the size of the set the fusion CHOSE FROM: the union of the
    two rankings, every table at least one retriever put inside the pool. It is
    the middle number of a three-stage funnel the panel could previously only
    show the ends of. Measured over the 39 golden questions with k=10 and a pool
    of 20, the union runs 21..35 distinct tables (median 28) — 30% to 49% of
    this 71-table catalogue. So "10 of 71" was hiding the fact that eighteen
    other tables were ranked and then dropped, and dropping them is what the
    fusion is for. One clause, no new rows: the funnel is a fact about the
    ranking already drawn above it, not a fourth ranking to draw.
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
    for hit in hits:
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
            f'<div class="ayd-tbl">{html.escape(hit.table)} '
            f'<i>· {html.escape(hit.domain)}</i></div>'
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
    # On a compiled turn there is no prompt and there are no tokens, so the
    # saving line is replaced rather than recomputed. Leaving it would have the
    # keyless app boasting about a prompt budget it never spends -- a true
    # sentence about the model path, printed under a turn the model had no part
    # in, which is the kind of claim this panel exists to make impossible.
    saving = (
        f'~<b>{tokens_used:,}</b> tokens of schema in the prompt, against '
        f'~{tokens_full:,} for the whole catalogue — <b>{pct}%</b> smaller.'
        if tokens_used else
        'No prompt and no tokens on this turn: the same ranking chose which '
        'tables the compiler was allowed to plan against.'
    )
    fusion_note = (
        f'<br><b>{both}</b> of {len(hits)} were ranked by both retrievers; '
        f'the rest were found by only one — which is the reason both are run.'
    ) if have_ranks else ""
    # The funnel's middle stage. Only stated when the caller measured it, and
    # stated with the depth beside it, because a candidate count means nothing
    # without knowing how deep each ranking was read to produce it.
    funnel = (
        f' Reading {pool} deep, the two rankings proposed <b>{candidates}</b> '
        f'distinct tables between them; the fusion kept these {len(hits)}.'
    ) if (have_ranks and candidates) else ""

    st.markdown(
        f"""
<div class="ayd-ground-panel ayd-hud">
  <div class="ayd-ground-head">
    <span>grounding · <b>{len(hits)}</b> of {total_tables} tables retrieved</span>
    <span>reciprocal rank fusion</span>
  </div>
  {header}
  {''.join(rows)}
  <div class="ayd-saving">{saving}{fusion_note}{funnel}</div>
</div>""",
        unsafe_allow_html=True,
    )


def schema_map(by_domain: dict[str, list[tuple[str, bool]]], *,
               retrieved: int, total: int,
               destination: str = "sent to the model") -> None:
    """One cell per table in the warehouse, lit where the retriever selected it.

    The grounding panel says "10 of 71". This draws the 71. The claim the whole
    retrieval module rests on is that most of a warehouse is irrelevant to any
    one question, and a wall of unlit cells argues that better than a ratio.

    `destination` names where the selected tables WENT, and it is a parameter
    because there are now two answers. The model path sends them to the model;
    the keyless path hands them to engine/planner.py, which never talks to
    anything. The header read "sent to the model" unconditionally, which on a
    compiled turn described a network call that did not happen.
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
    <span>catalogue · <b>{retrieved}</b> of {total} tables {destination}</span>
    <span>{len(by_domain)} domains</span>
  </div>
  {''.join(rows)}
</div>""",
        unsafe_allow_html=True,
    )


def guard_verdict(*, ok: bool, reason: str, checks: list[tuple[str, bool]],
                  forbidden: list[str] | None = None, blocked_verb: str = "") -> None:
    """What engine.sql_guard.validate_sql checked, and what it returned.

    `checks` are (label, passed) pairs describing the individual conditions.
    They are recomputed by the caller from the same module the guard uses, so
    this is a readout of the boundary rather than a decorative reassurance that
    one exists.

    `forbidden` is sql_guard.FORBIDDEN itself, drawn out verb by verb. The
    schema map's argument applied to the guard: "none of 26 forbidden verbs" is
    a claim, and the 26 on screen are the evidence for it. `blocked_verb` is the
    one the guard actually named, when it named one — it is not re-derived here
    by scanning the SQL, because a second scanner that disagreed with the guard
    would be showing a boundary the app does not actually enforce.

    Both are optional; passing neither renders exactly what this function
    rendered before they existed.
    """
    items = "".join(
        f'<span class="ayd-check" data-pass="{1 if passed else 0}">{html.escape(label)}</span>'
        for label, passed in checks
    )
    verbs = ""
    if forbidden:
        hit = blocked_verb.strip().upper()
        verbs = '<div class="ayd-verbs">' + "".join(
            f'<span class="ayd-verb" data-hit="{1 if verb.upper() == hit else 0}">'
            f'{html.escape(verb)}</span>'
            for verb in forbidden
        ) + "</div>"
    verdict = "pass" if ok else "blocked"
    st.markdown(
        f"""
<div class="ayd-guard ayd-hud" data-ok="{1 if ok else 0}">
  <div class="ayd-guard-head">read-only guard · <b data-ok="{1 if ok else 0}">{verdict}</b>
  — {html.escape(reason)}</div>
  <div class="ayd-checks">{items}</div>
  {verbs}
</div>""",
        unsafe_allow_html=True,
    )


# The three rungs of engine.verify's severity ladder, worst first. Kept as a
# plain tuple rather than imported so this module stays free of engine imports;
# the caller passes severities as strings and _SEVERITY_ORDER only decides which
# of them is the worst one present.
_SEVERITY_ORDER = ("error", "warn", "note")


def verification(findings, *, checks=None, verify_ms: float | None = None,
                 refused: bool = False, ran: bool = True) -> None:
    """What engine.verify decided about the query that just ran.

    `findings` is a list of (check, severity, message) triples — plain tuples,
    not Finding objects, so this module keeps its rule of importing nothing from
    engine. `checks` is the roster of rules that could have fired, as (name,
    severity) pairs; passing it draws the board, omitting it renders only the
    findings.

    `refused` says this finding set ENDED the turn. That is a genuinely
    different thing from an advisory note travelling with an answer, and it is
    the state the panel exists for: engine.assistant's final-attempt path
    returns refused=True with the blocking finding as the reason, having
    declined to execute SQL it could not stand behind. So the panel says the
    query was written and not run, and the caller shows it unrun below.

    `ran=False` renders nothing at all. A turn where the verifier did not run —
    a refusal before any SQL existed — must not draw an empty board, because a
    board of quiet rules is a claim that they were checked.
    """
    if not ran:
        return
    findings = list(findings or [])
    worst = next((level for level in _SEVERITY_ORDER
                  if any(str(s) == level for _c, s, _m in findings)), "clean")
    fired = {str(check): str(severity) for check, severity, _m in findings}

    rows = "".join(
        f'<div class="ayd-ver-row">'
        f'<div class="ayd-sev" data-sev="{html.escape(str(severity))}">'
        f'{html.escape(str(severity))}</div>'
        f'<div class="ayd-ver-check">{html.escape(str(check))}</div>'
        f'<div class="ayd-ver-msg">{html.escape(" ".join(str(message).split()))}</div>'
        f'</div>'
        for check, severity, message in findings
    )

    board = ""
    if checks:
        board = '<div class="ayd-rules">' + "".join(
            f'<span class="ayd-rule" data-hit="{html.escape(fired.get(str(name), "0"))}">'
            f'{html.escape(str(name))}</span>'
            for name, _severity in checks
        ) + "</div>"

    count = len(findings)
    # "clean" is not the same sentence as "one note", and neither is the same as
    # "blocked". The head states the outturn in words rather than leaving the
    # reader to infer it from a border colour they may not have learned yet.
    if refused:
        verdict = "refused"
    elif not count:
        verdict = "no findings"
    else:
        verdict = f"{count} finding{'' if count == 1 else 's'}"
    alert = "1" if worst in ("error", "warn") else "0"
    stamp = (f'{verify_ms:,.2f}ms' if verify_ms is not None and verify_ms >= 0.01
             else ("&lt;0.01ms" if verify_ms is not None else "structural + result"))
    scope = f'{len(checks)} rules' if checks else 'structural + result'

    refusal = ""
    if refused:
        refusal = (
            '<p class="ayd-ver-refused">The loop ran out of attempts with this '
            'still unresolved, so the query below was written and never executed. '
            'Running it would have produced a number and a confident sentence '
            'about it, which is the failure this layer exists to prevent.</p>'
        )

    st.markdown(
        f"""
<div class="ayd-ver ayd-hud" data-worst="{worst}">
  <div class="ayd-ver-head">
    <span>verifier · <b data-alert="{alert}">{verdict}</b> · {scope}</span>
    <span>{stamp}</span>
  </div>
  {refusal}{rows}{board}
</div>""",
        unsafe_allow_html=True,
    )


def exemplars(picks, *, corpus: int, in_prompt: bool, fused: bool = False,
              select_ms: float | None = None) -> None:
    """The solved questions the few-shot selector put nearest this one.

    `picks` are (question, domain, sql, score) tuples in selection order, best
    first — again plain tuples rather than engine.exemplars.Exemplar, so this
    module imports nothing from engine.

    `in_prompt` is the honesty switch and it is not decoration. In live mode
    these pairs are really in the system prompt and the panel says so. In demo
    mode no prompt is built at all — no model runs — so the panel says what is
    true there instead: this is the bank's ranking for this question, with the
    question's own pair excluded by the leave-one-out rule that makes an eval
    over this corpus mean anything.

    `score` is the cosine similarity returned by the local exact index, which is
    the text signal only. When the selector also had retrieved tables to fuse
    against, the ORDER is the fused one and the score is not what produced it —
    so the score is labelled `sim` rather than presented as the ranking key.

    `fused` is that second signal, and it is a parameter rather than a constant
    because the two callers genuinely differ. Demo mode hands select_exemplars
    the tables retrieval just chose. engine.assistant hands it the tables PRIOR
    SQL used, which on a first question is the empty tuple — so the same panel
    would be describing a text-only ranking there. Naming a signal that was not
    used is the one thing a readout of a mechanism must not do, so the caller
    that knows says which it was.
    """
    if not picks:
        return
    rows = []
    for index, (question, domain, sql, score) in enumerate(picks, 1):
        meta = f'{html.escape(str(domain))} · sim <b>{float(score):.2f}</b>'
        rows.append(
            f'<div class="ayd-ex-row">'
            f'<div class="ayd-ex-top">'
            f'<div class="ayd-ex-n">{index}</div>'
            f'<div class="ayd-ex-q">{html.escape(str(question))}</div>'
            f'<div class="ayd-ex-meta">{meta}</div>'
            f'</div>'
            f'<div class="ayd-ex-sql">{html.escape(" ".join(str(sql).split()))}</div>'
            f'</div>'
        )
    where = ("in this turn's system prompt" if in_prompt
             else "no prompt is built in demo mode")
    stamp = f'selected in {select_ms:,.0f}ms' if select_ms is not None else 'rrf'
    ranking = ("ranked by embedding similarity, fused by reciprocal rank with the "
               "tables retrieval selected for this question"
               if fused else
               "ranked by embedding similarity alone — no table signal was "
               "available to fuse against")
    foot = (
        f'<div class="ayd-ex-foot">Verified question/SQL pairs from '
        f'<b>evals/golden_questions.yaml</b>, the same file the accuracy contract '
        f'asserts against — {ranking}. A question is never shown its own pair.</div>'
    )
    st.markdown(
        f"""
<div class="ayd-ex ayd-hud">
  <div class="ayd-ex-head">
    <span>few-shot bank · <b>{len(picks)}</b> of {corpus} solved questions ·
    {html.escape(where)}</span>
    <span>{html.escape(stamp)}</span>
  </div>
  {''.join(rows)}
  {foot}
</div>""",
        unsafe_allow_html=True,
    )


def attempt_ledger(*, attempts: int, corrections: list[str], max_attempts: int,
                   ok: bool = True) -> None:
    """The self-correction loop, one row per attempt, in the order it ran.

    engine.assistant appends exactly one entry to `corrections` for every
    attempt that did not survive — a DuckDB error, or the verifier's correction
    note for SQL that ran and was wrong — and then either returns on the next
    attempt or exhausts MAX_ATTEMPTS. So the ledger is fully determined by
    (attempts, corrections): the first len(corrections) rows are the failures
    with the reason the loop fed back to the model, and the last row is the
    attempt that was accepted.

    Renders nothing for a single clean attempt. There is no ledger to show when
    nothing was corrected, and a panel reading "attempt 1: fine" would be
    ceremony.

    An exhausted loop has one correction for every attempt, so every row is a
    failure and there is no final row to label — the `ok=False` wording is for
    the other way the loop ends early, a refusal on a later attempt, where the
    model stopped rather than erroring.
    """
    if attempts <= 1 and not corrections:
        return
    rows = []
    for index in range(1, attempts + 1):
        failed = index <= len(corrections)
        why = corrections[index - 1] if failed else (
            "accepted — guard passed and the query returned rows"
            if ok else "the loop ended here without an accepted query"
        )
        # A DuckDB error can be a paragraph. The first line is the diagnosis and
        # the rest is a position marker that means nothing without the original
        # cursor, so the row shows the line and the title carries the whole text.
        head = str(why).strip().splitlines()[0]
        short = head if len(head) <= 150 else head[:149] + "…"
        rows.append(
            f'<div class="ayd-att-row" data-ok="{0 if failed else 1}">'
            f'<div class="ayd-att-n">attempt {index}</div>'
            f'<div class="ayd-att-v">{"failed" if failed else ("ran" if ok else "stopped")}</div>'
            f'<div class="ayd-att-why" title="{html.escape(str(why).strip())}">'
            f'{html.escape(short)}</div>'
            f'</div>'
        )
    st.markdown(
        f"""
<div class="ayd-att ayd-hud">
  <div class="ayd-att-head">self-correction · <b>{attempts}</b> of
  {max_attempts} attempts used</div>
  {''.join(rows)}
</div>""",
        unsafe_allow_html=True,
    )


def query_plan(nodes: list[dict], *, plan_ms: float | None = None,
               total: int | None = None,
               returned: int | None = None, truncated: bool = False) -> None:
    """DuckDB's physical operator tree for the SQL that ran.

    `nodes` is a flat depth-first list of dicts — guide, name, detail, card —
    produced by the caller from EXPLAIN (FORMAT JSON). The tree structure lives
    in the pre-rendered `guide` string rather than in an indent level, because
    the vertical continuation line of a two-child operator (a HASH_JOIN) cannot
    be drawn by a row that does not know its ancestors' sibling counts.

    `card` is the optimiser's ESTIMATED cardinality and is labelled as an
    estimate everywhere it appears. The panel foot puts the root's estimate next
    to the row count the query genuinely returned, which is the only honest way
    to show a prediction: with its outturn beside it.
    """
    if not nodes:
        return
    def guide_cells(guide: str) -> str:
        """One fixed 3ch box per tree level.

        The guide arrives as three characters per level — "│  ", "   ", "└─ ",
        "├─ " — so chunking by three is exact rather than a heuristic, and each
        chunk gets a box the monospace grid defines instead of relying on the
        corner glyph to be as wide as a space. It is not: see the note in the
        stylesheet.
        """
        return "".join(f'<i>{html.escape(guide[i:i + 3])}</i>'
                       for i in range(0, len(guide), 3))

    rows = []
    for node in nodes:
        name = str(node.get("name", ""))
        detail = str(node.get("detail", ""))
        card = node.get("card")
        plumb = 1 if name == "PROJECTION" else 0
        short = detail if len(detail) <= 110 else detail[:109] + "…"
        card_cell = (f'<div class="ayd-op-card"><em>~</em>{card:,}</div>'
                     if isinstance(card, int)
                     else '<div class="ayd-op-card none">—</div>')
        rows.append(
            f'<div class="ayd-op" data-plumb="{plumb}">'
            f'<div class="ayd-op-l" title="{html.escape(detail or name)}">'
            f'<span class="ayd-guide">{guide_cells(str(node.get("guide", "")))}</span>'
            f'<span class="ayd-op-name">{html.escape(name)}</span>'
            + (f' <span class="ayd-op-detail">{html.escape(short)}</span>' if short else "")
            + '</div>' + card_cell + '</div>'
        )

    root_card = nodes[0].get("card")
    foot = ""
    if isinstance(root_card, int) and returned is not None:
        outturn = (f'the row cap stopped the read at <b>{returned:,}</b>' if truncated
                   else f'<b>{returned:,}</b> came back')
        foot = (f'<div class="ayd-plan-foot">the optimiser estimated '
                f'<b>{root_card:,}</b> row{"" if root_card == 1 else "s"} at the top '
                f'of this plan; {outturn}.</div>')

    stamp = f'plan read in {plan_ms:,.1f}ms' if plan_ms is not None else 'EXPLAIN'
    # A renderer that stops at its own cap without saying so is claiming the
    # plan ended where it gave up. No golden query comes close - the deepest is
    # 36 operators against a cap of 200 - but model-authored SQL is not bounded
    # by the golden set, and the panel should not start lying the first time it
    # meets something bigger than what was measured.
    count = (f'<b>{len(nodes)}</b> of {total:,} operators shown'
             if total is not None and total > len(nodes)
             else f'<b>{len(nodes)}</b> operators')
    st.markdown(
        f"""
<div class="ayd-plan ayd-hud">
  <div class="ayd-plan-head">
    <span>physical plan · {count} · duckdb</span>
    <span>{html.escape(stamp)} · est. rows</span>
  </div>
  <div class="ayd-plan-body">{''.join(rows)}</div>
  {foot}
</div>""",
        unsafe_allow_html=True,
    )


def result_shape(columns: list[tuple[str, str]], *, rows: int, truncated: bool,
                 cap: int) -> None:
    """The shape and the column types of what came back.

    Types are DuckDB's own, from DESCRIBE against the same statement, not
    pandas' inference from the values — a column of integers that DuckDB calls
    BIGINT and pandas calls int64 is being described by two different systems,
    and only one of them is the warehouse.
    """
    if not columns:
        return
    noun = "row" if rows == 1 else "rows"
    cols = "column" if len(columns) == 1 else "columns"
    head = (f'<b>{rows:,}</b> {noun} <em>×</em> <b>{len(columns)}</b> {cols}'
            if not truncated else
            f'<u>{rows:,} {noun} shown — cap of {cap:,} reached</u> '
            f'<em>×</em> <b>{len(columns)}</b> {cols}')
    # An unaliased aggregate gets its whole expression as a column name - DuckDB
    # returned a 130-character `round(((100.0 * count_star() FILTER …)))` from
    # the first golden question - and a name that long is not a name, it is the
    # SQL again. It is clipped to a scannable head with the full text on hover;
    # the SQL is on screen directly above, so nothing is actually lost.
    def label(name: str) -> str:
        text = " ".join(str(name).split())
        return text if len(text) <= 34 else text[:33] + "…"

    cells = "".join(
        f'<span class="ayd-coltype" title="{html.escape(str(name))}">'
        f'<b>{html.escape(label(name))}</b> '
        f'<i>{html.escape(str(kind))}</i></span>'
        for name, kind in columns
    )
    st.markdown(
        f'<div class="ayd-shape"><span>result · {head}</span>'
        f'<span>row cap {cap:,}</span></div>'
        f'<div class="ayd-cols-list">{cells}</div>',
        unsafe_allow_html=True,
    )


def voice_dock(*, ready: bool, stt_model: str, tts_model: str) -> None:
    """The state of the optional voice edge around the governed query path."""
    state = "ready" if ready else "add key in sidebar"
    detail = (f"{stt_model} → review transcript → governed query → {tts_model}"
              if ready else "Recordings are sent only after voice is enabled")
    st.markdown(
        f"""
<div class="ayd-voice" data-ready="{'1' if ready else '0'}">
  <span class="ayd-voice-orb" aria-hidden="true"></span>
  <span class="ayd-voice-copy"><b>Ask by voice</b><span>{html.escape(detail)}</span></span>
  <span class="ayd-voice-state">{html.escape(state)}</span>
</div>""",
        unsafe_allow_html=True,
    )


def metric_definition(*, label: str, owner: str, definition: str,
                      derived_value, derived_why: str) -> None:
    """The policy behind a certified number, beside the schema-only contrast."""
    comparison = ("The schema-only compiler refuses this metric."
                  if derived_value is None
                  else f"Schema-only result: {derived_value}")
    st.markdown(
        f"""
<div class="ayd-metric ayd-hud">
  <div class="ayd-metric-head">certified metric <span>owner · {html.escape(owner)}</span></div>
  <div class="ayd-metric-def"><b>{html.escape(label)}</b> — {html.escape(definition)}</div>
  <div class="ayd-metric-compare"><b>{html.escape(comparison)}</b><br>
    {html.escape(derived_why)}</div>
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
