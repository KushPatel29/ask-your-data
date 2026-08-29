"""
The natural-language layer: turn a plain-English question into SQL, run it, and
answer from the result.

Design choices that keep this honest:

- The model's ONLY job is to write SQL. It never states a number from its own
  head — every figure in the final answer comes from a query that actually ran
  against the warehouse, and the SQL is returned alongside the answer so it can
  be audited. If the question can't be answered from the loaded tables, the
  model says so instead of inventing one.
- **Bounded self-correction.** When the SQL fails (a mistyped column, a guard
  block), the database error is fed back to the model for a corrected attempt —
  at most MAX_ATTEMPTS total, never an unbounded agent loop. Every failed
  attempt is kept on the result for transparency.
- **Conversation memory.** Follow-up questions ("and by region?") work because
  prior turns — question, the SQL used, the answer — are passed back as context.
  The caller owns the history, so the Assistant itself stays stateless and
  thread-safe.
- **Retrieved schema grounding.** A local embedding index selects the nine
  tables most relevant to the current question instead of sending all 36 on
  every turn. Tables used by prior-turn SQL are always retained for follow-ups.

The model is asked for structured output via a tool, so we get clean SQL (or a
refusal) rather than having to scrape it out of prose.
"""

import os
import re
from dataclasses import dataclass, field

# No `import anthropic` anywhere below, and that absence is the point: after the
# provider seam this module has no vendor SDK dependency at all. The Anthropic
# client lives in engine/providers.py alongside every other backend, so swapping
# the model never touches the reasoning loop.
from engine.exemplars import exemplar_block
from engine.providers import AnthropicProvider, ProviderUnavailable, build_provider
from engine.query import QueryResult, run_query
from engine.retrieval import schema_catalog_for
from engine.verify import Verifier, correction_message
from engine.warehouse import table_names

# Opus 5. Thinking is ON BY DEFAULT on this model - omitting the parameter runs
# adaptive - which is a real behaviour change from opus-4-8, where omitting it
# meant no thinking at all.
#
# Thinking is deliberately left on. The documented failure mode of disabling it
# on Opus 5 is that the model sometimes writes a tool call into its VISIBLE TEXT
# instead of emitting a tool_use block: the turn succeeds, the call never runs,
# and nothing raises. This assistant reads tool_use blocks and nothing else
# (see the `tool_use = next(...)` line in ask()), so that failure would present
# as a silent refusal. Lower `effort` is the supported way to spend less.
MODEL = os.environ.get("ASK_YOUR_DATA_MODEL", "claude-opus-5")

# Thinking and visible output share this budget. At the old value of 2048 a
# thinking model could spend the allowance reasoning and get truncated before
# emitting the tool call.
MAX_OUTPUT_TOKENS = 16000
MAX_ATTEMPTS = 3   # 1 initial attempt + up to 2 corrections
HISTORY_TURNS = 6  # how many prior turns are replayed as context

SYSTEM_RULES = """You are a careful analytics engineer answering questions about a \
read-only DuckDB warehouse by writing SQL.

Rules:
- Use ONLY the tables and columns listed in the schema. Never invent a column.
- Table names are exactly as written — they are domain-prefixed (e.g.
  healthcare_fact_claims, hr_fact_employees). The same base name can exist in
  several domains, so always use the full prefixed name.
- Write DuckDB SQL. SELECT statements only — no INSERT/UPDATE/DDL. If the user
  asks you to modify, delete, or export data, call cannot_answer and explain
  that this is a read-only interface.
- Read the column descriptions carefully. For example, pending healthcare claims
  have blank allowed_amount/paid_amount; net collection rate is paid/allowed.
- For "top", "most", "highest" questions add ORDER BY and a LIMIT.
- Round money to whole dollars and rates to a sensible precision in the SQL when
  it makes the answer clearer.
- Follow-up questions refer to the earlier conversation — reuse the same tables
  and filters unless the user changes them.
- If the question cannot be answered from these tables, call cannot_answer with a
  short reason — do not guess.
"""

SCHEMA_BLOCK = """SCHEMA
======
{catalog}
"""

TOOLS = [
    {
        "name": "answer_with_sql",
        "description": "Provide the DuckDB SELECT query that answers the question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single DuckDB SELECT query."},
                "explanation": {"type": "string",
                                "description": "One sentence on what the query computes."},
            },
            "required": ["sql", "explanation"],
        },
    },
    {
        "name": "cannot_answer",
        "description": "Use when the question cannot be answered from the available tables.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


@dataclass
class Turn:
    """One completed exchange, replayed as context for follow-up questions."""
    question: str
    sql: str
    answer: str


@dataclass
class AskResult:
    question: str
    sql: str = ""
    explanation: str = ""
    answer: str = ""
    result: QueryResult = None
    refused: bool = False
    reason: str = ""
    attempts: int = 1
    corrections: list = field(default_factory=list)  # errors from failed attempts
    usage: dict = field(default_factory=dict)        # token spend, incl. cache reads
    # Non-blocking verifier findings about the query that DID run - an empty
    # result set, a rate outside its own range. These were previously computed
    # and dropped on the floor, which is the worst of both worlds: the cost of
    # checking without the benefit of saying anything.
    findings: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.refused and self.result is not None and self.result.ok

    def as_turn(self) -> Turn:
        return Turn(self.question, self.sql, self.answer)


class AssistantUnavailable(RuntimeError):
    """The language model could not be reached (missing/invalid key, no credits,
    network). Raised instead of leaking SDK stack traces into the apps; the
    original error text is preserved in str(exc)."""


def _format_result(res: QueryResult, max_rows: int = 30) -> str:
    if not res.columns:
        return "(no columns)"
    lines = [" | ".join(res.columns)]
    for row in res.rows[:max_rows]:
        lines.append(" | ".join("" if v is None else str(v) for v in row))
    if res.truncated or len(res.rows) > max_rows:
        lines.append(f"... ({res.row_count}{'+' if res.truncated else ''} rows)")
    return "\n".join(lines)


def _history_messages(history):
    """Render prior turns as plain alternating messages. Plain text (rather than
    replayed tool_use blocks) keeps the protocol simple: only the current turn
    uses the tool call."""
    messages = []
    for turn in history[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant",
                         "content": f"(SQL used)\n{turn.sql}\n\n(answer)\n{turn.answer}"})
    return messages


def _retrieval_context(question: str, history: list[Turn]) -> str:
    """Give vague follow-ups enough language to retrieve their prior domain."""
    parts = [question]
    for turn in history[-2:]:
        parts.extend((turn.question, turn.sql))
    return "\n".join(part for part in parts if part)


def _tables_used_by_history(history: list[Turn], known_tables: tuple[str, ...]) -> tuple[str, ...]:
    """Find exact warehouse table names in prior SQL, preserving catalog order."""
    sql = "\n".join(turn.sql.lower() for turn in history[-HISTORY_TURNS:] if turn.sql)
    return tuple(
        name
        for name in known_tables
        if re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", sql)
    )


class Assistant:
    """Stateless NL->SQL assistant. Pass `history` (a list of Turn) to enable
    follow-up questions; the caller owns and appends to it."""

    def __init__(self, con, client=None, model: str = MODEL, catalog_builder=None,
                 provider=None):
        self.con = con
        # The provider is the seam that makes the model swappable. An injected
        # client still forces Anthropic, because that is what the test harness
        # passes and asserts against; otherwise ASK_PROVIDER decides, defaulting
        # to Anthropic. Local servers (Ollama, llama.cpp, vLLM) arrive through
        # OpenAICompatProvider without this file knowing.
        self.client = client
        self.provider = (
            provider if provider is not None
            else AnthropicProvider(client=client, model=model) if client is not None
            else build_provider(model=model)
        )
        self.model = model
        self.catalog_builder = catalog_builder or schema_catalog_for
        self.known_tables = tuple(table_names(con))
        self.verifier = Verifier(con)

    def _system_for(self, question: str, history: list[Turn]) -> list[dict]:
        """Stable rules plus the retrieved schema for this turn."""
        context = _retrieval_context(question, history)
        required = _tables_used_by_history(history, self.known_tables)
        catalog = self.catalog_builder(
            context,
            self.con,
            include_tables=required,
        )
        blocks = [
            {"type": "text", "text": SYSTEM_RULES},
            {"type": "text", "text": SCHEMA_BLOCK.format(catalog=catalog)},
        ]

        # Few-shot exemplars: the k most similar SOLVED questions, drawn from the
        # same golden set CI asserts against. Leave-one-out lives inside
        # select_exemplars(), not here - a question must never be shown its own
        # reference SQL, or a live eval would be scoring memorisation.
        try:
            examples = exemplar_block(context, retrieved_tables=required)
        except Exception:
            examples = ""
        if examples:
            blocks.append({"type": "text", "text": examples})

        # Cache the whole prefix. Measured: worth nothing ACROSS turns - 0 of 39
        # follow-ups produced a byte-identical prefix, because the retrieved
        # schema changes with the question. Worth something WITHIN a turn: the
        # self-correction loop re-sends this identical prefix on attempts 2 and
        # 3, and at ~3,900 tokens that is the difference between paying full
        # price twice more and paying a tenth.
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        return blocks

    def ask(self, question: str, history: list = None) -> AskResult:
        history = history or []
        messages = _history_messages(history)
        messages.append({"role": "user", "content": question})
        system = self._system_for(question, history)

        usage, corrections = {}, []
        sql, explanation, result = "", "", None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self.provider.create_tool_call(
                    system=system, messages=messages,
                    tools=TOOLS, max_tokens=MAX_OUTPUT_TOKENS,
                )
            except ProviderUnavailable as e:
                raise AssistantUnavailable(str(e)) from e
            for key, value in (resp.usage or {}).items():
                usage[key] = usage.get(key, 0) + value
            tool_use = resp.tool_call
            if tool_use is None:
                return AskResult(question, refused=True, attempts=attempt,
                                 corrections=corrections, usage=usage,
                                 reason="model did not produce a query")
            if tool_use.name == "cannot_answer":
                return AskResult(question, refused=True, attempts=attempt,
                                 corrections=corrections, usage=usage,
                                 reason=tool_use.input.get("reason", "out of scope"))

            sql = tool_use.input["sql"]
            explanation = tool_use.input.get("explanation", "")

            # Structural checks BEFORE execution. The retry loop already handles
            # SQL that crashes; this handles SQL that RUNS AND IS WRONG, which
            # nothing downstream can see. The measured case: a join across two
            # of the eleven independent synthetic domains returns 5,400 rows -
            # no error, not empty, just meaningless.
            findings = self.verifier.check_sql(sql)
            blocking = [f for f in findings if f.blocking]
            if blocking:
                note = correction_message(blocking)
                if attempt < MAX_ATTEMPTS:
                    corrections.append(note)
                    messages.append({"role": "assistant", "content": f"I tried this SQL:\n{sql}"})
                    messages.append({"role": "user", "content": note})
                    continue
                # Out of attempts, and the query is still structurally
                # meaningless. Running it anyway would produce a number and a
                # confident sentence about it - the exact failure this layer
                # exists to prevent, and worse than an error because nothing
                # about the output would look wrong. Refuse instead.
                return AskResult(question, sql=sql, explanation=explanation,
                                 refused=True, reason=note, attempts=attempt,
                                 corrections=corrections, usage=usage,
                                 findings=findings)

            result = run_query(self.con, sql)
            if result.ok:
                # Post-execution checks: shapes that only the returned rows can
                # reveal (an empty set, a rate outside its own range). These are
                # advisory - they travel with the answer rather than blocking it.
                findings = findings + self.verifier.check_result(sql, result, question)
                answer = self._summarize(question, result, usage)
                return AskResult(question, sql=sql, explanation=explanation,
                                 answer=answer, result=result, usage=usage,
                                 attempts=attempt, corrections=corrections,
                                 findings=findings)

            # Self-correction: hand the real error back and ask for a fix.
            corrections.append(result.error)
            messages.append({"role": "assistant", "content": f"I tried this SQL:\n{sql}"})
            messages.append({"role": "user", "content": (
                f"That query failed with: {result.error}\n"
                "Write a corrected single SELECT query. Use only tables and "
                "columns that appear in the schema."
            )})

        # Out of attempts — return the last failure honestly.
        return AskResult(question, sql=sql, explanation=explanation, result=result,
                         attempts=MAX_ATTEMPTS, corrections=corrections, usage=usage)

    def _summarize(self, question: str, result: QueryResult, usage: dict) -> str:
        """Turn the result table into one or two plain-English sentences, grounded
        strictly in the returned rows."""
        try:
            resp = self.provider.complete(
                max_tokens=400,
                system=("Answer the user's question in one or two sentences using ONLY "
                        "the SQL result provided. Never invent or round beyond what is "
                        "shown. If there are no rows, say nothing matched."),
                messages=[{
                    "role": "user",
                    "content": f"Question: {question}\n\nSQL result:\n{_format_result(result)}",
                }],
            )
        except ProviderUnavailable as e:
            raise AssistantUnavailable(str(e)) from e
        for key, value in (resp.usage or {}).items():
            usage[key] = usage.get(key, 0) + value
        return (resp.text or "").strip()
