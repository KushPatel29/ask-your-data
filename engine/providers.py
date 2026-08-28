"""
Model-provider abstraction: make the thing that writes SQL swappable.

WHY THIS FILE EXISTS
`Assistant._create()` calls `self.client.messages.create(...)` and reads an
Anthropic `tool_use` block out of the response. That is one vendor's wire format
sitting in the middle of the control flow, and it is the only reason this app
needs Anthropic specifically. Everything else — retrieval, the exemplar bank,
the read-only guard, the verifier, the executor — is vendor-neutral and would
work behind any model that can emit a SQL string.

So the abstraction is deliberately narrow. A provider does exactly one thing:

    given (system blocks, messages, tool schemas) -> return ONE ToolCall

That is the whole contract. It is not a general chat abstraction, because the
assistant does not need one.

THE HARD PART IS NOT THE HTTP CALL
Swapping the transport is twenty lines. The difficulty is that this app needs a
well-formed, schema-valid tool call on EVERY turn, and constrained tool use is
precisely where small open-weights models are weakest. Frontier models are
RL-trained on tool-call formatting; a 7B instruct model is mostly not, and it
fails in ways that are individually easy to describe and collectively fatal to
an unguarded parser:

  * wraps the JSON in ```json fences, or in prose ("Sure! Here's the query:")
  * emits the SCHEMA instead of an instance ({"type":"object","properties":...})
  * emits the arguments without the tool envelope ({"sql": ...} with no "name")
  * single-quotes keys, trailing commas, or Python None/True instead of JSON
  * puts a raw multi-line SQL string in a JSON field without escaping newlines
  * emits two tool calls, or a tool call plus a chatty epilogue
  * silently answers in prose instead of calling a tool at all

`parse_tool_call()` below handles the first six. The seventh is not a parsing
problem and must not be papered over — it is returned as a refusal, exactly as
`Assistant.ask()` already treats a missing tool_use block (assistant.py:281-285).

THREE WAYS A NON-ANTHROPIC MODEL CAN HONOUR THE CONTRACT, best first:

  1. GRAMMAR-CONSTRAINED DECODING. llama.cpp (`grammar` / GBNF), vLLM
     (outlines / xgrammar), and Ollama (`format` = a JSON schema) can restrict
     the sampler so that only tokens continuing a schema-valid document are
     ever emitted. This makes malformed JSON structurally impossible rather
     than unlikely. It does NOT make the SQL correct — it guarantees shape, not
     semantics, and conflating those two is the standard mistake here.
  2. NATIVE TOOL CALLING over an OpenAI-compatible /v1/chat/completions
     endpoint. Available on newer Ollama and vLLM builds and on hosted
     open-weights APIs. Quality varies by model far more than by server.
  3. PROMPT-AND-PARSE with a bounded repair retry. The universal fallback. Works
     against any endpoint, and is the only option for a plain
     /v1/completions server.

All three land in the same `ToolCall`, so `Assistant` never learns which one ran.

WHAT IS MEASURED HERE AND WHAT IS NOT
Measured, on this box, today: `python -m engine.providers` runs
`_PARSER_CASES` — 21 recorded malformed-output shapes — through
`parse_tool_call()` and reports how many are recovered. That is a real number
about THIS parser.

NOT measured, and not claimed anywhere in this file: how often any particular
local model produces correct SQL for this warehouse. There is no API key and no
local model in this environment, so no end-to-end accuracy number in this repo
may be attributed to a local provider until `scripts/run_live_eval.py` has
actually been run against one. See the module docstring of that script for the
harness; `evals/golden_questions.yaml` is the metric.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """One structured call, normalised across providers.

    Mirrors the two fields `Assistant.ask()` reads off an Anthropic `tool_use`
    block — `.name` and `.input` — so the assistant's branching is unchanged.
    """

    name: str
    input: dict


@dataclass
class ProviderResponse:
    tool_call: ToolCall | None
    usage: dict = field(default_factory=dict)
    text: str = ""          # visible text, kept for diagnostics
    repairs: int = 0        # how many reparse/retry rounds this answer cost


class ProviderUnavailable(RuntimeError):
    """Transport, auth, or configuration failure. Mapped by the assistant onto
    the existing `AssistantUnavailable` so the UI story does not change."""


class Provider(Protocol):
    def create_tool_call(
        self,
        *,
        system: list[dict],
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
    ) -> ProviderResponse: ...

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> ProviderResponse: ...

    def health(self) -> dict: ...


# --------------------------------------------------------------------------
# Parsing: the layer that decides whether a local model is usable at all
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_SCHEMA_KEYS = {"type", "properties", "required", "input_schema", "$schema"}


def _strip_fences(text: str) -> str:
    matches = _FENCE.findall(text or "")
    return matches[0].strip() if matches else (text or "").strip()


def _balanced_objects(text: str) -> list[str]:
    """Every top-level {...} run in `text`, brace-counted OUTSIDE string
    literals. A regex cannot do this: SQL routinely contains braces and quotes,
    and `json.loads` on the whole blob fails the moment a model adds a preamble.
    """
    out, depth, start, in_str, quote, esc = [], 0, -1, False, "", False
    for i, ch in enumerate(text or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:i + 1])
                start = -1
            elif depth < 0:
                depth = 0
    return out


def _relax(blob: str) -> str:
    """Coax near-JSON into JSON. Each substitution corresponds to a failure
    shape small models actually produce; none of them touch the inside of a
    correctly-quoted string."""
    blob = re.sub(r",\s*([}\]])", r"\1", blob)            # trailing commas
    blob = re.sub(r"\bNone\b", "null", blob)              # Python literals
    blob = re.sub(r"\bTrue\b", "true", blob)
    blob = re.sub(r"\bFalse\b", "false", blob)
    return blob


def _escape_raw_newlines_in_strings(blob: str) -> str:
    """Fix the single most common local-model JSON break: a multi-line SQL
    string pasted into a JSON field with literal newlines left unescaped.

    Walks the blob character by character and escapes newlines/tabs that occur
    while inside a double-quoted string. Anything outside a string is untouched,
    so formatting whitespace between fields survives.
    """
    out, in_str, esc = [], False, False
    for ch in blob:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            elif ch == "\n":
                out.append("\\n")
                continue
            elif ch == "\r":
                out.append("\\r")
                continue
            elif ch == "\t":
                out.append("\\t")
                continue
        elif ch == '"':
            in_str = True
        out.append(ch)
    return "".join(out)


def _loads(blob: str) -> Any:
    for candidate in (blob, _relax(blob), _escape_raw_newlines_in_strings(_relax(blob))):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _looks_like_schema(obj: dict) -> bool:
    """The model echoed the tool DEFINITION back instead of calling it.

    Worth detecting explicitly: such an object often has a "name" key and would
    otherwise parse into a ToolCall with an empty input, producing an empty SQL
    string and a confusing downstream error instead of a clean refusal.
    """
    if not isinstance(obj, dict):
        return False
    if _SCHEMA_KEYS & set(obj):
        inner = obj.get("properties") or obj.get("input_schema")
        if isinstance(inner, dict) or obj.get("type") == "object":
            return True
    return False


def parse_tool_call(text: str, tools: list[dict]) -> ToolCall | None:
    """Recover a ToolCall from free-form model output, or return None.

    None means "no usable tool call", which the assistant already treats as a
    refusal. Returning a half-populated ToolCall would be worse than returning
    nothing: it turns a visible failure into a silent one, which is the failure
    class this whole repo is built to avoid.
    """
    if not text or not text.strip():
        return None

    names = [t["name"] for t in tools]
    required = {t["name"]: set(t["input_schema"].get("required", [])) for t in tools}

    candidates: list[Any] = []
    stripped = _strip_fences(text)
    for blob in ([stripped] if stripped else []) + _balanced_objects(text):
        parsed = _loads(blob)
        if parsed is not None:
            candidates.append(parsed)

    for obj in candidates:
        if not isinstance(obj, dict) or _looks_like_schema(obj):
            continue

        # Shape A: a full envelope, in any of the spellings servers use.
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        if isinstance(name, dict):                      # OpenAI: function:{name,arguments}
            name = name.get("name")
        args = obj.get("input") or obj.get("arguments") or obj.get("parameters")
        if isinstance(args, str):                       # OpenAI sends a JSON *string*
            args = _loads(args)
        if name in names and isinstance(args, dict):
            if required[name] <= set(args):
                return ToolCall(str(name), args)

        # Shape B: the arguments alone, no envelope. Match on required keys.
        for candidate_name in names:
            if required[candidate_name] and required[candidate_name] <= set(obj):
                return ToolCall(candidate_name, obj)

        # Shape C: envelope naming a tool, arguments inlined as siblings.
        if name in names:
            inline = {k: v for k, v in obj.items()
                      if k not in {"name", "tool", "function", "type"}}
            if required[name] <= set(inline):
                return ToolCall(str(name), inline)

    return None


# --------------------------------------------------------------------------
# Anthropic: the current behaviour, unchanged, behind the new interface
# --------------------------------------------------------------------------

class AnthropicProvider:
    """Wraps the existing SDK call. Native tool use, so no parsing is needed:
    `tool_choice={"type":"any"}` already guarantees a tool_use block, and the
    thinking-on decision documented at assistant.py:40-49 keeps it that way."""

    def __init__(self, client=None, model: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model or os.environ.get("ASK_YOUR_DATA_MODEL", "claude-opus-5")

    def create_tool_call(self, *, system, messages, tools, max_tokens) -> ProviderResponse:
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                tool_choice={"type": "any"},
                messages=messages,
            )
        except self._anthropic.APIError as e:
            raise ProviderUnavailable(str(e)) from e
        except TypeError as e:  # SDK "could not resolve authentication"
            raise ProviderUnavailable(str(e)) from e

        usage = {}
        u = getattr(msg, "usage", None)
        for key in ("input_tokens", "output_tokens",
                    "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = getattr(u, key, None) if u else None
            if value:
                usage[key] = value

        block = next((b for b in msg.content if b.type == "tool_use"), None)
        text = "".join(getattr(b, "text", "") for b in msg.content if b.type == "text")
        call = ToolCall(block.name, dict(block.input)) if block else None
        return ProviderResponse(tool_call=call, usage=usage, text=text)

    def complete(self, *, system, messages, max_tokens) -> ProviderResponse:
        """Plain text, no tools - the summarize call.

        Separate from create_tool_call because that one forces
        tool_choice={"type": "any"}, and forcing a tool call to obtain a
        sentence is the wrong contract: it would make the summary the argument
        of a function the model has no reason to call.
        """
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=system, messages=messages,
            )
        except self._anthropic.APIError as e:
            raise ProviderUnavailable(str(e)) from e
        except TypeError as e:
            raise ProviderUnavailable(str(e)) from e

        usage = {}
        u = getattr(msg, "usage", None)
        for key in ("input_tokens", "output_tokens",
                    "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = getattr(u, key, None) if u else None
            if value:
                usage[key] = value
        text = "".join(getattr(b, "text", "") for b in msg.content if b.type == "text")
        return ProviderResponse(tool_call=None, usage=usage, text=text)

    def health(self) -> dict:
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY")
                       or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        return {"provider": "anthropic", "model": self.model,
                "status": "ok" if has_key else "no_api_key"}


# --------------------------------------------------------------------------
# OpenAI-compatible local servers: Ollama, llama.cpp --server, vLLM, LM Studio
# --------------------------------------------------------------------------

def _json_schema_for(tools: list[dict]) -> dict:
    """One schema covering every tool, for grammar-constrained decoding.

    A `oneOf` is deliberately avoided: several constrained-decoding backends
    handle it poorly or ignore it. A flat object with an enum'd `name` and a
    permissive `input` compiles reliably everywhere and still forces the two
    things that matter — valid JSON, and a name drawn from the real tool list.
    """
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": [t["name"] for t in tools]},
            "input": {"type": "object"},
        },
        "required": ["name", "input"],
    }


_CONTRACT = """You MUST reply with a single JSON object and nothing else. No prose, \
no markdown fences, no explanation before or after.

The object has exactly two keys:
  "name"  - one of: {names}
  "input" - the arguments object for that tool

Tool schemas:
{schemas}

Newlines inside the SQL string must be escaped as \\n. Reply with the JSON object only."""


class OpenAICompatProvider:
    """Any server speaking POST {base_url}/v1/chat/completions.

    Verified shapes this targets (none exercised here - no local server in this
    environment): Ollama >=0.4 (`/v1`), llama.cpp `--server`, vLLM
    `--api-server`, LM Studio. `mode` selects how the contract is enforced:

      "auto"     try native tools, fall back to schema, fall back to prompt
      "tools"    native OpenAI tool calling only
      "schema"   response_format json_schema - grammar-constrained where supported
      "prompt"   contract in the system prompt, parsed out, with a repair retry

    Uses urllib from the stdlib on purpose: this file must not add a dependency
    to requirements.txt for a code path most users will never enable.
    """

    def __init__(self, *, base_url: str | None = None, model: str | None = None,
                 mode: str = "auto", timeout: int = 120, max_repairs: int = 1,
                 temperature: float = 0.0, api_key: str | None = None):
        self.base_url = (base_url or os.environ.get("ASK_LOCAL_BASE_URL")
                         or "http://localhost:11434").rstrip("/")
        self.model = model or os.environ.get("ASK_LOCAL_MODEL", "qwen2.5-coder:7b")
        self.mode = (os.environ.get("ASK_LOCAL_MODE") or mode).lower()
        self.timeout = timeout
        self.max_repairs = max_repairs
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("ASK_LOCAL_API_KEY", "")

    # -- transport ---------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            raise ProviderUnavailable(f"{url} returned {e.code}: {body}") from e
        except Exception as e:
            raise ProviderUnavailable(f"{url} unreachable: {e}") from e

    # -- prompt shaping ----------------------------------------------------

    @staticmethod
    def _flatten_system(system: list[dict]) -> str:
        """Anthropic sends system as a list of blocks (assistant.py:228-251).
        OpenAI-compatible servers take one string. `cache_control` is dropped:
        no local server implements prompt caching, and pretending otherwise
        would put a meaningless key on the wire."""
        return "\n\n".join(b.get("text", "") for b in system if b.get("text"))

    def _messages(self, system: list[dict], messages: list[dict],
                  tools: list[dict], contract: bool) -> list[dict]:
        head = self._flatten_system(system)
        if contract:
            head += "\n\n" + _CONTRACT.format(
                names=", ".join(t["name"] for t in tools),
                schemas=json.dumps(
                    [{"name": t["name"], "input_schema": t["input_schema"]} for t in tools],
                    indent=2),
            )
        out = [{"role": "system", "content": head}]
        for m in messages:
            content = m["content"]
            out.append({"role": m["role"],
                        "content": content if isinstance(content, str) else str(content)})
        return out

    @staticmethod
    def _usage(body: dict) -> dict:
        u = body.get("usage") or {}
        return {k: v for k, v in (
            ("input_tokens", u.get("prompt_tokens")),
            ("output_tokens", u.get("completion_tokens")),
        ) if v}

    @staticmethod
    def _text_and_calls(body: dict) -> tuple[str, list[dict]]:
        choices = body.get("choices") or []
        if not choices:
            return "", []
        msg = (choices[0] or {}).get("message") or {}
        return str(msg.get("content") or ""), (msg.get("tool_calls") or [])

    # -- the contract ------------------------------------------------------

    def create_tool_call(self, *, system, messages, tools, max_tokens) -> ProviderResponse:
        modes = ({"tools": ["tools"], "schema": ["schema"], "prompt": ["prompt"]}
                 .get(self.mode, ["tools", "schema", "prompt"]))
        last_text, usage, repairs = "", {}, 0

        for mode in modes:
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": self._messages(system, messages, tools,
                                           contract=(mode == "prompt")),
            }
            if mode == "tools":
                payload["tools"] = [
                    {"type": "function",
                     "function": {"name": t["name"], "description": t.get("description", ""),
                                  "parameters": t["input_schema"]}}
                    for t in tools
                ]
                payload["tool_choice"] = "required"
            elif mode == "schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "tool_call", "strict": True,
                                    "schema": _json_schema_for(tools)},
                }

            try:
                body = self._post("/v1/chat/completions", payload)
            except ProviderUnavailable:
                if mode == modes[-1]:
                    raise
                continue

            usage = self._usage(body) or usage
            text, calls = self._text_and_calls(body)
            last_text = text or last_text

            if calls:
                fn = (calls[0] or {}).get("function") or {}
                args = fn.get("arguments")
                parsed = _loads(args) if isinstance(args, str) else args
                if fn.get("name") in {t["name"] for t in tools} and isinstance(parsed, dict):
                    return ProviderResponse(ToolCall(fn["name"], parsed), usage, text, repairs)

            call = parse_tool_call(text, tools)
            if call is not None:
                return ProviderResponse(call, usage, text, repairs)

            # Bounded repair: show the model its own bad output once. Bounded
            # for the same reason MAX_ATTEMPTS is (assistant.py:56) - a model
            # that cannot produce the shape twice will not produce it on the
            # tenth try, and an unbounded loop on a 40-token/s local model is a
            # hang, not a retry.
            if mode == "prompt" and repairs < self.max_repairs:
                repairs += 1
                retry = list(messages) + [
                    {"role": "assistant", "content": text[:2000]},
                    {"role": "user", "content":
                        "That was not a single JSON object matching the contract. "
                        "Reply with ONLY the JSON object - no prose, no fences."},
                ]
                payload["messages"] = self._messages(system, retry, tools, contract=True)
                try:
                    body = self._post("/v1/chat/completions", payload)
                except ProviderUnavailable:
                    break
                text, _ = self._text_and_calls(body)
                last_text = text or last_text
                call = parse_tool_call(text, tools)
                if call is not None:
                    return ProviderResponse(call, usage, text, repairs)

        # No usable tool call. Assistant.ask() turns this into an honest refusal
        # rather than executing anything.
        return ProviderResponse(None, usage, last_text, repairs)

    def complete(self, *, system, messages, max_tokens) -> ProviderResponse:
        """Plain text from an OpenAI-compatible server. No tools, no contract
        preamble - nothing here needs to be parsed back into a structure."""
        head = system if isinstance(system, str) else self._flatten_system(system)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": head}] + [
                {"role": m["role"],
                 "content": m["content"] if isinstance(m["content"], str) else str(m["content"])}
                for m in messages
            ],
        }
        body = self._post("/chat/completions", payload)
        text, _calls = self._text_and_calls(body)
        return ProviderResponse(tool_call=None, usage=self._usage(body), text=text or "")

    def health(self) -> dict:
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models", method="GET")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=min(10, self.timeout)) as resp:
                body = json.loads(resp.read().decode("utf-8") or "{}")
            served = [m.get("id") for m in (body.get("data") or []) if isinstance(m, dict)]
            return {"provider": "openai_compat", "status": "ok",
                    "base_url": self.base_url, "model": self.model,
                    "model_served": self.model in served, "models_visible": len(served)}
        except Exception as e:
            return {"provider": "openai_compat", "status": "error",
                    "base_url": self.base_url, "error": str(e)}


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def build_provider(name: str | None = None, **kwargs) -> Provider:
    """Pick a provider. Default stays Anthropic: nothing about existing
    behaviour changes unless ASK_PROVIDER is set deliberately."""
    token = (name or os.environ.get("ASK_PROVIDER") or "anthropic").strip().lower()
    if token in {"local", "ollama", "llamacpp", "llama_cpp", "vllm", "openai_compat"}:
        return OpenAICompatProvider(**kwargs)
    return AnthropicProvider(**kwargs)


# --------------------------------------------------------------------------
# Parser robustness harness - the one thing here that CAN be measured offline
# --------------------------------------------------------------------------

_TOOLS = [
    {"name": "answer_with_sql", "description": "",
     "input_schema": {"type": "object",
                      "properties": {"sql": {"type": "string"},
                                     "explanation": {"type": "string"}},
                      "required": ["sql", "explanation"]}},
    {"name": "cannot_answer", "description": "",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}},
                      "required": ["reason"]}},
]

_SQL = "SELECT COUNT(*) FROM hr_fact_employees WHERE is_active = 1"

# (label, raw model output, expected tool name or None)
# Every entry is a shape reported for small instruct models on structured-output
# tasks. The None cases matter as much as the others: a parser that "recovers"
# something from prose or from an echoed schema is worse than one that gives up,
# because the assistant would then execute a query nobody wrote.
_PARSER_CASES = [
    ("clean envelope",
     json.dumps({"name": "answer_with_sql", "input": {"sql": _SQL, "explanation": "x"}}),
     "answer_with_sql"),
    ("json fence",
     '```json\n{"name":"answer_with_sql","input":{"sql":"%s","explanation":"x"}}\n```' % _SQL,
     "answer_with_sql"),
    ("bare fence",
     '```\n{"name":"answer_with_sql","input":{"sql":"%s","explanation":"x"}}\n```' % _SQL,
     "answer_with_sql"),
    ("prose preamble",
     'Sure! Here is the query:\n{"name":"answer_with_sql","input":{"sql":"%s","explanation":"x"}}' % _SQL,
     "answer_with_sql"),
    ("prose epilogue",
     '{"name":"answer_with_sql","input":{"sql":"%s","explanation":"x"}}\nHope that helps!' % _SQL,
     "answer_with_sql"),
    ("openai function shape",
     json.dumps({"function": {"name": "answer_with_sql"},
                 "arguments": json.dumps({"sql": _SQL, "explanation": "x"})}),
     "answer_with_sql"),
    ("arguments as json string",
     json.dumps({"name": "answer_with_sql",
                 "arguments": json.dumps({"sql": _SQL, "explanation": "x"})}),
     "answer_with_sql"),
    ("parameters key",
     json.dumps({"name": "answer_with_sql",
                 "parameters": {"sql": _SQL, "explanation": "x"}}),
     "answer_with_sql"),
    ("no envelope, args only",
     json.dumps({"sql": _SQL, "explanation": "x"}),
     "answer_with_sql"),
    ("inline siblings",
     json.dumps({"name": "answer_with_sql", "sql": _SQL, "explanation": "x"}),
     "answer_with_sql"),
    ("trailing comma",
     '{"name":"answer_with_sql","input":{"sql":"%s","explanation":"x",},}' % _SQL,
     "answer_with_sql"),
    ("python literals",
     '{"name":"answer_with_sql","input":{"sql":"%s","explanation":"x","note":None}}' % _SQL,
     "answer_with_sql"),
    ("unescaped newlines in sql",
     '{"name":"answer_with_sql","input":{"sql":"SELECT COUNT(*)\nFROM hr_fact_employees\nWHERE is_active = 1","explanation":"x"}}',
     "answer_with_sql"),
    ("tabs in sql",
     '{"name":"answer_with_sql","input":{"sql":"SELECT\t1","explanation":"x"}}',
     "answer_with_sql"),
    ("braces inside sql string",
     '{"name":"answer_with_sql","input":{"sql":"SELECT {1}","explanation":"x"}}',
     "answer_with_sql"),
    ("apostrophe literal in sql",
     json.dumps({"name": "answer_with_sql",
                 "input": {"sql": "SELECT * FROM t WHERE s = 'Denied'", "explanation": "x"}}),
     "answer_with_sql"),
    ("refusal tool",
     json.dumps({"name": "cannot_answer", "input": {"reason": "no such table"}}),
     "cannot_answer"),
    ("two objects, first usable",
     '{"name":"answer_with_sql","input":{"sql":"%s","explanation":"x"}}\n{"name":"cannot_answer","input":{"reason":"y"}}' % _SQL,
     "answer_with_sql"),
    # must NOT parse
    ("prose only", "The answer is 1,483 active employees.", None),
    ("echoed schema", json.dumps(_TOOLS[0]["input_schema"]), None),
    ("empty", "", None),
]


def _run_parser_harness() -> int:
    ok = 0
    print(f"parse_tool_call() against {len(_PARSER_CASES)} recorded "
          f"malformed-output shapes\n")
    for label, raw, expected in _PARSER_CASES:
        got = parse_tool_call(raw, _TOOLS)
        name = got.name if got else None
        good = name == expected
        ok += good
        detail = ""
        if good and got and expected == "answer_with_sql":
            sql = str(got.input.get("sql", ""))
            detail = f"  sql={sql[:38]!r}"
        print(f"  [{'ok ' if good else 'FAIL'}] {label:<28} -> {str(name):<18}{detail}")
    print(f"\n{ok}/{len(_PARSER_CASES)} recovered correctly")
    return ok


if __name__ == "__main__":  # pragma: no cover - manual/CI smoke
    _run_parser_harness()
