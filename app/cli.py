"""
Ask questions from the terminal.

    python -m app.cli "which payer type collects the least of what it's billed?"
    python -m app.cli                    # interactive REPL
    python -m app.cli --plan "..."       # force the keyless compiler
    python -m app.cli --model "..."      # force the model (needs a key)

With ANTHROPIC_API_KEY set, a language model writes the SQL. Without one, the
deterministic compiler in `engine/planner.py` writes it instead -- no key, no
network, no cost, and a refusal rather than a guess when it cannot bind the
question. The warehouse, the guard and the executor are the same either way, and
the header on every answer says which engine ran.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.warehouse import build_warehouse  # noqa: E402


def _usage_line(usage):
    if not usage:
        return ""
    parts = [f"in {usage.get('input_tokens', 0):,}", f"out {usage.get('output_tokens', 0):,}"]
    if usage.get("cache_read_input_tokens"):
        parts.append(f"cache read {usage['cache_read_input_tokens']:,}")
    return "tokens: " + " / ".join(parts)


def _rows(result):
    if result.ok:
        print("\n  " + " | ".join(result.columns))
        for row in result.rows[:15]:
            print("  " + " | ".join("" if v is None else str(v) for v in row))
        if result.truncated:
            print(f"  ... (showing first {len(result.rows)} rows)")
    else:
        print(f"\n  query error: {result.error}")


def show(result):
    """A model-authored turn."""
    if result.refused:
        print(f"\n  I can't answer that from the loaded data: {result.reason}\n")
        return
    print(f"\n{result.answer}\n")
    if result.attempts > 1:
        print(f"  (self-corrected after {result.attempts} attempts)")
    print("  SQL, written by the model:")
    for line in result.sql.strip().splitlines():
        print(f"    {line}")
    _rows(result.result)
    if result.usage:
        print(f"\n  {_usage_line(result.usage)}")
    print()


def show_plan(result, ran):
    """A compiled turn, with the binding trace the model path cannot produce.

    Printed for the same reason the app draws it: a deterministic planner can
    name the evidence for every clause it wrote, so not printing it would be
    throwing away the one thing this engine has that the other does not.
    """
    if result.refused:
        print(f"\n  I can't compile that one: {result.reason}\n")
        if result.plan:
            print("  How far it got:")
            for key, value in result.plan.rationale():
                print(f"    {key:>12}  {value}")
        if result.unbound:
            print(f"    {'no vocabulary':>12}  {', '.join(sorted(result.unbound))}")
        print()
        return
    print(f"\n  compiled without a model — {result.plan.coverage:.0%} of the question "
          f"bound, {result.considered} candidate tables considered\n")
    for key, value in result.plan.rationale():
        print(f"    {key:>12}  {value}")
    print("\n  SQL, compiled from the schema:")
    for line in result.sql.strip().splitlines():
        print(f"    {line}")
    _rows(ran)
    if result.unbound:
        print(f"\n  words this warehouse has no vocabulary for: "
              f"{', '.join(sorted(result.unbound))}")
    print()


def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _planner_runner(con, access_scope):
    """Build the compiler path. Imports nothing that needs a key or a network."""
    from engine import planner
    from engine.deadline import RequestDeadline
    from engine.query import run_query
    from engine.semantics import Layer

    layer = Layer(con)
    try:
        from engine import retrieval

        retrieval.build_index(con)

        def hits(question):
            return [h.table for h in retrieval.retrieve_hybrid(question)]
    except Exception:
        # Retrieval is a ranking signal, never a requirement. A machine with no
        # embedding model still gets a working compiler.
        def hits(_question):
            return []

    def ask(question, history=None):
        request_deadline = RequestDeadline.configured()
        result = planner.plan_question(question, layer, retrieved=hits(question))
        ran = (run_query(con, result.sql, access=access_scope, deadline=request_deadline)
               if result.ok else None)
        show_plan(result, ran)
        _record(question, result=result, ran=ran, engine="plan",
                actor=access_scope.principal.subject,
                role=",".join(sorted(access_scope.principal.roles)) or "none")
        return result

    return ask


def _model_runner(con, access_scope):
    from engine.assistant import Assistant, AssistantUnavailable

    assistant = Assistant(con, access=access_scope)

    def ask(question, history=None):
        try:
            result = assistant.ask(question, history=history)
        except AssistantUnavailable as e:
            print(f"\n  The language model is unavailable: {e}\n"
                  "  Set ANTHROPIC_API_KEY (and check your credit balance), or run\n"
                  "  with --plan to use the keyless compiler instead.\n")
            return None
        show(result)
        _record(question, result=result, engine="model",
                actor=access_scope.principal.subject,
                role=",".join(sorted(access_scope.principal.roles)) or "none")
        return result

    return ask


def _record(question, *, result, engine, ran=None, actor="cli", role="demo"):
    """The terminal is an entry point too.

    An audit trail that covers the web app and not the CLI has a hole in it
    exactly where a developer does their most unusual queries. With no
    ASK_YOUR_DATA_AUDIT path set this writes to a ring that dies with the
    process, which is the right amount of ceremony for a one-shot command —
    the point is that the code path exists and is the same one.
    """
    from engine import audit

    refused = getattr(result, "refused", False) or not getattr(result, "ok", False)
    timed_out = bool(
        getattr(result, "timed_out", False) or getattr(ran, "timed_out", False)
    )
    timeout_stage = (
        getattr(result, "timeout_stage", "")
        or getattr(ran, "timeout_stage", "")
    )
    try:
        audit.record(
            actor=actor, role=role, engine=engine, question=question,
            sql=getattr(result, "sql", "") or "",
            outcome=("timeout" if timed_out else "refused" if refused else
                     ("error" if ran is not None and not ran.ok else "answered")),
            refusal_kind=getattr(result, "kind", "") or "",
            reason=getattr(result, "reason", "") or "",
            row_count=0 if ran is None else ran.row_count,
            truncated=bool(getattr(ran, "truncated", False)),
            attempts=int(getattr(result, "attempts", 1) or 1),
            coverage=getattr(getattr(result, "plan", None), "coverage", None),
            elapsed_ms=float(getattr(result, "elapsed_ms", 0.0) or 0.0),
            timeout_stage=timeout_stage,
        )
    except Exception:  # noqa: BLE001 - an observer must never break the observed
        pass


def main():
    argv = sys.argv[1:]
    force_plan = "--plan" in argv
    force_model = "--model" in argv
    argv = [a for a in argv if a not in ("--plan", "--model")]

    if force_model and not _has_key():
        print("  --model needs ANTHROPIC_API_KEY. Drop the flag to use the compiler.")
        return 2

    from engine import access

    con = build_warehouse()
    try:
        if access.auth_mode() == access.AUTH_OIDC:
            token = os.environ.get("ASK_ACCESS_TOKEN", "").strip()
            principal = access.authenticate_headers({"Authorization": f"Bearer {token}"})
        else:
            principal = access.Principal.demo("cli")
        query_access = access.access_scope(principal)
        access.validate_scope_schema(con, query_access)
    except (access.AuthenticationError, access.PolicyConfigurationError) as exc:
        print(f"  Access denied: {exc}")
        return 2
    use_model = force_model or (_has_key() and not force_plan)
    ask = (_model_runner(con, query_access) if use_model
           else _planner_runner(con, query_access))
    engine_name = "the model" if use_model else "the keyless compiler"

    if argv:
        ask(" ".join(argv))
        return 0

    print(f"Ask your data — answering with {engine_name}. Blank line to quit.\n")
    history = []
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        result = ask(question, history=history)
        # Only the model path carries conversation state. The compiler is
        # stateless by construction -- it binds words to columns and has nothing
        # to remember -- and pretending otherwise would put a follow-up in a
        # transcript that never influenced anything.
        if use_model and result is not None and result.ok:
            history.append(result.as_turn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
