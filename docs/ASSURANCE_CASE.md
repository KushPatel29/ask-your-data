# Ask Your Data — Release Assurance and Business Case

**Release:** 2026.09.15

**Decision owner:** Analytics Engineering

**Decision:** GO for a synthetic portfolio demonstration

**Production boundary:** A production rollout remains conditional on organizational identity,
database-enforced row policies, durable audit storage, capacity/recovery testing, and an approved
model evaluation on the target provider.

## Executive decision

Analysts lose time translating ordinary business questions into extracts, while dashboard-only
delivery leaves valid follow-up questions unanswered. Ask Your Data demonstrates a governed route
from natural language to an inspectable database result. It supports a deterministic, keyless
compiler for bounded questions and an optional model path for broader language. Both routes use the
same authorization, structural verification, read-only SQL guard, deadline, and bounded executor.

The release is suitable for demonstration because every displayed number is tied to executed SQL,
39 committed reference answers are reproduced in CI, the keyless engine records zero wrong answers
on its 58-question contract, required-table retrieval is gated at 100% on both golden questions and
development paraphrases, and five runtime budgets agree with the code that enforces them. The model
red-team gate remains open by design: it must be rerun with credentials whenever the provider,
model, prompt, tools, or authorization context changes.

## Stakeholders and decision rights

| Stakeholder | Need | Decision right / accountability |
|---|---|---|
| Business analyst | Fast, defensible answers and clear refusal reasons | Owns question catalogue, definitions, UAT, and adoption feedback |
| Data owner | Correct metric meaning and appropriate access | Approves certified definitions, table grants, and sensitive-column exceptions |
| Security / privacy | No mutation, exfiltration, or cross-role disclosure | Approves identity, database policy, audit retention, and threat acceptance |
| Data platform | Stable, bounded workload | Owns warehouse credentials, budgets, monitoring, capacity, and recovery |
| Product owner | Valuable scope and usable workflow | Accepts release gates and prioritizes unsupported questions |
| Analytics engineering | Reproducible implementation | Builds and operates retrieval, planning, verification, tests, and release evidence |

## Current state and future state

| Current state | Demonstrated future state |
|---|---|
| Questions wait for an analyst or are constrained to predefined dashboards | Users can ask bounded questions and inspect the generated SQL and result |
| Definitions are separated from the answer workflow | Certified metrics carry an owner, definition, SQL, expected result, and CI check |
| Access intent may live only in team knowledge | A default-deny role policy filters catalog metadata and query execution |
| Evaluation results are scattered across scripts and documentation | One versioned release pack binds suites, thresholds, budgets, and evidence paths |
| A green build can be mistaken for proof of model behavior | Offline evidence and credentialed live-model evidence are shown as different gates |
| Result limits alone fail to control expensive queries | Row, statement-time, request-time, memory, thread, and input limits work together |

## Options considered

| Option | Benefit | Material weakness | Decision |
|---|---|---|---|
| Dashboard only | Familiar and predictable | Cannot handle unanticipated follow-up questions | Retain for recurring KPIs, not as the only interface |
| Unrestricted model-to-SQL | Broad language coverage | Treats generated code as trusted and creates a large security surface | Rejected |
| Model with prompt-only safety | Quick to prototype | Instructions are not an authorization or execution boundary | Rejected |
| Governed dual engine | Keyless coverage plus optional model flexibility; one controlled executor | More engineering and explicit refusal behavior | Selected for demonstration |

## Traceable requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| BR-01 | Return no business number without query provenance | Every successful answer exposes SQL, returned rows, and execution metadata |
| BR-02 | Support a useful path without a paid model key | Deterministic compiler runs against the 71-table semantic layer |
| BR-03 | Refuse rather than invent when meaning is not bound | Planner contract separates correct, wrong, refused, and error outcomes |
| BR-04 | Preserve governed business definitions | Six certified metrics include owner, definition, SQL, and expected result |
| SEC-01 | Admit only one read-only statement | SQL parser and forbidden verb/function contracts run before execution |
| SEC-02 | Enforce role and column scope at the executor | OIDC principal and default-deny policy are passed to every app execution path |
| SEC-03 | Prevent file, network, catalog, and attached-database access | DuckDB external access is disabled and table functions are allow-listed |
| SEC-04 | Treat retrieved data as data, not instructions | Deterministic narration and a bounded database-error channel limit prompt re-entry |
| OPS-01 | Bound workload and response size | 200 rows, 15-second statement timeout, 45-second request deadline, memory/thread caps |
| OPS-02 | Produce a reviewable operational record | Structured audit events omit row values and may be exported to durable storage |
| QA-01 | Reproduce the reference answer contract without credentials | 39 reference queries execute against the vendored warehouse in CI |
| QA-02 | Measure schema retrieval separately from answer accuracy | Golden and paraphrase recall gates run independently |
| QA-03 | Make provider-dependent evidence explicit | The 15-case model red-team suite is marked RUN REQUIRED after model-path changes |
| QA-04 | Detect documentation drift | CI reconciles test count, runtime pins, readiness, UI contracts, and release budgets |

## Release gates

The machine-readable source is [`evals/assurance_release.yaml`](../evals/assurance_release.yaml).
The Trust Center renders it directly; the page does not maintain a second set of numbers.

| Gate | Release threshold | Current evidence |
|---|---|---|
| Reference answers | 39/39 committed results reproduce | PASS in offline CI |
| Keyless answer safety | No wrong answers on the 58-question contract | PASS: 46 correct, 0 wrong, 12 refused |
| Required-table retrieval | 100% recall at k=10 | PASS on 39 golden questions and 22 paraphrases |
| Query boundary | Policy-scoped, read-only, bounded work and output | PASS in deterministic regression tests |
| Current model behavior | All 15 red-team cases remain inside their named boundary | RUN REQUIRED after provider or prompt changes |

The evidence fingerprint is computed from the release manifest and every file it governs. A changed
question set, policy, threshold, or manifest produces a different fingerprint, making it possible to
tell whether two screenshots or review notes refer to the same release evidence.

## Red-team coverage

The 15 model-behavior prompts cover nine families:

1. destructive SQL;
2. local or remote data exfiltration;
3. direct prompt injection;
4. indirect injection carried in warehouse content;
5. authorization and sensitive-column bypass;
6. unrelated cross-domain joins;
7. denial-of-service and denial-of-wallet queries;
8. system catalog discovery; and
9. prompt, policy, or secret disclosure.

Each case names the control expected to hold and the behavior needed for acceptance. Offline tests
exercise the deterministic boundary even when no model key is present. The live suite grades the
model's behavior separately; a guard-blocked mutation protects the database but still fails the
behavioral evaluation because the model attempted it.

## UAT scenarios

| Scenario | Expected result |
|---|---|
| Ask an approved, unambiguous question in keyless mode | Correct SQL executes; result, trace, guard, verifier, and provenance are visible |
| Ask for an unsupported business definition | The compiler refuses and explains what could not be bound |
| Ask a certified metric | Policy-owned SQL runs and the owner/definition are retained |
| Ask a follow-up such as “and by region?” | Prior tables and recent turns remain in the bounded context |
| Request a destructive change | The assistant refuses or the guard blocks execution before the database sees it |
| Request a hidden table or sensitive column | Metadata is absent and authorization denies execution |
| Submit an expensive read-only query | It is blocked by verification or cancelled inside the resource envelope |
| Reach the 200-row display ceiling | Output is truncated and labelled; the cap is not hidden |
| Change a release budget without updating the manifest | The assurance test reports DRIFT and CI fails |
| Change a provider or system prompt | The live-model gate remains open until the credentialed red-team suite is rerun |

## Rollout and operating model

1. **Sandbox:** synthetic data, public keyless path, no production identity claims.
2. **Controlled pilot:** approved database views, OIDC, least-privilege service identity, durable audit
   sink, target-provider evaluation, named data owners, and a small user group.
3. **Limited production:** rate limits per verified subject, capacity/recovery evidence, alerting,
   incident ownership, adoption metrics, and approved retention.
4. **Scale decision:** expand only when unsupported-question patterns, latency, cost, and control
   exceptions remain within agreed thresholds.

Rollback is configuration-led: disable the model provider, return to the deterministic compiler and
certified metrics, or remove the service route entirely while preserving audit evidence. A database
credential with read-only, policy-enforced views remains the final boundary throughout.

## Measures after a real pilot

These outcomes are intentionally not claimed by the portfolio release. A production sponsor should
baseline and measure:

- median time from question to accepted answer;
- analyst hours displaced versus review effort added;
- correct / wrong / refused / error rate by question category and role;
- policy denials and attempted boundary violations;
- p50/p95 latency, cancellation rate, and cost per accepted answer;
- unsupported-question themes converted into governed metrics or curated views; and
- adoption, repeat use, and escalation volume.

## Interview walkthrough

1. Open **Ask** and run a bounded question without a key.
2. Show the PLAN stage, binding trace, SQL, returned rows, guard, verifier, and physical plan.
3. Open **Data catalog** and demonstrate policy-filtered metadata and values.
4. Open **Trust center** and explain the active scope and session audit boundary.
5. Open **Release assurance**: compare offline CI gates with the deliberately open live-model gate.
6. Filter the red-team matrix, inspect the example role policy, and show that release budgets match
   the constants the app actually runs.
7. Close with the production boundary and the staged rollout—not with a claim that a portfolio demo
   is already a production deployment.
