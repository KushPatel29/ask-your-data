# Service levels and incident runbook

## Service-level objectives

Measure SLIs at the authenticated gateway. Exclude planned maintenance and
policy-correct refusals from availability; never exclude provider, query,
timeout, or internal errors.

| SLI | 30-day target | Fast alert | Burn alert |
|---|---:|---:|---:|
| Authenticated request availability | 99.9% | <99% over 10 min | 14.4× budget over 1 h or 6× over 6 h |
| Keyless answer latency | p95 <5 s, p99 <10 s | p95 >8 s for 10 min | p95 >5 s for 2 h |
| Model answer latency | p95 <30 s, p99 <45 s | p95 >40 s for 10 min | p95 >30 s for 2 h |
| Query execution | p99 <5 s; 100% cancelled by configured bound plus cooperative-cancel allowance | timeout rate >1% for 10 min | timeout rate >0.2% for 6 h |
| Critical KPI contract accuracy | 100% for released metric/version pairs | any mismatch | any mismatch blocks release |
| Cross-tenant/restricted-column denial | 100% in negative tests and probes | any unauthorized success | immediate severity 1 |
| Audit delivery reconciliation | 99.99% within 5 min | backlog age >5 min | any unreconciled gap >30 min |

The 99.9% availability target allows about 43 minutes of bad time in a 30-day
window. Freeze non-remediation releases after 25% of the monthly error budget is
spent; require reliability review after 50%; at 100%, ship only restoration and
security work until the window recovers.

## Telemetry contract

Every request emits one ID shared across gateway, planner, verifier, query,
provider, audit outbox, and n8n event. Record tenant pseudonym, authenticated
subject pseudonym, policy/model/prompt versions, outcome, refusal class, table
IDs, row count, truncation, token counts, stage durations, timeout stage, and
correction count. Do not emit JWTs, API keys, questions, SQL, row values,
provider error bodies, audio, or answer text to metrics/traces.

## First response

1. Acknowledge the page; assign incident commander and communications owner.
2. Determine blast radius by tenant, engine, model version, policy version, and
   deployment revision. Do not paste sensitive questions or rows into chat.
3. For suspected unauthorized access, disable model/manual paths at the gateway,
   preserve append-only audit evidence, rotate affected credentials, and page
   security/privacy. Do not destroy sessions or logs.
4. For elevated latency, preserve keyless traffic, shed model traffic by
   principal quota, verify provider/query saturation, and disable optional TTS
   and automation delivery before touching the answer path.
5. For wrong answers, stop the affected metric/model release, retain SQL and
   policy evidence in the restricted investigation store, switch to a known-good
   version or certified metric, and identify all exposed decisions.

## Failure-specific recovery

- **Identity/JWKS:** fail closed. Confirm issuer/audience and key rotation; never
  switch to unsigned headers or disabled auth as an availability workaround.
- **Policy service/file:** fail closed for queries and schema browsing. Roll back
  only to a signed, previously tested policy version.
- **Warehouse:** cancel expensive work, trip admission control, and fail over to
  a replica only after RLS/policy parity probes pass.
- **Model provider:** use the deterministic compiler where coverage permits;
  preserve an explicit refusal elsewhere. Do not silently switch model aliases.
- **Audit sink:** keep answers available only while the durable outbox accepts
  records. If the outbox itself is unavailable, fail closed for regulated data.
- **n8n:** queue signed events for replay. It is not a reason to fail answers.
- **Speech:** disable voice controls and retain typed interaction.

## Evidence and closure

Capture request/release/policy/model IDs, redacted timeline, affected tenants,
SLI impact, containment, restoration proof, and audit reconciliation. A severity
1 post-incident review is due within five business days with owners and dates.
Close only after negative authorization probes, critical KPI contracts,
synthetic availability, audit reconciliation, and rollback validation all pass.
