# RAG review — September 7, 2026

## Verdict

The current RAG is appropriate for this 71-table analytics application. It is
**schema RAG, not a general document chatbot**: it retrieves table definitions
and permitted SQL examples, then answers from an executed query. Adding a vector
database or another model is not, by itself, an enterprise upgrade.

The review found and fixed two concrete gaps:

1. **Scope before the retrieval budget.** Vector and keyword candidates now
   respect allowed tables before top-k selection and reciprocal-rank fusion.
   Previously forbidden global candidates could crowd out authorized tables;
   a subsequent filter could leave too little context or invoke an oversized
   catalog fallback. Empty scopes do not query the index. Final prompt and
   display masking remains in force.
2. **Schema-aware cache invalidation.** In-process index reuse and UI retrieval
   caching now include a revision of live columns and committed descriptions.
   A separate metadata cursor avoids disrupting concurrent query results.
   Unchanged schemas reuse the index; column/description changes invalidate it.
   No result values enter this revision or leave the server.

## Measured retrieval quality

At k=10, percentage of questions for which **every required table** is retrieved:

| Evaluation set | Keyword | Vector | Hybrid RRF |
|---|---:|---:|---:|
| 39 golden questions | 100.0% | 94.9% | 100.0% |
| 22 development paraphrases | 86.4% | 90.9% | 100.0% |

Required-table labels come from each question's committed reference SQL.
Paraphrases use distinct wording across all 11 domains; their labels were
checked against the reference query's domain, population and measure before the
final scoring run. This is a development challenge set, **not independently
collected real-user evidence**. Retrieval coverage does not prove correct SQL,
correct business interpretation, or a factually correct final answer.

The hybrid prompt averages approximately 2,253 tokens on the golden set and
2,176 on the paraphrases, versus 12,741 for the full catalog. These are rough
character-based token estimates, not provider tokenizer measurements. On the
local validation machine, warm hybrid retrieval p50/p95 was approximately
149/169 ms for golden wording and 176/214 ms for paraphrases while other tests
were running. These exclude initial model loading and are not production SLOs.
First-relevant-table MRR is also printed by the harness; hybrid optimizes
coverage here and does not outperform vector on every ranking metric.

CI now fails if hybrid full-question coverage falls below 100% on either
committed set. A regression test proves that a missing required table makes the
gate fail. Failures remain visible rather than being removed from the dataset.

```powershell
python scripts/run_retrieval_eval.py --min-coverage 1
python scripts/run_retrieval_eval.py --paraphrases --min-coverage 1
python -m pytest tests/test_rag_boundaries.py -q
```

## Existing controls retained

- Local, checksum-verified MiniLM embeddings; exact cosine search plus keyword
  matching and reciprocal-rank fusion. No external embedding service.
- Follow-up context carries recent questions/SQL, with prior SQL tables included
  explicitly in model prompts, subject to the current access policy.
- Restricted tables/examples are excluded, masked columns are omitted, and
  descriptions that could disclose masked metadata are removed from prompts.
- A failed retrieval can use the permitted catalog; read-only SQL authorization
  and structural verification still apply. Successful query facts are rendered
  from returned cells rather than unconstrained model-generated prose.

## Remaining enterprise work

- Collect independent real-user questions, ambiguous-domain examples,
  unanswerable requests and multi-turn topic switches. Evaluate retrieval and
  end-to-end SQL/result correctness separately with a real configured provider.
- Ranking scores are not answer-confidence probabilities. The retriever is not
  an answerability classifier; guarded execution/refusal is a separate stage.
- Validate real IdP roles, warehouse-enforced row isolation and metadata policies.
  The public demo remains synthetic and does not certify a multi-tenant boundary.
- Benchmark a much larger, representative catalog before adopting reranking,
  adaptive context budgets or a distributed vector store. Scope is applied to
  candidate selection; the trusted server still holds a shared embedding index.
- Add deployment telemetry for schema revision, fallback frequency, per-role
  recall, cold-start and concurrency latency. This release's cache refresh is
  not a complete source-ingestion or schema-migration service; other semantic
  snapshots still require controlled lifecycle/reload management.

See [release verification](RELEASE_2026_09_07.md) for voice/UI checks and the
remaining operational prerequisites for real organizational data.
