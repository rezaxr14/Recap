# Phase 4 Ask endpoint evidence

This page records the original coverage verification for `POST /api/v1/ask`. It is coverage
evidence, not a live-model answer-quality evaluation. The follow-up 40-question mini-eval is
recorded in [`phase4-ask-mini-eval.md`](phase4-ask-mini-eval.md) (deterministic) and
[`phase4-ask-mini-eval-live-qwen.md`](phase4-ask-mini-eval-live-qwen.md) (live `qwen3.5-4b`).

## Command

```powershell
$env:RECAP_RUN_INTEGRATION='1'
.venv\Scripts\python.exe -m pytest -q `
  tests\unit\test_ask.py tests\integration\test_ask.py --run-integration
```

Result on 2026-10-22: `10 passed` against the local PostgreSQL and Elasticsearch services.

## Covered behavior

- Extractive answer over an indexed memory, with a verified citation.
- Answer to a question that names a recorded run, citing the run.
- Quarantined memory retrieved through the bounded PostgreSQL lifecycle supplement (it is
  absent from the operational Elasticsearch index by design).
- `insufficient_evidence` for a question with no matching recorded evidence.
- `configuration_error` for `mode=llm` when no agent provider is configured; no silent
  extractive fallback.
- LLM mode with a scripted provider: a fabricated quote is downgraded to
  `insufficient_evidence`, while a verbatim quote is accepted and the provider/model fields are
  reported.

## What this does not establish

- No live model was used in this suite. The LLM path was exercised with a deterministic
  scripted provider so the verification contract, not the model's writing quality, is tested.
- Verification checks source-id existence and quote substring grounding only. It does not
  prove entailment or answer completeness.
- Retrieval over run/event history is bounded, and the quarantined-memory supplement is a
  recent-row query rather than an exhaustive lifecycle index.
- The separate static replay app's Q&A is report-grounded and does not call this endpoint.

See [`../ask.md`](../ask.md) for the contract and limitations.
