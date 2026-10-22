# Ask RECAP: grounded Q&A with verified citations

`POST /api/v1/ask` answers questions over recorded runs, events, and memories. It is not
a chatbot with general knowledge: every answer must cite a source in the request workspace,
and the API verifies that each quote actually occurs in that source.

```bash
curl -s http://localhost:8000/api/v1/ask \
  -H 'content-type: application/json' \
  -d '{"workspace_id":"demo","question":"Which memory fixes INVALID_CURSOR?"}'
```

```json
{
  "status": "answered",
  "answer": "Memory mem_... is recorded as active. ...",
  "citations": [{"type": "memory", "id": "mem_...", "quote": "Status: active"}],
  "answerer": "extractive",
  "verification": {"checked": 1, "valid": 1, "invalid": 0, "rejected": []},
  "retrieval": {"mode": "hybrid_rrf", "memory_hits": 1, "run_hits": 0, "event_hits": 0}
}
```

Request fields: `workspace_id` (required), `question` (required, 3-2000 characters),
`mode` (`extractive` or `llm`), `max_sources` (1-20), and `max_runs` (1-200). The response
reports `answerer`, provider/model for LLM answers, a citation verification record, and the
retrieval trace. `answerer=extractive` is a deterministic template over quotes, not a hidden
LLM call.

## Modes

| Mode | What generates the answer | Requires a model? |
|---|---|---|
| `extractive` (default) | assembles a sentence from the highest-ranked verified source quotes | no |
| `llm` | configured agent LLM (`llm` provider role) writes a short answer from the supplied evidence | yes |

`extractive` is the CI-tested path. `llm` is implemented and unit/integration-tested with a
scripted provider, but no live-model Ask evaluation has been run. Its confidence field is
not calibrated and is never used instead of citation verification.

If `mode=llm` is requested without a configured provider, the endpoint fails with
`configuration_error`; it never silently falls back to the extractive answer.

## Retrieval

- **Memories:** hybrid BM25 + dense-vector retrieval through the operational Elasticsearch
  index, fused with RRF. If the embedder is unavailable, the response is marked degraded and
  retrieval continues with BM25 only.
- **Lifecycle supplement:** quarantined/retired memories are deliberately absent from the
  operational index, so a bounded recent PostgreSQL query adds them for status questions.
  An explicitly named `mem_...` id is fetched directly.
- **Runs and events:** bounded lexical ranking over recent runs in the workspace. An
  explicitly named `run_...` id is always included even if it falls outside the row cap. Events
  are read only for runs that contribute evidence.
- Only sources with a positive query-term overlap are given to the answerer. If nothing
  matches, the response is `insufficient_evidence`, not a plausible guess.

This is intentionally not a general web-search or multi-hop planner. Retrieval over run
history is bounded by `max_runs` and the recent-row cap; memory retrieval inherits the index
generation and embedding model configured for the workspace.

## Programmatic verification

For every citation:

1. The cited type/id must be one of the sources placed in the model context.
2. The quote must be a substring of that source's evidence text after whitespace is
   normalized. Case is preserved, so a rewritten quote fails.
3. If any citation fails, or the answer has no citations, the whole response is downgraded
   to `status: "insufficient_evidence"` and the rejected citations are reported.
4. For `extractive` answers, the same verifier runs on generated citations as a self-check.

Verification proves **source grounding of the quote**, not that the answer is a complete or
correct interpretation of the evidence. A quote can be real while the surrounding sentence is
misleading; this is a known limitation, not something an LLM-as-judge would fix.

## Tests and evidence

- `tests/unit/test_ask.py`: citation existence, whitespace normalization, case-sensitive quote
  matching.
- `tests/integration/test_ask.py` (real Postgres + Elasticsearch): memory answer, named-run
  answer, quarantined-memory supplement, unanswerable refusal, missing-provider failure, and
  LLM fabricated-quote downgrade/valid-quote acceptance with a scripted provider.

```powershell
$env:RECAP_RUN_INTEGRATION='1'
.venv\Scripts\python.exe -m pytest -q tests\unit\test_ask.py tests\integration\test_ask.py --run-integration
# 10 passed
```

No claim is made that this endpoint improves answer accuracy. It demonstrates grounded
retrieval, explicit abstention, and citation verification on the same service graph as the
rest of RECAP.
