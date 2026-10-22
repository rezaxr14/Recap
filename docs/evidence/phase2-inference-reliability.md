# Phase 2.3 evidence: inference reliability and live repair acceptance

This page records the Phase 2.3 work that made slow local inference observable and
recoverable. It separates three kinds of evidence:

1. **Live local inference** against LM Studio on this machine.
2. **Unit/CI coverage** with deterministic mock transports.
3. **Historical negative evidence** from Phase 1, which is kept for the record and superseded
   by the result below.

The deterministic reference agent and deterministic repair generator remain the defaults.
Nothing here claims that an LLM is required for the pipeline.

## Configuration under test

| Item | Value |
|---|---|
| Chat model | `gemma-4-e4b-uncensored-hauhaucs-aggressive` |
| Chat endpoint | `http://localhost:1234/v1` (LM Studio, OpenAI-compatible) |
| Embedding model | `text-embedding-nomic-embed-text-v1.5` (768-d) |
| Repair generator | `llm` (opt-in), `max_candidate_generation_attempts=2` |
| Streaming | enabled for both chat roles |
| Chat concurrency | `max_concurrency=1` (one local generation at a time) |
| Agent role deadlines | connect 10 s, read 600 s, write 30 s, pool 30 s, overall 900 s, no-progress 180 s |
| Repair role deadlines | connect 10 s, read 600 s, write 30 s, pool 30 s, overall 1200 s, no-progress 240 s |
| Output bound | 512 tokens (agent), 1024 tokens (repair) |
| Transport retries | 2, with backoff; repair structured-format retries are separate and bounded |
| Postgres / Elasticsearch | live Docker services; Elasticsearch `8.19.3`, cluster status `green` |

These values are in `configs/default.yaml`. Every value is overridable through the environment
variables documented in [`docs/providers.md`](../providers.md).

## What changed

- **SSE streaming.** The OpenAI-compatible adapter consumes `text/event-stream`, reassembles
  content and tool-call deltas, reads usage when the server sends it, and records time to first
  token. A stream that ends without a payload is an explicit response error.
- **Per-phase deadlines.** Connecting, uploading, waiting for a token, and finishing the whole
  call are separate limits. A stalled stream is cancelled cleanly instead of being mistaken for
  a dead model.
- **No-progress and overall caps.** The stream is closed when no new SSE line arrives within the
  no-progress deadline or when the overall wall-clock budget is exhausted. `ProviderTimeout`
  (code `provider_timeout`, HTTP 504) makes the cause explicit.
- **Bounded retries and accounting.** `ChatResult` records `attempts`, `retries`,
  `first_token_ms`, and `streamed`; agent runs aggregate them; benchmark rows expose them. The
  benchmark success-rate denominator now excludes provider infrastructure outcomes, which are
  reported separately.
- **One local generation at a time by default.** A semaphore around the chat provider avoids
  queueing multiple requests inside LM Studio and then attributing the queue time to the model.
- **Structured repair retries.** Invalid JSON and truncated structured output are counted as
  format failures and retried within the configured attempt budget; endpoint outages remain a
  separate provider-error category.
- **Operator aliases.** Model-generated `equals`/`not_equals` are normalized to canonical
  `eq`/`neq`; unknown operators are still rejected. The repair prompt now lists the exact
  operators, allowed fields, and the required
  `tool_schema_fingerprints.list_records` precondition.

## Live doctor result

Source: [`phase2-inference-doctor-gemma.txt`](phase2-inference-doctor-gemma.txt)

| Probe | Result |
|---|---|
| Database | `ok` |
| Elasticsearch | `8.19.3`, `green`, one node |
| Agent chat | streamed, elapsed `622.274 ms`, first token `611.446 ms`, 1 attempt, 0 retries |
| Repair chat | streamed, elapsed `315.187 ms`, first token `309.103 ms`, 1 attempt, 0 retries |
| Embedding | 768 dimensions in `65.272 ms` |
| Provider usage | not reported by the streaming endpoint; `completion_tokens=0`, `usage_reported=false`, `tokens_per_second=null` |

The endpoint did not return completion-token usage for the streaming probes, so no tokens per
second value is invented. The report also prints the configured deadlines and a conservative
recommended read timeout derived from the observed probe.

## Live repair-generation acceptance

Source: [`phase2-repair-generator-gemma-final.json`](phase2-repair-generator-gemma-final.json)
(console transcript: [`phase2-repair-generator-gemma-final-console.txt`](phase2-repair-generator-gemma-final-console.txt))

| Metric | Result |
|---|---:|
| Runs | 4 |
| Schema-valid candidates | 4 / 4 |
| Schema success rate | 1.000 |
| Generation attempts | 4 |
| Retries | 0 |
| Format failures | 0 |
| Generation errors | 0 |
| Static validation passes | 4 / 4 |
| Mock-sandbox validation passes | 4 / 4 |
| End-to-end pass rate | 1.000 |
| Total elapsed | `181,378 ms` |

All four candidates reproduced the original failure, held their preconditions (including the
exact tool-schema fingerprint), were rejected on the mismatched schema version, executed the
allowlisted strategy, and satisfied their postconditions. The candidates were proposals only:
the normal promotion gate still decides whether any memory may be promoted.

This is four live runs on one model and one machine. It is evidence that the configured model
can complete this bounded task with the current prompt and validation path; it is not a
multi-model reliability estimate and it is not CI coverage.

## Historical negative result

The Phase 1 failures remain published in
[`phase1-summary.md`](phase1-summary.md) and the two `phase1-repair-generator-*.json` files.
They failed at the provider boundary (`json_schema` timeouts) and on an unsupported predicate
vocabulary (`equals`/`not_equals`). The Phase 2.3 result supersedes that outcome for the exact
configuration above, but does not erase it.

## Verification commands

The live commands used:

```powershell
$env:LLM_PROVIDER='openai_compatible'
$env:LLM_BASE_URL='http://localhost:1234/v1'
$env:LLM_MODEL='gemma-4-e4b-uncensored-hauhaucs-aggressive'
$env:LLM_API_KEY='<local endpoint token>'
$env:LLM_READ_TIMEOUT_SECONDS='600'
$env:LLM_OVERALL_TIMEOUT_SECONDS='900'
$env:LLM_NO_PROGRESS_TIMEOUT_SECONDS='180'

$env:REPAIR_LLM_PROVIDER='openai_compatible'
$env:REPAIR_LLM_BASE_URL='http://localhost:1234/v1'
$env:REPAIR_LLM_MODEL='gemma-4-e4b-uncensored-hauhaucs-aggressive'
$env:REPAIR_LLM_API_KEY='<local endpoint token>'
$env:REPAIR_LLM_READ_TIMEOUT_SECONDS='600'
$env:REPAIR_LLM_OVERALL_TIMEOUT_SECONDS='1200'
$env:REPAIR_LLM_NO_PROGRESS_TIMEOUT_SECONDS='240'

$env:EMBEDDING_PROVIDER='openai_compatible'
$env:EMBEDDING_BASE_URL='http://localhost:1234/v1'
$env:EMBEDDING_MODEL='text-embedding-nomic-embed-text-v1.5'
$env:EMBEDDING_DIMENSIONS='768'
$env:EMBEDDING_READ_TIMEOUT_SECONDS='120'

$env:REPAIR_GENERATOR='llm'
.\.venv\Scripts\recap.exe doctor --live
.\.venv\Scripts\recap.exe eval repair-generator --runs 4 `
  --output docs/evidence/phase2-repair-generator-gemma-final.json
```

Deterministic coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_providers.py -q
.\.venv\Scripts\python.exe -m pytest tests/unit/test_domain.py -q
.\.venv\Scripts\python.exe -m pytest tests/unit -q
```

The provider tests cover streaming content, tool-call delta reassembly, stalled streams,
slow-but-progressing streams, transport retries, non-streaming JSON fallback, and timeout
validation. Those tests use `httpx.MockTransport`; they do not call LM Studio.

Phase gate on this change set:

| Check | Result |
|---|---|
| `ruff check src tests` | passed |
| `ruff format --check src tests` | 112 files already formatted |
| `mypy src` | no issues in 78 source files |
| full suite with live Postgres/Elasticsearch (`pytest -q --run-integration`) | `279 passed, 1 skipped` (the opt-in live-provider test is the skip) |

The full-suite command now uses pytest's importlib mode because the unit and integration suites
share a few test basenames; running them together previously collided in pytest's module cache.
