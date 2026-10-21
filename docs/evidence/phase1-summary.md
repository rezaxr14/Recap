# Phase 1 evidence: real models, reindex, and reranker

Generated: 2026-10-20 UTC. This file separates live evidence from CI/unit coverage and
records the exact model names used. No number below is projected.

## Models

| Role | Exact model / artifact | Source |
|---|---|---|
| Agent and repair chat | `gemma-4-e4b-uncensored-hauhaucs-aggressive` | LM Studio, `http://localhost:1234/v1` |
| Embeddings | `text-embedding-nomic-embed-text-v1.5`, 768 dimensions | LM Studio; nomic `search_document: ` / `search_query: ` prefixes |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | local directory `models/ms-marco-MiniLM-L-6-v2`, loaded with `local_files_only=True` |

The user selected `gemma-4-e4b-uncensored-hauhaucs-aggressive` for both chat roles. The
reranker weights are not committed; `models/` is ignored.

## Live provider and agent evidence

- `docs/evidence/phase1-doctor-gemma.txt`: `recap doctor --live` sees three LM Studio models,
  returns database/Elasticsearch `ok`, `semantic_retrieval=true`, and a 768-d embedding probe.
- `docs/evidence/phase1-live-provider-tests-gemma.txt`: `6 passed in 84.56s` for the opt-in
  live provider suite (chat, structured JSON, embeddings, real ES retrieval, tool agent).
- `docs/evidence/live-runs/agent-v1-gemma.json`: run
  `run_9d88d3a1651e4b7aaa40cccd12bd42a8`, environment v1, succeeded with 1 tool call,
  2 model turns, 755 input / 41 output tokens, 6,657 ms.
- `docs/evidence/live-runs/agent-v2-gemma.json`: run
  `run_341dfc26291a4217abe36cf863b378b4`, environment v2, succeeded with 1 tool call,
  2 model turns, 757 input / 41 output tokens, 5,243 ms.

These are real local inference transcripts. They are not CI coverage; `test_live_providers.py`
still skips unless `RECAP_RUN_PROVIDER_INTEGRATION=1` is set.

## LLM repair generation: honest negative result

The acceptance harness creates held-out failure windows, asks the configured `repair_llm` for
a typed candidate, then runs static and mock-sandbox validation. Two configurations were
measured with the selected gemma model:

| Configuration | Runs | Schema successes | Candidates | Static / sandbox passes | Result |
|---|---:|---:|---:|---:|---|
| `json_schema`, 60 s timeout, 2 retries | 4 | 0 | 0 | 0 / 0 | all 4 runs failed after provider timeouts (12 HTTP timeout messages); `751,248 ms` total |
| `json_object`, 240 s timeout, 1 attempt | 2 | 0 | 0 | 0 / 0 | 2 `ProviderResponseError: provider rejected the request` |

Evidence:

- `phase1-repair-generator-gemma-json-schema-timeout.json`
- `phase1-repair-generator-gemma-json-object.json`

The harness did not produce a single schema-valid candidate with this model, so it **does not**
support a claim that gemma can generate repair candidates reliably. It also does not justify a
"0% format failure" claim: the failures happened at the provider boundary before Pydantic
format validation. The earlier schema issue that rejected an `Any` field was fixed; the
remaining failure is model/provider behavior, not a hidden unit-test result. A follow-up should
try the configured `llama-3.2-3b-instruct` role or a smaller repair schema before treating LLM
repair generation as production-ready.

## Index generation rebuild

Implementation: `recap search reindex --to-generation vN`.

It loads active memories from Postgres, re-embeds them with the configured provider, creates a
new concrete index, validates vector dimension and embedding fingerprint, refreshes and counts
the target, swaps read/write aliases atomically, writes generation provenance to Postgres, and
deletes the partial target on pre-swap failure. The old index is retained unless `--delete-old`
is passed after success. Writes target the active write alias, so a reindex remains durable
across processes that still have the old generation in configuration.

Evidence: `tests/integration/test_reindex.py` (3 tests) covers successful swap and DB
provenance, alias-authoritative reinitialization with stale config, dimension rejection, and
rollback. The command is documented in README and `docs/operations.md`.

## Local CrossEncoder reranker

The reranker is behind `RerankerProvider`; `lexical` remains the default and `none` disables
the stage. `cross_encoder` requires an existing local path and never downloads weights.

Evidence: `docs/evidence/phase1-rerank-benchmark/` is a real benchmark run with
`RERANK_PROVIDER=cross_encoder` and the MiniLM artifact above. The report records
`retrieval.active_reranker=cross-encoder:ms-marco-MiniLM-L-6-v2`, the configured model path,
and per-stage latency:

- `hybrid_rerank` total p50: `278.381 ms` (p95 `296.667 ms`)
- rerank stage p50: `119.441 ms` (p95 `127.230 ms`)

The quality metrics on this two-task dev split are degenerate and must not be presented as a
reranker win. The useful evidence here is that the local model runs, the stage is measured, and
provenance is recorded. The benchmark still uses the deterministic reference agent and the
hashing embedder, not `gemma`/nomic; see the report's `protocol` and `embedding` sections.

## Verification commands

```powershell
# unit only
.\.venv\Scripts\python.exe -m pytest tests/unit -q

# real Postgres + Elasticsearch
docker compose up -d postgres elasticsearch
.\.venv\Scripts\python.exe -m pytest -q --run-integration

# live LM Studio provider suite
$env:RECAP_RUN_PROVIDER_INTEGRATION='1'
.\.venv\Scripts\python.exe -m pytest tests/integration/test_live_providers.py -q --run-integration

# reranker benchmark evidence
$env:RERANK_PROVIDER='cross_encoder'
$env:RERANK_MODEL_PATH='models/ms-marco-MiniLM-L-6-v2'
.\.venv\Scripts\recap.exe eval run --split dev --output docs/evidence/phase1-rerank-benchmark
```

The local Compose services are configured with `restart: "no"` so they do not return on
Docker/Windows startup. Start them explicitly for integration tests and stop them afterward.
