# RECAP

[![CI](https://github.com/rezaxr14/Recap/actions/workflows/ci.yml/badge.svg)](https://github.com/rezaxr14/Recap/actions/workflows/ci.yml)

**R**etrieval-**E**nhanced **C**orrection and **A**daptive **P**ersistence for tool-using agents.

RECAP is a memory and recovery engine for agents that call tools. It records what an agent
did, compiles a typed repair from a failure, validates that repair by *executing* it in a
safe sandbox, promotes it to a retrievable memory only when the promotion policy passes, and
then serves those memories back with explicit compatibility and outcome-aware selection.

The central claim RECAP is built to test is not "RAG for agents". It is narrower and more
falsifiable: **retrieval relevance alone is not enough to decide whether a memory is safe to
apply.** A memory that is textually similar can be stale, schema-incompatible, or have a bad
verified track record. RECAP separates retrieval from selection so it can say *no safe
memory* instead of confidently returning the wrong one.

---

## What actually works

Everything below is implemented and exercised against real PostgreSQL and a real
Elasticsearch 8.19.3 node. There are no stub endpoints in the implemented paths.

| Subsystem | Status | Evidence |
|---|---|---|
| Run recording with idempotent events + redaction | Working | `tests/unit/test_run_recorder.py`, `tests/integration/test_security.py` |
| PostgreSQL source of truth (Postgres 17, Alembic) | Working | `migrations/versions/`, `tests/integration/test_postgres.py` |
| Elasticsearch BM25 + dense vector + hybrid RRF | Working | `tests/integration/test_elasticsearch.py` |
| Deterministic lexical reranker (`hybrid_rerank`) | Working | `tests/unit/test_reranker.py`, integration |
| Outcome-aware selection with rejection reasons | Working | `tests/unit/test_recall_policy.py` |
| RepairCompiler (failure window -> typed candidate) | Working | `tests/unit/test_repair_compiler.py` |
| Static + executing mock-sandbox validators | Working | `tests/unit/test_repair_compiler.py` |
| Promotion gate (no bypass path) | Working | `tests/unit/test_repair_compiler.py`, `tests/integration/test_api.py` |
| MemoryShift drift detection + revalidation queue | Working | `tests/integration/test_postgres.py`, demo step 8 |
| Benchmark with card + chunk ablations and safety audit | Working | `recap eval run`, `benchmark/reports/` |
| Transactional outbox + retry worker | Working | `tests/unit/test_outbox_worker.py`, `recap search drain` |
| Explicit memory versions + supersession | Working | `tests/unit/test_memory_versions.py`, `tests/e2e/test_lifecycle.py` |
| Run-level memory snapshot pinning | Working | `tests/integration/test_memory_snapshots.py` |
| Deterministic event-aware chunker (no downloads, fail-closed redaction) | Working | `tests/unit/test_chunking.py` |
| Chunk persistence, indexing, lifecycle, `reconcile --chunks` | Working | `tests/integration/test_chunks.py`, `migrations/versions/0004_memory_chunks.py` |
| Chunk-level retrieval (`*_chunk` modes) with supporting evidence | Working | `tests/integration/test_chunks.py`, `recap eval run` |
| API-first chat/embedding providers (OpenAI-compatible) | Implemented; live endpoint not configured in CI | `tests/unit/test_providers.py`, opt-in `tests/integration/test_live_providers.py` |
| LLM repair-candidate generation with typed validation | Working when `REPAIR_GENERATOR=llm` | `tests/unit/test_llm_generator.py` |
| Real LLM tool-using agent (`recap agent run`) | Implemented; live endpoint not configured in CI | `tests/unit/test_llm_agent.py`, `tests/integration/test_llm_agent_recording.py` |
| End-to-end demo (10 steps) | Working | `recap demo run` |

Honest limitations are listed in [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md). The
most important ones: the default embedding provider is a deterministic **hashing lexical
embedder**, not a neural model; the demo/benchmark still use the deterministic reference
agent for reproducibility; the Postgres **embedding cache is designed but not implemented**
(ADR 0004 section 5), so no cache metric is reported; and the chunk token counter is a
download-free approximation, not a model tokenizer. A configured-LLM tool-using agent exists
(`recap agent run`) and is validated with a scripted provider plus the real PostgreSQL
recorder; it has not been run against a real model endpoint here. The real-model provider
layer is unit-tested against the exact wire format, but no real endpoint is configured in CI,
so live-provider tests skip there. Benchmark numbers measure pipeline behaviour and safety
properties, not LLM capability. Reports and `recap providers` say which mode produced a
result.

---

## Quickstart

### 1. Services

```bash
docker compose up -d postgres elasticsearch
```

- PostgreSQL 17 on `localhost:5433` (`recap` / `recap` / `recap`)
- Elasticsearch 8.19.3 on `localhost:9200`, security disabled for local use only

### 2. Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Unix
pip install -e ".[dev]"
```

Python 3.12+ is required.

### 3. Initialize and verify

```bash
recap db migrate
recap search init
recap health
python scripts/run_smoke_test.py
```

`run_smoke_test.py` is operator-facing: it prints each check as `PASS`/`FAIL` and exits
non-zero on any deviation.

### 4. Configure real models (API-first)

RECAP never downloads model weights. Point it at an already-running OpenAI-compatible
endpoint:

```bash
export EMBEDDING_PROVIDER=openai_compatible
export EMBEDDING_BASE_URL=https://api.example.com/v1
export EMBEDDING_API_KEY=...
export EMBEDDING_MODEL=text-embedding-3-small
export EMBEDDING_DIMENSIONS=1536

export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://api.example.com/v1
export LLM_API_KEY=...
export LLM_MODEL=gpt-4.1-mini

recap providers          # offline capability report, no secrets
recap doctor --live      # probes configured endpoints, one small chat + embedding
recap agent run --environment-version 2 --workspace live-agent
```

`recap agent run` drives the configured chat model through a bounded tool-calling loop
against the mock API. The model selects `list_records` arguments; RECAP validates the tool
name and arguments, executes only that local tool, and computes success from the records
actually collected. It refuses to run when no `llm` role is configured and never falls back
to the deterministic agent. The recorded run includes provider, model, prompt version, token
usage, and every tool call/failure.

`hashing`/`fake` embeddings remain available for offline development and unit tests, but they
are reported as non-semantic. Production rejects them unless `EMBEDDING_ALLOW_DEGRADED=true`
is explicitly set. The embedding dimension must match the Elasticsearch mapping; changing
models requires a new index generation. Details: [`docs/providers.md`](docs/providers.md).

### 5. Run the demo

```bash
recap demo run
```

This executes the whole story end to end and writes `demo_report.md`:

1. Record a failed run against mock API v1 (expired cursor).
2. Compile a repair candidate from the failure window.
3. Validate it in the mock sandbox (actually reproduces the failure, applies the strategy).
4. Promote it through the policy gate.
5. Retrieve it for a new run and apply it; the run succeeds.
6. Upgrade the environment to v2 (parameter renamed `cursor` -> `page_token`).
7. Try to reuse the v1 memory: it is rejected as incompatible. The run **fails closed**
   rather than applying a stale repair.
8. Compile, validate, and promote a v2 memory.
9. The v2 memory is retrieved and the run succeeds.
10. Run the benchmark and emit JSON/CSV/Markdown reports.

### 6. Benchmark

```bash
recap eval run --split dev
recap eval run --split test
```

Reports land in `benchmark/reports/`. See
[`docs/evaluation-methodology.md`](docs/evaluation-methodology.md) for what the numbers do
and do not mean.

### 7. HTTP API

```bash
recap serve --port 8000
```

`GET /health/live`, `GET /health/ready`, `GET /api/v1/diagnostics`. Full route list in
[`docs/operations.md`](docs/operations.md).

---

## CLI surface

```
recap version
recap health
recap providers
recap doctor [--live]
recap serve

recap db migrate | init
recap search init | health | reconcile [--chunks] | drain | outbox
recap memory search | show | compatibility
recap chunk build | show | count
recap repair compile | validate | promote
recap agent run --environment-version 1
recap demo seed | run
recap eval run
```

Every command performs real work. Nothing prints success for an operation that did not run.

---

## Architecture in one paragraph

PostgreSQL is the source of truth. Elasticsearch is a rebuildable index behind
`recap-memories-read` / `recap-memories-write` aliases with a versioned index name; a second
chunk index (`recap-chunks-*`) stores deterministic chunks derived from the same memories and
their evidence runs, so retrieval can be ablated at card or chunk granularity. Model
access is isolated behind `ChatProvider`/`EmbeddingProvider` ports with independent agent,
repair, and embedding roles; the remote adapters speak the OpenAI-compatible API and never
download weights. A run is recorded as an ordered event stream; a failure window is extracted
from that stream; the RepairCompiler turns it into a typed, allowlisted strategy; validators
execute it in a disposable mock environment; the PromotionService is the only path to
`active`; and the RecallEngine fuses lexical and vector branches, applies hard filters,
evaluates preconditions and compatibility, scores utility with logged features, and persists
the full decision including *why each candidate was rejected*. MemoryShift watches outcomes
and environment fingerprints, downgrades memories to `suspect`, and queues revalidation
instead of silently rewriting history.

Deeper reading:

- [`docs/architecture.md`](docs/architecture.md) — components, boundaries, request flow
- [`docs/data-model.md`](docs/data-model.md) — tables, lifecycle, invariants
- [`docs/retrieval.md`](docs/retrieval.md) — branches, fusion, policy, ablation design
- [`docs/repair-validation.md`](docs/repair-validation.md) — failure windows, validators, promotion
- [`docs/drift-detection.md`](docs/drift-detection.md) — compatibility, degradation, revalidation
- [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md) — baselines, metrics, leakage
- [`docs/providers.md`](docs/providers.md) — model roles, API endpoints, degraded mode, live tests
- [`docs/threat-model.md`](docs/threat-model.md) — assets, attackers, controls, non-goals
- [`docs/operations.md`](docs/operations.md) — runbook, health, recovery, failure modes

ADRs:

- [`0001-source-of-truth.md`](docs/adr/0001-source-of-truth.md)
- [`0002-elasticsearch-hybrid-retrieval.md`](docs/adr/0002-elasticsearch-hybrid-retrieval.md)
- [`0003-memory-promotion-policy.md`](docs/adr/0003-memory-promotion-policy.md)

---

## Safety posture

- **Retrieved memory is untrusted data.** It is rendered as structured fields; RECAP never
  executes code from a memory. Actions are typed enum templates interpreted by the caller.
- **Fail closed.** Unknown preconditions, incompatible or suspect compatibility, missing
  validation, and low utility all produce `no_safe_memory` rather than a best guess.
- **No promotion bypass.** `PromotionService` is the only writer of `active` memories from a
  candidate, and it re-checks evidence, validation, duplicates, and secret leakage.
- **Redaction at the boundary.** Payloads are redacted before persistence, and memory free
  text is re-redacted before indexing.
- **No secrets in git.** `.env` is ignored; `.env.example` contains only local defaults.

See [`docs/threat-model.md`](docs/threat-model.md) for the full analysis, including what
RECAP explicitly does *not* defend against.

---

## Development

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy src

pytest tests/unit              # unit tests only, no services
pytest --run-integration       # + live Postgres/Elasticsearch
```

Integration tests are opt-in and use an isolated database (`recap_test`) and index prefix
(`recap-test-*`), so they cannot touch development data.

Live model-endpoint tests are additionally gated:

```bash
export RECAP_RUN_PROVIDER_INTEGRATION=1
pytest tests/integration/test_live_providers.py --run-integration
```

They skip when no real endpoint is configured. They are the only acceptable evidence for
real-model integration; mocked transport tests and the deterministic demo are not.

Windows note: `make` is often unavailable. `scripts/dev_targets.py` mirrors the Makefile
targets; `Makefile` works on Unix and in CI.

### Repository layout

```
src/recap/
  domain/            pure models, enums, predicates, lifecycle rules
  recorder/          event ingestion, normalization, redaction
  recall/            query building, eligibility, applicability, utility policy, engine
  repair_compiler/   failure windows, generation, validation, promotion
  memory_shift/      compatibility, degradation, revalidation
  providers/         ChatProvider/EmbeddingProvider ports + OpenAI-compatible adapters
  infrastructure/    SQLAlchemy, Elasticsearch, embeddings
  environments/      mock API, deterministic reference agent, configured-LLM agent
  evals/             datasets, metrics, benchmark runner
  api/               FastAPI routes
  demo.py            the 10-step end-to-end flow
migrations/          Alembic
configs/             default.yaml
docs/                design + operations documentation
```

---

## License

Apache-2.0. See `pyproject.toml`.
