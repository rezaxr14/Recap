# RECAP

[![CI](https://github.com/rezaxr14/Recap/actions/workflows/ci.yml/badge.svg)](https://github.com/rezaxr14/Recap/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

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

**Explore the recorded runs:** the recruiter-facing replay app in [`web/`](web/) is a
no-build static page. It shows a silent-drift contrast, the v3 acceptance gate, the full
baseline matrix, rejected candidates with reasons, and a report-grounded Q&A. The provenance
bar separates deterministic evidence from live-model evidence. Run it with
`python -m http.server 8000 --directory web` after exporting a report; see
[`web/README.md`](web/README.md).

## Project status

**Alpha, single-node, deterministic by default.** The vertical slice is real: every
implemented local path is exercised against PostgreSQL 17 and Elasticsearch 8.19.3, and the
full suite runs in CI. It is not a hosted product. There is no multi-tenant auth layer (only
an optional shared API key), the default embedder is a deterministic hashing embedder rather
than a semantic model, and the
bundled evaluation split is deliberately small. The deterministic snapshot measures pipeline
and safety behaviour, while the separate live-model snapshot measures one configured model in
a deterministic mock environment. Neither is a general LLM-capability claim. See
[Known limits](#known-limits) for the honest gaps.

| At a glance | |
|---|---|
| **Core loop** | record -> compile a typed repair -> execute it in a sandbox -> promote only if valid -> retrieve with compatibility and outcome-aware selection |
| **Source of truth** | PostgreSQL 17 (runs, memories, chunks, outbox, snapshots, audit) |
| **Search** | Elasticsearch 8.19: BM25, dense vectors, hybrid RRF, card and chunk indices |
| **Interfaces** | `recap` CLI, FastAPI HTTP API, Python package, static replay web app |
| **Stack** | Python 3.12, FastAPI, SQLAlchemy 2, Alembic, Pydantic 2, Docker Compose |
| **Verification** | `311 passed, 1 skipped` locally with `--run-integration`; real-service integration tests; benchmark reproducibility gate |

---

## What actually works

Everything below is implemented and exercised against real PostgreSQL and a real
Elasticsearch 8.19.3 node. There are no stub endpoints in the implemented paths.

| Subsystem | Status | Evidence |
|---|---|---|
| Run recording with idempotent events + redaction | Working | `tests/unit/test_run_recorder.py`, `tests/integration/test_security.py` |
| PostgreSQL source of truth (Postgres 17, Alembic) | Working | `migrations/versions/`, `tests/integration/test_postgres.py` |
| Elasticsearch BM25 + dense vector + hybrid RRF | Working | `tests/integration/test_elasticsearch.py` |
| Elasticsearch retrieval depth (identifier analyzer, pre-filtered kNN, app/native RRF, int8) | Implemented and measured on a 10k-document v3 corpus; opt-in features stay off by default where the data did not justify them | `scripts/retrieval_depth.py`, `docs/evidence/phase4-retrieval-depth.md`, `tests/unit/test_es_queries.py` |
| 100k-document index/query scale harness | Working; deterministic and live-nomic measurements recorded | `scripts/scale_test.py`, `docs/evidence/phase4-scale-100k.md` |
| Grounded Q&A with verified citations (`POST /api/v1/ask`) | Working; extractive path in CI, optional LLM path tested with a scripted provider, plus a 40-question mini-eval with deterministic and live `qwen3.5-4b` evidence | `tests/unit/test_ask.py`, `tests/unit/test_ask_eval.py`, `tests/integration/test_ask.py`, `scripts/ask_eval.py`, `docs/evidence/phase4-ask-mini-eval*.md` |
| Lexical reranker + opt-in local CrossEncoder (`hybrid_rerank`) | Working; cross-encoder latency measured, quality comparison not yet meaningful | `tests/unit/test_reranker.py`, `tests/integration/test_cross_encoder_live.py`, `docs/evidence/phase1-rerank-benchmark/` |
| Outcome-aware selection with rejection reasons | Working | `tests/unit/test_recall_policy.py` |
| RepairCompiler (failure window -> typed candidate) | Working | `tests/unit/test_repair_compiler.py` |
| Static + executing mock-sandbox validators | Working | `tests/unit/test_repair_compiler.py` |
| Promotion gate (no bypass path) | Working | `tests/unit/test_repair_compiler.py`, `tests/integration/test_api.py` |
| MemoryShift drift detection + revalidation queue | Working | `tests/integration/test_postgres.py`, demo step 8 |
| Benchmark v3 (memory-required tiers, silent drift, 2,000 decoys, field-scoped compatibility) | Implemented; held-out deterministic matrix and dedicated integration smoke pass; live-model run recorded separately | `tests/unit/test_benchmark_v3.py`, `tests/integration/test_benchmark_v3.py`, `docs/benchmark-v3.md`, `docs/results-v3.md`, `docs/evidence/phase3b-v3-deterministic-test.md`, `docs/evidence/phase3b-v3-live-test.md` |
| Benchmark v2 (10 failure families, 120 tasks, family-held-out splits) | Legacy ceiling/regression suite; both dev and held-out test run with real nomic embeddings and the live model | `tests/unit/test_benchmark_v2.py`, `tests/integration/test_benchmark_v2.py`, `docs/benchmark-v2.md`, `docs/evidence/phase2-summary.md`, `docs/evidence/phase3-llm-test-core/summary.md` |
| Benchmark with card + chunk ablations and safety audit | Working (chunk ablation still degenerate on current data) | `recap eval run --suite v2`, `docs/results/2026-10-20-dev.md` |
| Transactional outbox + retry worker | Working | `tests/unit/test_outbox_worker.py`, `recap search drain` |
| Explicit memory versions + supersession | Working | `tests/unit/test_memory_versions.py`, `tests/e2e/test_lifecycle.py` |
| Run-level memory snapshot pinning | Working | `tests/integration/test_memory_snapshots.py` |
| Deterministic event-aware chunker (no downloads, fail-closed redaction) | Working | `tests/unit/test_chunking.py` |
| Chunk persistence, indexing, lifecycle, `reconcile --chunks` | Working | `tests/integration/test_chunks.py`, `migrations/versions/0004`-`0008` |
| Chunk-level retrieval (`*_chunk` modes) with supporting evidence | Working | `tests/integration/test_chunks.py`, `recap eval run` |
| Content-addressed Postgres embedding cache + usage reporting | Working | `tests/unit/test_embedding_cache.py`, `recap embedding-cache`, `recap eval run` |
| Versioned index rebuild + atomic alias swap (`search reindex`) | Working | `tests/integration/test_reindex.py`, `recap search reindex` |
| API-first chat/embedding providers (OpenAI-compatible) | Implemented; streaming, per-phase timeouts, bounded retries, and live LM Studio checks (`gemma-4-e4b-uncensored-hauhaucs-aggressive` + `nomic-embed-text-v1.5`); not in CI | `tests/unit/test_providers.py`, opt-in `tests/integration/test_live_providers.py`, `docs/evidence/phase2-inference-reliability.md` |
| LLM repair-candidate generation with typed validation | Implemented and unit-tested; live acceptance now passes locally (4/4 schema, static, and sandbox) with the exact configuration in the evidence page; deterministic generator remains default | `tests/unit/test_llm_generator.py`, `docs/evidence/phase2-repair-generator-gemma-final.json`, `docs/evidence/phase2-inference-reliability.md` |
| Real LLM tool-using agent (`recap agent run`) | Implemented; live v1/v2 runs pass locally with gemma, not in CI | `tests/unit/test_llm_agent.py`, `tests/integration/test_llm_agent_recording.py`, `docs/evidence/live-runs/` |
| End-to-end demo (10 steps) | Working | `recap demo run` |

## Known limits

The project is deliberately explicit about what is not built. The capability table above,
the evidence under [`docs/evidence/`](docs/evidence/), and the limitations below are the
public record. A local implementation log is intentionally not published.

- The default embedding provider is a deterministic **hashing lexical embedder**, not a
  neural model. Both benchmark splits were also run with real
  `text-embedding-nomic-embed-text-v1.5` (768-d) through LM Studio; deterministic and
  live-model results are labelled separately.
- The demo and the default benchmark use a **scripted reference agent** for reproducibility.
  The opt-in benchmark mode (`--agent llm`) uses the configured tool-using model. The
  configured-LLM agent completed live v1/v2 runs with
  `gemma-4-e4b-uncensored-hauhaucs-aggressive`, including the `response_field` and
  `idempotency_key` families; the opt-in live suite passes locally
  (`docs/evidence/phase1-live-provider-tests-gemma.txt`), but still skips in CI.
- Benchmark v3 is the current memory-required suite. Its deterministic headline is not a
  live-model result. The separate live v3 run passes the gate on the recorded aggregate
  (`no_memory` 6/24), but all six are `novel_failure` controls; on the 18 selectable tasks it
  is 0/18. `outcome_aware` selects the correct memory on 18/18 selectable live tasks yet the
  small model completes only 8, so the live result measures execution as much as selection.
  It is one model, one seed, one machine: [`docs/results-v3.md`](docs/results-v3.md).
- Benchmark v2 dev and held-out test are both measured with the live model. Both are a
  **ceiling** for this model: `no_memory` succeeds on every task, so task success does not
  distinguish memory policies. `raw_trajectory_rag` and multiple paired seeds are still
  missing; do not infer them from the reports.
- **Live LLM repair generation now passes a small local acceptance run, but is not a general
  reliability claim.** With `gemma-4-e4b-uncensored-hauhaucs-aggressive`, streaming, the
  timeouts listed in `configs/default.yaml`, and two bounded format attempts, 4/4 runs produced
  schema-valid candidates that passed static and mock-sandbox validation. That is four runs on
  one model/machine; the deterministic repair generator remains the default. The Phase 1
  negative result is kept in `docs/evidence/phase1-summary.md`, and the superseding evidence is
  in `docs/evidence/phase2-inference-reliability.md`.
- The local streaming endpoint did not report completion-token usage during the doctor probes,
  so `recap doctor --live` reports `tokens_per_second=null` rather than estimating it. The
  recommended read timeouts are conservative heuristics; configure them from observed runs.
- The **CrossEncoder reranker** is local-only and opt-in. Its real benchmark run records
  latency and provenance, but the two-task dev split cannot show a quality improvement over
  lexical or bi-encoder retrieval.
- The **Ask mini-eval is a seeded 40-question contract test, not an open-domain QA benchmark.**
  Ten memories and ten runs are created with unique token/marker pairs, so the 30 answerable
  questions are answerable by construction; the 10 unanswerable controls use absent tokens.
  The deterministic extractive path and the live `qwen3.5-4b` run both met status, citation,
  and answer-token checks with zero quote failures. The live run is one model/machine, used
  temperature 0, `json_schema`, a 512-token output cap, and 120 s read/overall timeouts; its
  median latency was 7.15 s. It also returned duplicate citations for the same source in 10
  of 30 answerable cases. Set-based precision/recall is unaffected, but the duplicate links
  are an output-quality limitation.
- **Retrieval depth was measured on a 10,005-document synthetic v3 corpus, not production
  traffic.** With 256-d hashing embeddings and three repeats: application RRF reached
  recall@3 `0.556` / recall@10 `1.000` / MRR `0.472` at p50 `130 ms`; native ES RRF failed
  closed with a Basic-license `403`; the identifier analyzer left natural-language recall
  unchanged (`0.818`) but raised identifier-probe MRR from `0.933` to `1.000`; kNN
  `num_candidates=500` gave `0.424` recall@3 versus `0.394` at 200 with flat search latency;
  and `int8_hnsw` fell to `0.212` recall@3 while the index grew from `21.7 MB` to `24.2 MB`.
  `num_candidates=200` remains the default because it is the value used in the 100k latency
  run; changing it needs a 100k sweep. See
  [`docs/evidence/phase4-retrieval-depth.md`](docs/evidence/phase4-retrieval-depth.md).
- The Postgres **embedding cache** is implemented and keyed by the exact fingerprint; it
  caches vectors, not query results, so a hit does not bypass retrieval or policy checks.
  Cache hits, misses, provider batches, latency, and provider-reported prompt tokens appear in
  the benchmark JSON/Markdown; missing provider usage is reported as `null`. The cache has no
  automatic eviction yet; `last_accessed_at`/`hit_count` support manual pruning, and
  `recap embedding-cache --clear` is always safe.
- The chunk token counter is a download-free approximation, not a model tokenizer.
- The current dev split has no same-environment distractor set, so the `naive_fixed` vs
  `event_aware` chunk ablation is **degenerate**. It needs same-environment noise and
  near-duplicate memories before it can support a claim.
- The stack is single-node and local-first. Elasticsearch security is disabled in the local
  Compose file. The API supports an optional shared `RECAP_API_KEY` and CORS allowlist, but
  there is no multi-tenant identity, RBAC, key rotation, or rate limiting. A shared key is not
  a tenant boundary; do not expose the API to an untrusted network.
- Deterministic benchmark numbers measure **pipeline and safety behaviour**; live-model runs
  measure one configured model in a deterministic mock environment. Neither is a general LLM
  capability claim. Reports and `recap providers` state which mode produced a result.

## Results snapshot (benchmark v3 held-out test)

The current suite asks a narrower question than v2: can the policy refuse a memory that is
relevant but unsafe, and act when a field-scoped strategy is still compatible? These are
measured numbers, not projections. Deterministic reference agent, 24 held-out Tier C tasks,
2,000 synthetic decoys, seed 1, deterministic 256-d hashing embedder, live PostgreSQL and
Elasticsearch. **Not a live-LLM result.**

| Baseline | Task success | Selectable tasks (18) | Harmful transfer | Incompatible selection | False abstention |
|---|---:|---:|---:|---:|---:|
| `no_memory` | 0.0% | 0/18 | 0.0% | 0.0% | 0.0% |
| `bm25` | 41.7% | 10/18 | 58.3% | 29.2% | 0.0% |
| `outcome_aware` | **62.5%** | **15/18** | **12.5%** | **0.0%** | **0.0%** |

Tier C `no_memory` acceptance gate: **0/24**, threshold ≤ 50%, **PASS**. The other six failures
for `outcome_aware` are the `novel_failure` controls, where it correctly selected nothing;
false abstention is 0% and abstention precision/recall are 1.0. Design, CIs, and retained
failures: [`docs/results-v3.md`](docs/results-v3.md) and
[`docs/evidence/phase3b-v3-deterministic-test.md`](docs/evidence/phase3b-v3-deterministic-test.md).

A separate live-model run on the same held-out tasks uses
`gemma-4-e4b-uncensored-hauhaucs-aggressive`, nomic 768-d embeddings, streaming, seed 1,
and a 600 s read / 900 s overall timeout:

| Live baseline | Task success (24) | Selectable tasks (18) | Harmful transfer | Incompatible selection |
|---|---:|---:|---:|---:|
| `no_memory` | 25.0% (6/24) | **0/18** | 0.0% | 0.0% |
| `bm25` | 41.7% (10/24) | 4/18 | 58.3% | 29.2% |
| `outcome_aware` | 58.3% (14/24) | **8/18** | 41.7% | **0.0%** |

The six `no_memory` successes are all `novel_failure` negative controls; it succeeds on zero
tasks that require a memory. The policy selected the correct gold memory on 18/18 selectable
tasks, but the small model failed to execute 10 of them, so the 41.7% “harmful transfer” is
mostly **execution failure after correct selection**, not wrong-memory selection. This is
reported, not hidden: [`docs/evidence/phase3b-v3-live-test.md`](docs/evidence/phase3b-v3-live-test.md).

### Ask mini-eval (seeded citation-contract fixture)

Forty questions over ten memories and ten failed runs: 30 answerable from one gold source and
10 unanswerable controls. Live PostgreSQL/Elasticsearch, hashing 256-d embeddings, one source
per question, two modes. The fixture is answerable by construction; this validates retrieval
plumbing and the citation contract, not open-domain recall.

| Mode | Answerable status | Unanswerable status | Citation P / R | Answer tokens | Quote failures | Duplicate citations | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Extractive (deterministic) | 30/30 | 10/10 | 1.000 / 1.000 | 30/30 | 0 | 0 | 141.7 ms |
| `qwen3.5-4b` (live) | 30/30 | 10/10 | 1.000 / 1.000 | 30/30 | 0 | 10/30 | 7152.5 ms |

Live model evidence is one local run: LM Studio `qwen3.5-4b`, temperature 0, `json_schema`,
512 max tokens, 120 s read/overall timeout. Per-question evidence:
[`docs/evidence/phase4-ask-mini-eval.md`](docs/evidence/phase4-ask-mini-eval.md) and
[`docs/evidence/phase4-ask-mini-eval-live-qwen.md`](docs/evidence/phase4-ask-mini-eval-live-qwen.md).

### Retrieval depth (Phase 4.2)

Measured on a 10,005-document v3 corpus with 256-d hashing embeddings, 3 repeats, live
Elasticsearch 8.19.3 Basic. This is a synthetic depth harness, not production traffic.

| Experiment | Measured result |
|---|---|
| Application RRF | recall@3 `0.556`, recall@10 `1.000`, MRR `0.472`, p50 `130 ms` |
| Native ES RRF | not usable on Basic: license `403`; flag fails closed |
| Identifier analyzer | natural-language recall unchanged at `0.818`; identifier-probe MRR `0.933 -> 1.000` |
| kNN `num_candidates` | `200`: recall@3 `0.394`; `500`: `0.424`; p50 ~`50 ms` throughout |
| float32 `hnsw` vs `int8_hnsw` | recall@3 `0.404` vs `0.212`; index `21.7 MB` vs `24.2 MB` |

The int8 index is larger because Elasticsearch retains raw float vectors for rescoring.
`app_rrf`, `identifier_analysis=false`, `knn_num_candidates=200`, and
`dense_vector_quantization=none` remain the defaults; each opt-in path requires a new index
generation. Full ranges and decisions:
[`docs/evidence/phase4-retrieval-depth.md`](docs/evidence/phase4-retrieval-depth.md).

### Legacy benchmark v2 dev, real nomic embeddings

These are measured numbers from `recap eval run --suite v2 --split dev`, not projections.
They use 24 tasks, the scripted reference agent, and real
`text-embedding-nomic-embed-text-v1.5` 768-d embeddings through LM Studio. They remain useful
as a ceiling/regression suite, not as evidence that memory is required.

| Baseline | recall@3 | MRR | task success | harmful transfer | incompatible selection |
|---|---:|---:|---:|---:|---:|
| `no_memory` / `retry_only` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bm25` / `vector` / `hybrid_rrf` | 1.000 | 0.750 | 0.500 | 0.500 | 1.000 |
| `hybrid_rrf_version_filter` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `outcome_aware` | 1.000 | 0.750 | 1.000 | **0.000** | **0.000** |
| `*_naive_chunk` / `*_chunk` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

On this split the fair version filter also fixes stale-version selection, so the full policy
does **not** show a success advantage over it. The policy matches its success while rejecting
wrong-family and bad-track-record distractors; its MRR is lower (0.750 vs 1.000). The chunk
comparison is **degenerate** because there is no same-environment distractor set. Full matrix,
CIs, and caveats: [`docs/evidence/phase2-summary.md`](docs/evidence/phase2-summary.md) and
[`docs/benchmark-v2.md`](docs/benchmark-v2.md).

### Legacy v2 live-model check

A separate v2 run uses the live LM Studio model `gemma-4-e4b-uncensored-hauhaucs-aggressive`
(temperature 0, prompt `llm-tool-agent-v1`, six-tool budget) on the same 24 dev tasks:

| Baseline | recall@3 | MRR | task success | harmful transfer | incompatible selection | input tokens |
|---|---:|---:|---:|---:|---:|---:|
| `no_memory` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 21,282 |
| `bm25` | 1.000 | 0.750 | 1.000 | 0.000 | 1.000 | 54,556 |
| `hybrid_rrf_version_filter` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 50,863 |
| `outcome_aware` | 1.000 | 0.750 | 1.000 | 0.000 | 0.000 | 50,073 |

The live model solved every dev task with no memory. That is a **ceiling effect**, not evidence
that memory is useless: `bm25` selected an incompatible memory on 24/24 tasks and still
succeeded, because the model verified against the live schema. The retrieval metrics expose the
bad ranking; task success does not.

The held-out test families (`required_parameter`, `enum_change`, `date_format`) show the same
ceiling: `no_memory` succeeded on 36/36 tasks. `bm25` again selected incompatible memories on
36/36 tasks with zero harmful transfer; `hybrid_rrf_version_filter` ranked at MRR 1.000 versus
0.750 for `outcome_aware`, with identical task success and tool calls. These runs are one model,
one seed, and four of sixteen baselines. Details and limits:
[`docs/evidence/phase3-llm-dev-core/summary.md`](docs/evidence/phase3-llm-dev-core/summary.md),
[`docs/evidence/phase3-llm-test-core/summary.md`](docs/evidence/phase3-llm-test-core/summary.md),
[`docs/results-llm.md`](docs/results-llm.md), and
[`docs/results-llm-test.md`](docs/results-llm-test.md).

## Documentation

| If you want to... | Read |
|---|---|
| Understand the system | [`docs/architecture.md`](docs/architecture.md), [`docs/data-model.md`](docs/data-model.md), [`docs/retrieval.md`](docs/retrieval.md) |
| Run and operate it | [`docs/operations.md`](docs/operations.md), [Quickstart](#quickstart) |
| Understand repair and safety | [`docs/repair-validation.md`](docs/repair-validation.md), [`docs/threat-model.md`](docs/threat-model.md) |
| Ask questions over recorded evidence | [`docs/ask.md`](docs/ask.md) (`POST /api/v1/ask`, verified citations, 40-question mini-eval) |
| Evaluate it | [`docs/benchmark-v3.md`](docs/benchmark-v3.md), [`docs/results-v3.md`](docs/results-v3.md), [`docs/evidence/phase4-retrieval-depth.md`](docs/evidence/phase4-retrieval-depth.md), [`docs/evidence/phase4-ask-mini-eval.md`](docs/evidence/phase4-ask-mini-eval.md), [`docs/benchmark-v2.md`](docs/benchmark-v2.md), [`docs/evidence/phase2-summary.md`](docs/evidence/phase2-summary.md) |
| Explore the recorded runs | [`web/README.md`](web/README.md), [`docs/evidence/phase3b-v3-live-test.md`](docs/evidence/phase3b-v3-live-test.md) |
| Connect a model | [`docs/providers.md`](docs/providers.md) |
| Contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`CHANGELOG.md`](CHANGELOG.md) |
| Repository metadata | [`docs/github-repository.md`](docs/github-repository.md) |

The full index is [`docs/README.md`](docs/README.md).

---

## Quickstart

### 1. Services

```bash
docker compose up -d postgres elasticsearch
```

- PostgreSQL 17 on `localhost:5433` (`recap` / `recap` / `recap`)
- Elasticsearch 8.19.3 on `localhost:9200`, security disabled for local use only
- Optional packaged API + outbox worker: `docker compose up -d api worker` (API on `:8000`)

If another RECAP checkout is already running on this machine, stop that stack first
(`docker compose down` in the other checkout). The Compose file uses fixed container names
and host ports, so two checkouts cannot run simultaneously without overriding them.

The local services use `restart: "no"` and are not started by Docker/Windows at login. Start
them explicitly for integration tests, then stop them to release RAM:

```bash
docker compose up -d postgres elasticsearch
docker compose stop postgres elasticsearch
```

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

For a shared deployment, set `RECAP_API_KEY` and `RECAP_CORS_ORIGINS` and send
`X-API-Key` on application routes; `/health/live` and `/health/ready` stay public. This is a
single shared key, not multi-tenant auth. See [`docs/operations.md`](docs/operations.md#optional-static-api-key).

### 4. Configure real models (API-first)

RECAP never downloads model weights. Point it at an already-running OpenAI-compatible
endpoint:

PowerShell:

```powershell
$env:EMBEDDING_PROVIDER = "openai_compatible"
$env:EMBEDDING_BASE_URL = "https://api.example.com/v1"
$env:EMBEDDING_API_KEY = "..."
$env:EMBEDDING_MODEL = "text-embedding-3-small"
$env:EMBEDDING_DIMENSIONS = "1536"

$env:LLM_PROVIDER = "openai_compatible"
$env:LLM_BASE_URL = "https://api.example.com/v1"
$env:LLM_API_KEY = "..."
$env:LLM_MODEL = "gpt-4.1-mini"

recap providers          # offline capability report, no secrets
recap doctor --live      # probes configured endpoints, one small chat + embedding
recap agent run --environment-version 2 --workspace live-agent
```

Bash / zsh:

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

recap providers
recap doctor --live
recap agent run --environment-version 2 --workspace live-agent
```

`recap agent run` drives the configured chat model through a bounded tool-calling loop
against the mock API. The model selects `list_records` arguments; RECAP validates the tool
name and arguments, executes only that local tool, and computes success from the records
actually collected. It refuses to run when no `llm` role is configured and never falls back
to the deterministic agent. The recorded run includes provider, model, prompt version, token
usage, and every tool call/failure. Add `--transcript path.json` to write a redacted
run/event transcript.

### 5. Elasticsearch index generations and reranking

Changing an embedding model or vector dimension requires a new index generation. Do not
edit the old mapping in place:

```bash
recap search reindex --to-generation v2
# inspect v2, then optionally remove the old concrete index
recap search reindex --to-generation v3 --delete-old
```

The reindex command re-embeds active memories from Postgres into a fresh concrete index,
verifies its document count, swaps the read and write aliases in one atomic request, and
updates the Postgres generation provenance. The old index stays behind the read alias until
the swap, is retained by default for rollback, and is deleted only with `--delete-old` after
the swap and database update succeed. See
[`docs/operations.md`](docs/operations.md#rebuild-a-memory-index-generation-without-downtime).

The optional local CrossEncoder reranker is also opt-in and never downloads weights:

```bash
pip install -e ".[rerank]"
$env:RERANK_PROVIDER = "cross_encoder"
$env:RERANK_MODEL_PATH = "models/ms-marco-MiniLM-L-6-v2"
recap memory search "cursor expired" --mode hybrid_rerank
```

`recall.reranker=lexical` remains the default. The cross-encoder path requires an existing
local model directory and `local_files_only=True`; every decision records the reranker name,
model path, and per-stage rerank latency.

`hashing`/`fake` embeddings remain available for offline development and unit tests, but they
are reported as non-semantic. Production rejects them unless `EMBEDDING_ALLOW_DEGRADED=true`
is explicitly set. The embedding dimension must match the Elasticsearch mapping; changing
models requires a new index generation. Details: [`docs/providers.md`](docs/providers.md).

### 6. Run the demo

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

### 7. Benchmark

```bash
# current memory-required suite (deterministic reference agent)
recap eval run --suite v3 --split test --agent reference --fixture-count 2000 --seed 1

# legacy ceiling/regression suite
recap eval run --suite v2 --split dev
recap eval run --suite v2 --split test
```

Reports land in `benchmark/reports/`. See [`docs/benchmark-v3.md`](docs/benchmark-v3.md),
[`docs/results-v3.md`](docs/results-v3.md), and
[`docs/evaluation-methodology.md`](docs/evaluation-methodology.md) for what the numbers do
and do not mean.

Reproducibility check (the same gate CI runs):

```bash
python scripts/run_benchmark_repro.py
```

### 8. Replay web app

```bash
python -m http.server 8000 --directory web
```

Open <http://localhost:8000>. The app is a no-build static replay: it reads committed JSON
under `web/replay/` and needs no database, model, or backend. A dataset selector switches
between the deterministic test/dev artifacts and the live-model artifact, and the provenance
bar shows agent mode, exact model, embedder, reranker, dataset hash, and timeout. Re-export
from reports with `python scripts/export_replay.py --input <report.json> --out web/replay`.
See [`web/README.md`](web/README.md).

### 9. HTTP API

```bash
recap serve --port 8000
```

`GET /health/live`, `GET /health/ready`, `GET /api/v1/diagnostics`. Full route list in
[`docs/operations.md`](docs/operations.md).

Grounded Q&A is available at `POST /api/v1/ask`. The default extractive mode needs no model;
`mode=llm` uses the configured agent role. Every citation is verified against the supplied
source text, and an unverifiable answer becomes `insufficient_evidence`:

```bash
curl -s http://localhost:8000/api/v1/ask \
  -H 'content-type: application/json' \
  -d '{"workspace_id":"demo","question":"Which memory fixes INVALID_CURSOR?"}'
```

See [`docs/ask.md`](docs/ask.md) for the contract, limits, and tests.

---

## CLI surface

```
recap version
recap health
recap providers
recap doctor [--live]
recap serve

recap db migrate | init
recap search init | health | reconcile [--chunks] | drain | worker | outbox
recap memory search | show | compatibility
recap chunk build | show | count
recap repair compile | validate | promote
recap agent run --environment-version 1
recap demo seed | run
recap eval run --suite v3 --split test --agent reference
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

```mermaid
flowchart LR
    A[Run + events] --> B[Failure window]
    B --> C[Typed repair candidate]
    C --> D{Validators execute it}
    D -- failed --> C
    D -- passed --> E[Promotion gate]
    E --> F[(Postgres<br/>source of truth)]
    F --> G[Event-aware chunks]
    G --> H[(Elasticsearch<br/>card + chunk indices)]
    H --> I[BM25 + vector + RRF]
    I --> J[Hard filters + applicability<br/>+ compatibility + utility]
    J --> K{Safe?}
    K -- yes --> L[Selected memory<br/>with evidence]
    K -- no --> M[no_safe_memory<br/>+ rejection reasons]
    F -. outcomes .-> N[MemoryShift<br/>suspect / revalidate]
```

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
  ask/               grounded retrieval, answer synthesis, citation verification
  api/               FastAPI routes
  demo.py            the 10-step end-to-end flow
migrations/          Alembic
configs/             default.yaml
docs/                design + operations documentation
```

---

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the local setup,
test matrix, migration rules, and pull-request checklist. Participation follows the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report vulnerabilities privately using the
process in [`SECURITY.md`](SECURITY.md), not a public issue. Release notes are in
[`CHANGELOG.md`](CHANGELOG.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
