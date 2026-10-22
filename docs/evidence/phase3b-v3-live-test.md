# Evidence — benchmark v3 live-model held-out test

**Date:** 2026-10-22
**Status:** complete live-model run against real PostgreSQL and Elasticsearch
**Scope:** one small local model, one seed, one machine. This is **not** a general
LLM-capability or production-traffic claim.

## Exact environment

| Item | Value |
|---|---|
| Suite / split | `v3` / held-out `test` |
| Agent | `llm_verifying` (`benchmark-llm-agent`), prompt `llm-tool-agent-v1` |
| Model | `gemma-4-e4b-uncensored-hauhaucs-aggressive` |
| Provider | OpenAI-compatible LM Studio, `http://localhost:1234/v1` |
| Temperature | `0.0` |
| Seed | `1` |
| Tool budget | `6` |
| Streaming | enabled |
| Max output tokens | `2048` |
| Read timeout | `600 s` |
| Overall timeout | `900 s` |
| No-progress timeout | `180 s` |
| Max provider retries | `2` |
| Max concurrency | `1` |
| Tasks | 24 held-out Tier C tasks |
| Baselines | `no_memory`, `bm25`, `outcome_aware` |
| Store | 3 gold fixtures + 2,000 synthetic decoys = 2,003 memories |
| Provenance | `real_gold_synthetic_decoys` |
| Embedder | `text-embedding-nomic-embed-text-v1.5`, 768-d, OpenAI-compatible |
| Embedding fingerprint | `sha256:376c38156f69e20800da276053b1a6557b13d6c376133f34f5be618fba3c6812` |
| Reranker | local `cross-encoder/ms-marco-MiniLM-L-6-v2`, `local_files_only=True` |
| Postgres | PostgreSQL 17, `recap_test` |
| Elasticsearch | 8.19.3, single node, `recap-v3-live` prefix and dedicated aliases |
| Dataset hash | `sha256:8438ccf598486dbdb44f8b29593187720e769b1389666fc1ff6dc568716697eb` |
| Provider timeouts / errors | `0 / 0`; `24/24` runs completed |
| Report timestamp | `2026-10-22T06:52:18.899081Z` |

The local endpoint returned `0` for input/output token fields. Treat usage as **missing**, not
as zero cost; no token or cost claim is made from this run.

## Command

The long environment block is required so the live run cannot silently reuse the deterministic
256-d index or the default chat timeout. The essential command is:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://recap:recap@localhost:5433/recap_test"
$env:ELASTICSEARCH_URL = "http://localhost:9200"
$env:ELASTICSEARCH_INDEX_PREFIX = "recap-v3-live"
$env:ELASTICSEARCH_MEMORIES_READ_ALIAS = "recap-v3-live-memories-read"
$env:ELASTICSEARCH_MEMORIES_WRITE_ALIAS = "recap-v3-live-memories-write"
$env:ELASTICSEARCH_CHUNKS_READ_ALIAS = "recap-v3-live-chunks-read"
$env:ELASTICSEARCH_CHUNKS_WRITE_ALIAS = "recap-v3-live-chunks-write"
$env:EMBEDDING_PROVIDER = "openai_compatible"
$env:EMBEDDING_BASE_URL = "http://localhost:1234/v1"
$env:EMBEDDING_API_KEY = "lm-studio"
$env:EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
$env:EMBEDDING_DIMENSIONS = "768"
$env:EMBEDDING_DOCUMENT_PREFIX = "search_document: "
$env:EMBEDDING_QUERY_PREFIX = "search_query: "
$env:LLM_PROVIDER = "openai_compatible"
$env:LLM_BASE_URL = "http://localhost:1234/v1"
$env:LLM_API_KEY = "lm-studio"
$env:LLM_MODEL = "gemma-4-e4b-uncensored-hauhaucs-aggressive"
$env:LLM_STREAM = "true"
$env:LLM_READ_TIMEOUT_SECONDS = "600"
$env:LLM_OVERALL_TIMEOUT_SECONDS = "900"
$env:LLM_NO_PROGRESS_TIMEOUT_SECONDS = "180"
$env:LLM_MAX_TOKENS = "2048"
$env:RERANK_PROVIDER = "cross_encoder"
$env:RERANK_MODEL_PATH = (Resolve-Path "models\ms-marco-MiniLM-L-6-v2").Path

recap eval run --suite v3 --split test --agent llm_verifying `
  --baselines no_memory,bm25,outcome_aware `
  --fixture-count 2000 --seed 1 --no-resume --output .tmp\live-v3-test
```

## Result matrix

| Baseline | Success (24) | 95% CI | Selectable tasks (18) | recall@3 | MRR | harmful transfer | incompatible selection | false abstention |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `no_memory` | 6/24 (25.0%) | [8.3%, 41.7%] | **0/18** | 0.000 | 0.000 | 0.0% | 0.0% | 0.0% |
| `bm25` | 10/24 (41.7%) | [20.8%, 62.5%] | 4/18 | 0.750 | 0.583 | 58.3% | 29.2% | 0.0% |
| `outcome_aware` | 14/24 (58.3%) | [37.5%, 79.2%] | **8/18** | 0.542 | 0.378 | 41.7% | **0.0%** | **0.0%** |

`no_memory` has **six successes, all six `novel_failure` negative controls**. The model made one
call, selected nothing, and the control scored as success. It succeeded on **zero of the 18
tasks that required selecting a memory**. The aggregate 25.0% is therefore not evidence that
no-memory solves silent drift.

Acceptance gate: Tier C `no_memory` = `6/24 (25.0%)`, threshold ≤ `50.0%`, **PASS** on the
recorded gate. The stricter selectable-task view is `0/18` for `no_memory`.

## Family breakdown

| Family | Tasks | `no_memory` | `bm25` | `outcome_aware` | outcome harmful |
|---|---:|---:|---:|---:|---:|
| `silent_enum` | 6 | 0/6 | 3/6 | **6/6** | 0/6 |
| `silent_account` | 6 | 0/6 | 1/6 | 2/6 | 4/6 |
| `silent_field` | 6 | 0/6 | 0/6 | 0/6 | 6/6 |
| `novel_failure` (negative control) | 6 | 6/6 | 6/6 | 6/6 | 0/6 |

`outcome_aware` selected the correct gold memory on **18/18** selectable tasks (selection
precision `1.0`, selection recall `1.0`, incompatible selection `0`). It still failed to
complete 10 of them. The three `silent_field` and some `silent_account` failures are execution
failures after a correct selection, not evidence that a wrong memory was chosen.

The field-task transcript is the clearest case: the policy selected and marked the correct
field-strategy memory as used, but the model stopped after two tool calls with the claim
“the task is complete and all 6 records have been returned”. The verified record set did not
match, so the run is scored as harmful transfer. The report's `harmful_transfer` field is
outcome-based; it does **not** assert that the memory was semantically wrong. The separate
`selected_gold = true` and `selection_precision = 1.0` fields make that distinction auditable.

## Interpretation

Supported by this run:

- On tasks that require memory, the policy's correct selection rate is 18/18, and it never
  selects an independently incompatible memory.
- Memory improves verified completion over no memory on selectable tasks: 8/18 versus 0/18.
- `outcome_aware` performs better than BM25 on selectable-task success (8/18 vs 4/18), harmful
  transfer (41.7% vs 58.3%), and incompatible selection (0% vs 29.2%).
- The six `novel_failure` controls are handled safely by all three arms, including no-memory.

Not supported:

- “RECAP makes this model solve memory tasks.” It completes 8/18 selectable tasks; 10 fail.
- “The live run matches the deterministic 15/18 selectable-task result.” It does not.
- “Harmful transfer means a wrong memory was selected.” Here it mostly means the small model
  failed to execute a correctly selected strategy. Both facts are required to read the run.
- Any multi-seed, production, scale, or general model-capability conclusion.

## Reproduce and artifacts

Full per-task JSON/CSV/Markdown report:
`.tmp/live-v3-test/benchmark_test_20261022T062346Z.{json,md,csv}` (local; `benchmark/reports/`
is gitignored by policy). Copy into `benchmark/reports/` if a permanent local artifact is
wanted. The static replay export includes this run as a selectable live dataset.

Related documents: [`../benchmark-v3.md`](../benchmark-v3.md),
[`../results-v3.md`](../results-v3.md), and
[`phase3b-v3-deterministic-test.md`](phase3b-v3-deterministic-test.md).
