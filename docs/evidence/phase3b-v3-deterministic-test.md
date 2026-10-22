# Evidence — benchmark v3 deterministic held-out test

**Date:** 2026-10-22
**Status:** complete deterministic run; live-model run is a separate artifact
**Scope:** pipeline, retrieval, selection, and safety behavior — **not** an LLM capability
claim.

## Exact environment

| Item | Value |
|---|---|
| Suite / split | `v3` / `test` |
| Agent | deterministic reference agent, `benchmark-reference-agent`, mode `deterministic` |
| Agent/prompt version | `reference-agent-v1` / `deterministic-policy-v1` |
| Seed | `1` |
| Tool budget | `6` |
| Tasks | 24 held-out Tier C tasks |
| Store | 3 synthetic gold fixtures + 2,000 synthetic decoys = 2,003 memories |
| Provenance | `real_gold_synthetic_decoys` |
| Postgres | PostgreSQL 17, local `recap_test` |
| Elasticsearch | 8.19.3, local single node |
| Embedding | deterministic `recap-hashing-embedding-v1`, 256-d; cache hit rate 1.0 on this run |
| Reranker | local `cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded with `local_files_only=True` |
| Dataset hash | `sha256:8438ccf598486dbdb44f8b29593187720e769b1389666fc1ff6dc568716697eb` |
| Report timestamp | `2026-10-22T06:21:24Z` |

The deterministic agent has no provider, model timeout, or token usage. Those fields are
`null`, not invented. A live-model run must list its exact model, temperature, max tokens,
and timeouts separately.

## Command

```powershell
$env:RERANK_PROVIDER = "cross_encoder"
$env:RERANK_MODEL_PATH = (Resolve-Path "models\ms-marco-MiniLM-L-6-v2").Path
recap eval run --suite v3 --split test --agent reference `
  --fixture-count 2000 --seed 1 --no-resume --output .tmp\v3-test-final
```

The same run is reproducible via the suite integration test:

```powershell
$env:RECAP_RUN_INTEGRATION = "1"
pytest tests/integration/test_benchmark_v3.py -q --run-integration
```

## Result matrix

Scenario-clustered 95% bootstrap intervals are in the JSON/Markdown report. The table below
keeps the point estimates; do not treat overlapping intervals as a significance test.

| Baseline | Success | 95% CI | recall@3 | MRR | nDCG@3 | harmful transfer | incompatible selection | false abstention | mean tool calls |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `no_memory` | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.75 |
| `bm25` | 0.417 | [0.250, 0.625] | 0.750 | 0.576 | 0.622 | 0.583 | 0.292 | 0.000 | 4.17 |
| `vector` | 0.208 | [0.083, 0.375] | 0.292 | 0.275 | 0.261 | 0.417 | 0.750 | 0.000 | 3.08 |
| `hybrid_rrf` | 0.292 | [0.125, 0.458] | 0.625 | 0.451 | 0.475 | 0.583 | 0.583 | 0.000 | 3.46 |
| `hybrid_rerank` | 0.292 | [0.125, 0.458] | 0.625 | 0.443 | 0.475 | 0.542 | 0.625 | 0.000 | 3.25 |
| `hybrid_rrf_version_filter` | 0.125 | [0.000, 0.292] | 0.750 | 0.625 | 0.658 | 0.000 | 0.125 | 0.667 | 2.88 |
| `fingerprint_filter_oracle` | 0.125 | [0.000, 0.292] | 0.750 | 0.625 | 0.658 | 0.000 | 0.125 | 0.667 | 2.88 |
| `outcome_aware` | **0.625** | [0.417, 0.833] | 0.458 | 0.324 | 0.319 | **0.125** | **0.000** | **0.000** | 3.50 |

**Acceptance gate:** Tier C `no_memory` = `0/24 (0.000)`, threshold ≤ `0.50`, **PASS**.

**Non-discrimination warning:** `fingerprint_filter_oracle` and
`hybrid_rrf_version_filter` produced identical metrics on this split. On this test data the
environment-id filter and whole-fingerprint filter happen to reject the same candidates, so
the oracle adds no discrimination here. The report emits a warning rather than hiding it.

## Failure inspection

`outcome_aware` fails nine tasks; all nine are retained in the report:

| Failure kind | Count | What happened |
|---|---:|---|
| Harmful transfer | 3 | `silent_enum` ×2 and `silent_account` ×1 selected the field-strategy gold. The selected memory was compatible with its own declared paths and passed the policy, but the verified result was wrong. |
| Correct `no_safe_memory` | 6 | `novel_failure` negative controls. No memory was selected; the policy failed closed. These are counted as task failures by design, not as successful completions. |

There are **zero** false abstentions on tasks with a relevant memory, and abstention
precision/recall are both `1.0`. This matters: the policy is not winning by refusing to act.

The 3 harmful transfers are the honest remaining gap. They are not “safe failures”; they are
wrong actions. A future policy needs stronger action/effect compatibility for enum and
account strategies, or a post-action verification path that can detect the silent mismatch
before the run is scored.

## Interpretation

Supported:

- relevance-only retrieval is unsafe on this suite: BM25/hybrid harmful transfer is ~54–58%,
  vector incompatible selection is 75%;
- the outcome-aware policy eliminates independent incompatible selection and false
  abstention on this split;
- exact whole-fingerprint filtering is too strict (66.7% false abstention), which is why
  field-scoped path compatibility is used;
- the suite is genuinely memory-required (`no_memory` 0/24).

Not supported:

- “RECAP solves memory safety”; it does not; 12.5% harmful transfer remains;
- “RECAP ranks better than BM25/oracle”; it does not on recall@3/MRR;
- live-LLM performance, production traffic performance, multi-tenant security, or
  Elasticsearch scale;
- statistical significance across models/runs. This is one seed on one machine.

## Artifacts

The full per-task JSON/CSV/Markdown report is generated under `benchmark/reports/` and is
ignored by git by repository policy. The curated design and result narrative are in
[`../benchmark-v3.md`](../benchmark-v3.md) and [`../results-v3.md`](../results-v3.md). The
integration smoke test is `tests/integration/test_benchmark_v3.py`.
