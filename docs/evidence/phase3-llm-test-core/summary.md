# Phase 3 live-model benchmark: v2 held-out test, four core baselines

This is a **real chat model** run on the held-out test families. It is not the deterministic
reference agent and it is not a mock/unit test.

## Provenance

| Item | Value |
|---|---|
| Report | `benchmark_test_20261021T102715Z.json` / `.md` / `.csv` |
| Generated | `2026-10-21T10:46:26Z` |
| Suite / split | `v2` / `test` |
| Tasks | 36 (12 `required_parameter`, 12 `enum_change`, 12 `date_format`; versions 1 and 2) |
| Baselines | `no_memory`, `bm25`, `hybrid_rrf_version_filter`, `outcome_aware` |
| Rows | 144 |
| Model | `gemma-4-e4b-uncensored-hauhaucs-aggressive` via LM Studio (`openai_compatible`) |
| Prompt | `llm-tool-agent-v1` |
| Temperature / seed / tool budget | `0.0` / `1` / `6` |
| Embeddings | `text-embedding-nomic-embed-text-v1.5`, 768-d, nomic document/query prefixes |
| Vector index | `recap-test-nomic-memories-v1` |
| Dataset hash | `sha256:fce743bc8176368fbd5731f5da1683690ad0e286bca59523b115a745aa6c1a21` |
| Checkpoint | `checkpoint_v2_test_llm_seed1.jsonl` (144 keys) |
| Provider errors | 0 |

Command:

```bash
recap eval run --suite v2 --split test --agent llm --seed 1 \
  --baselines no_memory,bm25,hybrid_rrf_version_filter,outcome_aware \
  --output docs/evidence/phase3-llm-test-core
```

The same LM Studio, nomic embedding, index-generation, and prompt settings were used as in the
dev run. The report records the exact model, prompt version, temperature, seed, tool budget,
embedding fingerprint, and active index.

## Results

| Baseline | Success | 95% CI | recall@3 | MRR | nDCG@3 | Tool calls | Harmful transfer | Incompatible selected | Hard-neg selected | Hard-neg outrank | False abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_memory` | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | 0.000 | 1.50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bm25` | 1.000 | [1.000, 1.000] | 1.000 | 0.750 | 0.815 | 1.86 | 0.000 | 1.000 | 1.000 | 0.500 | 0.000 |
| `hybrid_rrf_version_filter` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `outcome_aware` | 1.000 | [1.000, 1.000] | 1.000 | 0.750 | 0.815 | 1.50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Token and duration cost:

| Baseline | Input tokens | Output tokens | Summed task duration |
|---|---:|---:|---:|
| `no_memory` | 36,108 | 2,466 | 198,421 ms |
| `bm25` | 80,605 | 2,819 | 207,721 ms |
| `hybrid_rrf_version_filter` | 69,641 | 2,405 | 418,822 ms |
| `outcome_aware` | 69,636 | 2,400 | 175,455 ms |

Retrieval latency (see the cache caveat below):

| Baseline | p50 | p95 |
|---|---:|---:|
| `bm25` | 117.2 ms | 134.4 ms |
| `hybrid_rrf_version_filter` | 1,994.8 ms | 5,034.8 ms |
| `outcome_aware` | 161.4 ms | 217.4 ms |

## What this run shows

1. **The held-out test families are also a ceiling for this model.** `no_memory` succeeded on
   all 36 tasks, including `required_parameter`, `enum_change`, and `date_format`. The tool
   schema and error messages contain enough information for the model to adapt without memory.
2. **The safety metrics still separate the baselines.** `bm25` selected incompatible memories
   on every task (72 selections across 36 tasks; `incompatible_selection_rate = 1.000`) and
   still had 0 harmful transfer and 1.000 task success. `hybrid_rrf_version_filter` and
   `outcome_aware` had zero incompatible selections.
3. **Version filtering again had the best retrieval ranking.** MRR/nDCG were 1.000 for
   `hybrid_rrf_version_filter` versus 0.750/0.815 for `outcome_aware`. Task success and tool
   calls were identical. This split still contains no evidence that lets the utility policy
   distinguish a correct same-family memory from a stale one.
4. **Memory again cost more input tokens without a success gain.** Memory baselines used about
   1.9-2.2x the input tokens of `no_memory` and produced no additional successful tasks.
5. **No provider or format failures occurred.** All 144 rows completed.

## Cache and order caveat

The benchmark reuses the content-addressed embedding cache across baselines. The version-filter
baseline ran before `outcome_aware`, and the run recorded 54 embedding-cache misses / 36
provider requests. Its p50 therefore includes uncached query-embedding work that the later
baseline did not pay. The latency table is **not** a clean policy-overhead comparison. Quality
metrics and task outcomes are unaffected by this ordering.

## What this run does not show

- It is one model, one seed, four of sixteen baselines. It is not the full Phase 3 matrix.
- It does not show that memory is useless; it shows that these schema-discovery tasks do not
  require memory for this model.
- It does not measure multiple paired seeds, same-environment noise, abstention controls,
  `raw_trajectory_rag`, the CrossEncoder reranker, or chunked baselines.
- The mock environment is deterministic; the model is not. Temperature 0 reduces but does not
  eliminate provider-request variability.
- It does not make live repair generation production-ready; that remains a separate known gap.

## Consequence for the benchmark

The dev and held-out test splits now agree: an LLM at this capability level can solve the
current failure families from the schema and error feedback alone. Task success is therefore
not a discriminating metric for memory policy on this suite. The next benchmark iteration must
add tasks where the required fact is **not** in the schema or error text, such as a
tenant-specific value, a side-effect rule, or a learned rate-limit sequence. Until then, the
honest headline remains the retrieval/safety comparison, not a memory-driven success win.
