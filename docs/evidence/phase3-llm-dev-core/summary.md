# Phase 3 live-model benchmark: v2 dev, four core baselines

This is a **real chat model** run. It is not the deterministic reference agent and it is not a
mock/unit test.

## Provenance

| Item | Value |
|---|---|
| Report | `benchmark_dev_20261021T100400Z.json` / `.md` / `.csv` |
| Generated | `2026-10-21T10:16:43Z` |
| Suite / split | `v2` / `dev` |
| Tasks | 24 (12 `idempotency_key`, 12 `response_field`; versions 1 and 2) |
| Baselines | `no_memory`, `bm25`, `hybrid_rrf_version_filter`, `outcome_aware` |
| Rows | 96 |
| Model | `gemma-4-e4b-uncensored-hauhaucs-aggressive` via LM Studio (`openai_compatible`) |
| Prompt | `llm-tool-agent-v1` |
| Temperature / seed / tool budget | `0.0` / `1` / `6` |
| Embeddings | `text-embedding-nomic-embed-text-v1.5`, 768-d, nomic document/query prefixes |
| Vector index | `recap-test-nomic-memories-v1` |
| Dataset hash | `sha256:14579e31a18d2193f29f3946482a2942664e019a61abfbec20deb817e5acf539` |
| Checkpoint | `checkpoint_v2_dev_llm_seed1.jsonl` (96 keys) |
| Provider errors | 0 |

Command (environment variables abbreviated to the values that affect the result):

```bash
export EMBEDDING_PROVIDER=openai_compatible
export EMBEDDING_BASE_URL=http://localhost:1234/v1
export EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
export EMBEDDING_DIMENSIONS=768
export EMBEDDING_DOCUMENT_PREFIX='search_document: '
export EMBEDDING_QUERY_PREFIX='search_query: '
export RECAP_VECTOR_DIMENSION=768
export ELASTICSEARCH_INDEX_PREFIX=recap-test-nomic
export RECAP_INDEX_GENERATION=1
export RECAP_CHUNKS_INDEX_GENERATION=2
export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=http://localhost:1234/v1
export LLM_MODEL=gemma-4-e4b-uncensored-hauhaucs-aggressive
export LLM_TEMPERATURE=0
export LLM_TIMEOUT_SECONDS=120

recap eval run --suite v2 --split dev --agent llm --seed 1 \
  --baselines no_memory,bm25,hybrid_rrf_version_filter,outcome_aware \
  --output docs/evidence/phase3-llm-dev-core
```

The run took about 12.7 minutes of wall time. The checkpoint is append-only and keyed by
`(task, baseline, seed)`; the JSON report is the rendered source of truth.

## Results

| Baseline | Success | 95% CI | recall@3 | MRR | nDCG@3 | Tool calls | Harmful transfer | Incompatible selected | Hard-neg selected | Hard-neg outrank | False abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_memory` | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | 0.000 | 1.25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bm25` | 1.000 | [1.000, 1.000] | 1.000 | 0.750 | 0.815 | 1.96 | 0.000 | 1.000 | 1.000 | 0.500 | 0.000 |
| `hybrid_rrf_version_filter` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.75 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `outcome_aware` | 1.000 | [1.000, 1.000] | 1.000 | 0.750 | 0.815 | 1.71 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Token and wall-clock cost:

| Baseline | Input tokens | Output tokens | Summed task duration |
|---|---:|---:|---:|
| `no_memory` | 21,282 | 1,260 | 115,931 ms |
| `bm25` | 54,556 | 1,732 | 132,804 ms |
| `hybrid_rrf_version_filter` | 50,863 | 1,596 | 294,469 ms |
| `outcome_aware` | 50,073 | 1,575 | 119,745 ms |

Retrieval latency (separate from model latency):

| Baseline | p50 | p95 |
|---|---:|---:|
| `bm25` | 116.9 ms | 133.8 ms |
| `hybrid_rrf_version_filter` | 223.4 ms | 5,072.6 ms |
| `outcome_aware` | 158.6 ms | 176.0 ms |

The version-filter p95 includes a single large retrieval outlier; its p50 is the more useful
number for this 24-task split.

## What this run shows

1. **The live model solves the dev split without memory.** `no_memory` succeeded on all 24
   tasks. The dev families (`idempotency_key`, `response_field`) are discoverable from the tool
   schema and error responses, so this split has a **ceiling effect** for this model. Task
   success cannot distinguish memory policies here.
2. **Selection safety and task success are different measurements.** `bm25` selected an
   incompatible/stale memory on every task (24/24) but still scored 1.000 task success and
   0.000 harmful transfer. The model appears to treat retrieved memory as advisory and verify
   against the live schema. The retrieval metrics still correctly expose the bad ranking; the
   success metric does not.
3. **Version filtering produced the cleanest retrieval.** `hybrid_rrf_version_filter` had
   MRR/nDCG of 1.000 and zero incompatible selections. `outcome_aware` matched it on task
   success but ranked the relevant memory lower (MRR 0.750), because this split contains no
   utility evidence that lets the policy separate a correct same-family memory from a stale
   one.
4. **Memory increased prompt cost without a success gain.** Every memory baseline used about
   2.3-2.6x the input tokens of `no_memory`, with no task-success improvement on this split.
   That is a real cost signal, not a fabricated benchmark win.
5. **No provider or format failures occurred.** All 96 rows completed, so the result is not
   dominated by the repair-generation issue seen elsewhere.

## What this run does not show

- It is **one model, one seed, one split, four of sixteen baselines**. It is not the full Phase 3
  matrix.
- It does not show that outcome-aware selection is useless. It shows that the current dev split
  is too easy and lacks memory-required evidence.
- It does not measure the held-out `test` families (`required_parameter`, `enum_change`,
  `date_format`).
- The mock environment is deterministic; the model is not. Temperature 0 reduces but does not
  eliminate provider-request variability.
- It does not measure the local CrossEncoder reranker or chunked baselines.
- It does not establish that repair generation is production-ready; live repair remains
  disabled in this run.

## Next benchmark work

The next run needs evidence that memory is actually required, not just present:

- held-out `test` split with the same model and seeds;
- at least three paired seeds for the core baselines;
- tasks where the schema does not reveal the required adaptation (for example, a side-effect
  rule, a tenant-specific constraint, or a rate-limit sequence that must be learned from prior
  outcomes);
- a same-environment noise control and an abstention control;
- `raw_trajectory_rag` and the chunked baselines under the same protocol.

Until those exist, the honest headline is: **the live model passes this split without memory;
the retrieval policies differ in ranking and cost, not in task success.**
