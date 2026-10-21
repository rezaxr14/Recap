# Benchmark v2

Benchmark v2 is the larger, family-held-out suite that replaces the original six-task
demonstration. It is implemented and has been run on the **dev** split with both the
deterministic hashing embedder and real `text-embedding-nomic-embed-text-v1.5` embeddings
through LM Studio. The **test** split is implemented but has not been run yet.

The measured dev results are in
[`evidence/phase2-summary.md`](evidence/phase2-summary.md). This page documents the design,
the exact commands, and the parts that are still missing so the numbers are not over-read.

## Suite

`build_benchmark_v2_tasks()` in `src/recap/evals/datasets.py` returns 120 deterministic tasks:

| Property | Value |
|---|---|
| Failure families | 10 |
| Tasks | 120 (10 families x 2 versions x 6 paraphrases) |
| Environment versions | `1` and `2`, with distinct tool-schema hashes |
| Splits | family-held-out: `train` 5 families, `dev` 2, `test` 3 |
| Dev families | `idempotency_key`, `response_field` |
| Test families | `required_parameter`, `enum_change`, `date_format` |
| Instruction source | six committed deterministic paraphrases per family |
| Memory creation | real compile -> mock-sandbox validation -> promotion path |

Families:

| Family | Version 1 / 2 difference | Typed recovery |
|---|---|---|
| `invalid_cursor` | cursor is rejected | refresh cursor |
| `renamed_parameter` | `page_token` vs `cursor` | map parameter |
| `rate_limit` | first call is throttled | retry with backoff |
| `timeout` | first call times out | retry with backoff |
| `auth_expiry` | stale vs fresh token | refresh auth token |
| `idempotency_key` | `idempotency_key` vs `request_id` | supply required parameter |
| `response_field` | `items` vs `data` | map response field |
| `required_parameter` | `account` vs `account_id` | supply required parameter |
| `enum_change` | `created` vs `created_at` | map parameter value |
| `date_format` | slash date vs ISO-8601 | normalize timestamp |

## Hard negatives and leakage

For a dev task, the store contains:

- the correct memory for its `(family, version)`;
- a stale memory for the other version of the same family;
- a wrong-family memory;
- a **bad-track-record** distractor from a family held out of the split. It is created through
  the real promotion path, then given two verified failures. It remains `active` (below the
  configured three-consecutive-failure and five-observation thresholds), so it is a tempting
  but lower-reliability distractor. The synthetic failures are a benchmark fixture and are
  labelled as such in the runner.

The bad-track-record fixture is proved by
`tests/integration/test_benchmark_v2.py::test_v2_seeding_adds_an_active_bad_track_record_distractor`.

Not yet seeded: hand-written **near-duplicate/noise** memories. Leakage detection does cover
near-duplicate *tasks*: `check_leakage()` normalizes instructions and compares 3-token
shingles across splits, in addition to exact duplicates and an optional family-holdout check.
The v2 suite passes `check_leakage(tasks, require_family_holdout=True)` with zero issues.

## Baselines

All baselines use the same tasks, environment, tool budget, and agent. The card-level baselines
are policy-free; the policy is the ablation.

| Baseline | Meaning |
|---|---|
| `no_memory` | no memory supplied |
| `retry_only` | retry the same failing call |
| `bm25` | lexical card retrieval, no policy |
| `vector` | dense vector card retrieval, no policy |
| `hybrid_rrf` | BM25 + vector RRF, no policy |
| `hybrid_rrf_version_filter` | RRF then keep only candidates whose declared environment id matches the task; no compatibility/applicability/utility policy |
| `hybrid_rerank` | RRF + configured reranker (lexical by default, optional local CrossEncoder) |
| `outcome_aware` | RECAP's compatibility, applicability, schema, utility, and reliability policy |
| `bm25_chunk`, `vector_chunk`, `hybrid_rrf_chunk` | event-aware chunk retrieval, environment-scoped |
| `outcome_aware_chunk` | event-aware chunks fed into the policy |
| `bm25_naive_chunk`, `vector_naive_chunk`, `hybrid_rrf_naive_chunk` | naive fixed-window chunks, same tasks and embedder |
| `outcome_aware_naive_chunk` | naive chunks fed into the policy |

`hybrid_rrf_version_filter` is the fair non-policy baseline. It gets the same environment id
as the policy, so it is not a strawman: it can fix stale-version selection, but it cannot
reason about applicability, schema fingerprints, reliability, or bad-track-record memories.

**`raw_trajectory_rag` is not implemented.** The existing chunk modes index chunks derived
from memories and run evidence, not raw full trajectories. No row in the reports is labelled
`raw_trajectory_rag`, and no claim is made about it.

## Metrics

The reports include:

- task success and a percentile bootstrap 95% CI over tasks;
- tool calls, retries, and per-stage retrieval latency;
- `harmful_transfer`: a memory was applied and the task still failed;
- `incompatible_selected`: independent post-hoc compatibility/applicability/schema audit;
- `hard_negative_selected` and `hard_negative_outranked_correct`;
- `false_abstention` for `outcome_aware`: relevant memory existed but nothing was selected;
- recall@3, MRR, and nDCG@3.

Not yet implemented: **abstention precision/recall with negative-control tasks** and
**paired bootstrap CIs across >= 3 seeds**. The current CIs resample tasks within one run and
must not be described as seed-paired intervals. There are no tasks for which the correct
answer is "abstain", so a precision number would be fabricated; only false abstention is
reported.

## Running it

The runner needs Postgres and Elasticsearch. The services in `docker-compose.yml` are started
with:

```bash
docker compose start postgres elasticsearch
```

Hashing embedder (offline, deterministic, not semantic):

```bash
recap db migrate
recap eval run --suite v2 --split dev
recap eval run --suite v2 --split test
```

Real nomic embeddings through LM Studio:

```bash
export EMBEDDING_PROVIDER=openai_compatible
export EMBEDDING_BASE_URL=http://localhost:1234/v1
export EMBEDDING_API_KEY=lm-studio
export EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
export EMBEDDING_DIMENSIONS=768
export EMBEDDING_DOCUMENT_PREFIX='search_document: '
export EMBEDDING_QUERY_PREFIX='search_query: '
export RECAP_VECTOR_DIMENSION=768
# Use an isolated prefix and aliases so a 768-d run cannot swap the default
# 256-d aliases. The committed evidence used the prefix `recap-test-nomic`.
export ELASTICSEARCH_INDEX_PREFIX=recap-nomic
export ELASTICSEARCH_MEMORIES_READ_ALIAS=recap-nomic-memories-read
export ELASTICSEARCH_MEMORIES_WRITE_ALIAS=recap-nomic-memories-write
export ELASTICSEARCH_CHUNKS_READ_ALIAS=recap-nomic-chunks-read
export ELASTICSEARCH_CHUNKS_WRITE_ALIAS=recap-nomic-chunks-write
export RECAP_INDEX_GENERATION=1
export RECAP_CHUNKS_INDEX_GENERATION=2
recap search init
recap eval run --suite v2 --split dev
```

A dimension or model change requires a new index generation or a new isolated
prefix/aliases; never mix vector spaces in one index.

## What the dev run does and does not show

The measured dev run is in [`evidence/phase2-summary.md`](evidence/phase2-summary.md).
In short:

- policy-free card retrieval (`bm25`, `vector`, `hybrid_rrf`) recovers 50% of tasks, applies a
  stale/incompatible memory in 50%, and selects an incompatible memory in 100%;
- `hybrid_rrf_version_filter` and `outcome_aware` both recover 100% with 0% harmful transfer
  and 0% incompatible selection on this split;
- therefore this dev split does **not** show a utility advantage of the full policy over the
  fair version filter. It shows that both fix stale-version selection, and that the policy
  matches the fair baseline while rejecting bad-track-record/wrong-family distractors.
- chunk baselines are all 100% because chunk retrieval is environment-scoped and the split has
  no same-environment distractor set. This is still a degenerate chunk ablation; do not cite
  it as evidence that event-aware chunking beats naive chunking.

The next credible step is to run the held-out test families and add same-environment noise,
near-duplicate memories, and negative-control tasks before making a generalization or
abstention-quality claim.
