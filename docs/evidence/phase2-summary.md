# Phase 2 evidence: benchmark v2 and live new-family runs

Generated 2026-10-21 UTC. This file separates three kinds of evidence:

1. **Live local model runs** against LM Studio (real HTTP, real inference).
2. **Real-embedding benchmark runs** with `text-embedding-nomic-embed-text-v1.5` (768-d).
3. **Deterministic pipeline runs** with the hashing embedder.

The benchmark agent is still the deterministic reference agent, not an LLM. No number below
is a claim of LLM task success.

## Exact models and stack

| Role | Exact artifact | Source |
|---|---|---|
| Agent chat | `gemma-4-e4b-uncensored-hauhaucs-aggressive` | LM Studio `http://localhost:1234/v1` |
| Embeddings (benchmark) | `text-embedding-nomic-embed-text-v1.5`, 768 dimensions | LM Studio; `search_document: ` / `search_query: ` prefixes |
| Reranker (configured) | lexical overlap `lexical-overlap-v1` | deterministic, used by `hybrid_rerank` |
| Postgres | 17-alpine, container `recap-postgres` | port 5433, database `recap_test` |
| Elasticsearch | 8.19.3, container `recap-elasticsearch` | port 9200, index prefix `recap-test-nomic` |

## Live model runs for new failure families

These are real model-driven tool-use runs, persisted through the API-first agent. They were
run **without a memory advisory**, so they test whether the model can read the current tool
schema and complete the task; they are not memory-transfer evidence.

| Task | Model | Success | Tool calls | Tokens in/out | Duration | Run id |
|---|---|---:|---:|---:|---:|---|
| `response_field` v2 | `gemma-4-e4b-uncensored-hauhaucs-aggressive` | yes | 1 | 759 / 41 | 5,444 ms | `run_0f9bb5e28d894be1a5f8423bee7a76cb` |
| `idempotency_key` v2 | `gemma-4-e4b-uncensored-hauhaucs-aggressive` | yes | 1 | 781 / 51 | 6,076 ms | `run_6652d48712834e9ab2da30575b1f53b4` |

Transcripts:

- `docs/evidence/live-runs/phase2-agent-response-field-v2-gemma-fixed.json`
- `docs/evidence/live-runs/phase2-agent-idempotency-key-v2-gemma.json`

The first response-field attempt
(`phase2-agent-response-field-v2-gemma.json`) **failed** for a real reason: the LLM agent
verifier only counted records under `items`, while v2 returns them under `data`. The model had
received all seven records and correctly stopped, but RECAP reported failure. The verifier now
reads both declared record-list fields, and a unit test covers the rename
(`test_response_field_rename_still_counts_real_records`). The failed transcript is kept as
evidence; it was not overwritten.

These runs do not use memory and do not prove that memory helps. They do prove that the live
model can supply `request_id` and read `data` when the schema changes.

## Benchmark v2 dev run (real nomic embeddings)

Command shape: `run_benchmark(container, suite="v2", split="dev")` with the settings in
`docs/benchmark-v2.md`. Report:

- `docs/evidence/phase2-benchmark-v2-dev-nomic/benchmark_dev_20261021T090252Z.md`
- `docs/evidence/phase2-benchmark-v2-dev-nomic/benchmark_dev_20261021T090252Z.json`
- `docs/evidence/phase2-benchmark-v2-dev-nomic/benchmark_dev_20261021T090252Z.csv`

| Item | Value |
|---|---|
| Suite / split | `v2` / `dev` |
| Tasks | 24 (2 families x 2 versions x 6 paraphrases) |
| Dataset hash | `sha256:14579e31a18d2193f29f3946482a2942664e019a61abfbec20deb817e5acf539` |
| Agent | deterministic `reference-agent-v1` |
| Embedder | `text-embedding-nomic-embed-text-v1.5`, 768-d, fingerprint `sha256:376c38156f69e20800da276053b1a6557b13d6c376133f34f5be618fba3c6812` |
| Index generation | `recap-test-nomic-memories-v1` |
| Memory setting | frozen snapshot per run |
| Hard negatives | stale version, wrong family, active bad-track-record fixture |

Key rows (24 tasks each):

| Baseline | Success | 95% task CI | recall@3 | MRR | Harmful transfer | Incompatible selection | Hard-neg outrank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_memory` | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `retry_only` | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bm25` | 0.500 | [0.292, 0.708] | 1.000 | 0.750 | 0.500 | 1.000 | 0.500 |
| `vector` | 0.500 | [0.292, 0.708] | 1.000 | 0.750 | 0.500 | 1.000 | 0.500 |
| `hybrid_rrf` | 0.500 | [0.292, 0.708] | 1.000 | 0.750 | 0.500 | 1.000 | 0.500 |
| `hybrid_rrf_version_filter` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `hybrid_rerank` (lexical) | 0.500 | [0.292, 0.708] | 1.000 | 0.750 | 0.500 | 1.000 | 0.500 |
| `outcome_aware` | 1.000 | [1.000, 1.000] | 1.000 | 0.750 | 0.000 | 0.000 | 0.000 |
| `bm25_chunk` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `bm25_naive_chunk` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `hybrid_rrf_chunk` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `outcome_aware_chunk` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `outcome_aware_naive_chunk` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

### Interpretation

- Policy-free **card** retrieval selects a version-incompatible memory in every task
  (`incompatible_selection_rate=1.0`) and applies one that leads to failure in half
  (`harmful_transfer_rate=0.5`). This is the safety gap RECAP targets.
- Both `hybrid_rrf_version_filter` and `outcome_aware` fix that gap on this split: 100%
  success, 0% harmful transfer, 0% incompatible selection.
- **This split does not show a success advantage of the full policy over the fair version
  filter.** The full policy has a lower MRR (0.750 vs 1.000) because the correct memory is
  sometimes ranked second, even though it is selected. Reporting only success would hide that.
  The next benchmark iteration needs same-environment noise, near-duplicate memories, and
  subtler drift (for example a schema-hash change without a version bump) to test whether the
  policy's extra checks matter.
- **Chunk baselines remain degenerate.** Chunk retrieval is environment-scoped, and this split
  has no same-environment distractor set, so every chunk baseline is 100%. Do not cite these
  rows as evidence that event-aware chunking beats naive chunking.
- `false_abstention_rate` is 0.000 for `outcome_aware`. There are no negative-control tasks,
  so abstention **precision** is not reported rather than invented.

## Deterministic hashing run

For comparison, the same suite was run with the default 256-d lexical hashing embedder:

- `docs/evidence/phase2-benchmark-v2-dev/benchmark_dev_20261021T090532Z.md`

The headline pattern is the same. The hashing run is the offline reproducibility check; the
nomic run is the real-embedding evidence. Neither is a real-LLM benchmark.

## Still missing

- No full `test` split run yet.
- No same-environment near-duplicate/noise memories; only stale, wrong-family, and
  bad-track-record negatives are seeded.
- No negative-control tasks, so no abstention precision/recall.
- No paired CIs across >= 3 seeds; the displayed CIs resample tasks in one run.
- No `raw_trajectory_rag` baseline.
- No LLM benchmark agent. The live agent runs above are separate from the benchmark matrix.
