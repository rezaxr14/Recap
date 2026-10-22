# Evaluation Methodology

> **Current suite: benchmark v3.** It is the memory-required, leak-resistant suite.
> Benchmark v2 and the legacy suite are retained as regression/ceiling suites; do not
> cite their task success as evidence that memory helps. See
> [`benchmark-v3.md`](benchmark-v3.md) for the design, acceptance gate, and held-out
> deterministic result.

## 0. Which suite answers which question

| Suite | Question | Status |
|---|---|---|
| **v3** | Does memory change the verified answer, and can the policy reject unsafe memory? | Current; Tier C `no_memory` 0/24 on the held-out deterministic run |
| v2 | Does the full vertical slice remain stable across ten failure families? | Legacy ceiling/regression suite; live `no_memory` succeeds on every task |
| legacy | Small six-task smoke path | Regression only |

The rest of this document describes the shared harness and then the v2 details that
remain useful for regression testing. v3-specific rules are in
[`benchmark-v3.md`](benchmark-v3.md).

## 1. What this benchmark is

A benchmark with two agent modes over the same deterministic environments:

- the default **deterministic pipeline benchmark**, using a reference agent that follows
  memory-mapped parameter names;
- an opt-in **real-model benchmark** (`--agent llm`), using the configured chat provider in the
  same bounded tool loop and the same persisted-run accounting.

Both modes use a deterministic mock API with two schema versions, a small hand-built task
suite, and either real nomic embeddings or a deterministic hashing embedder. The default mode
measures whether the retrieval and selection machinery behaves as designed. The live mode
measures one configured model's behavior in that mock environment; it is not a general
LLM-capability claim. Every report records the agent kind, exact model, prompt version,
temperature, seed, and tool budget, so the two modes cannot be confused.

What the deterministic mode *can* show, and does show:

- a policy-free retrieval baseline selects schema-incompatible memories
- the outcome-aware policy does not
- removing memory prevents recovery entirely
- retrying the same failing call prevents recovery entirely
- the promotion gate actually gates (memories are created through the real compile ->
  validate -> promote path)

The live mode adds a different question: does a real model actually need the memory? On the
v2 dev and held-out test splits the answer was no for the tested model: it solved all 24 dev
and all 36 test tasks with `no_memory`. On v3 the question is different: the aggregate
`no_memory` result is 6/24, but all six are `novel_failure` controls; on the 18 tasks that
require selecting a memory it is 0/18. The live reports keep aggregate and selectable-task
counts separate, and record the exact model, timeout, and seed rather than treating “model
solved it” as the same thing as “memory was unnecessary”.

## 2. Task suite

The original six-task suite remains in `default_tasks()` for the legacy smoke path. The
current evaluation suite is **benchmark v3**; its design and commands are in
[`benchmark-v3.md`](benchmark-v3.md). Benchmark v2 is retained as the legacy
ceiling/regression suite in [`benchmark-v2.md`](benchmark-v2.md). v3 adds 48 tasks across
three memory-required tiers, a 2,000-decoy synthetic store, silent-drift environments, and
field-scoped schema compatibility.

### Legacy suite

`src/recap/evals/datasets.py`

| task_id | split | env version | failure mode |
|---|---|---|---|
| `paginate-train-01` | train | 1 | `invalid_cursor` |
| `paginate-train-02` | train | 1 | `invalid_cursor` |
| `paginate-dev-01` | dev | 1 | `invalid_cursor` |
| `paginate-dev-02` | dev | 2 | `renamed_parameter` |
| `paginate-test-01` | test | 1 | `invalid_cursor` |
| `paginate-test-v2-01` | test | 2 | `renamed_parameter` |

Each task carries: task family, task version, instruction, environment version, failure
mode, expected memory signature, relevant memory tags, seed, difficulty, side-effect risk.

### Benchmark v2

`build_benchmark_v2_tasks()` returns 120 hierarchical tasks: ten failure families, two
environment versions, and six deterministic paraphrases per family/version. Splits are
family-held-out: five families in `train`, two in `dev`, three in `test`. Every memory is
created through the real compile -> validate -> promote path. The store also contains stale,
wrong-family, and bad-track-record hard negatives.

### Splits and leakage

- **train** — available for developing the policy and weights.
- **dev** — used for weight tuning and for the demo benchmark.
- **test** — held out; run explicitly and reported separately.

`check_leakage(tasks)` normalizes instructions and flags exact duplicates. It also compares
3-token shingles across splits and, for benchmark v2, enforces family holdout via
`require_family_holdout=True`. The v2 suite currently reports zero issues. This is
near-duplicate task detection, not embedding-similarity detection over memories.

**Frozen-memory setting.** Retrieval is read-only during evaluation and no task writes a
memory. Test episodes therefore cannot leak memories into later test episodes. The protocol
block in every report records `memory_setting: frozen_memory`.

## 3. Baselines

All baselines run the same tasks, the same environment, the same agent, and the same tool
budget.

| Baseline | What it does |
|---|---|
| `no_memory` | agent attempts the task with no memory at all |
| `retry_only` | agent retries the same failing call (a naive baseline) |
| `bm25` | lexical retrieval, rank by relevance, **no policy** |
| `vector` | dense-vector retrieval, **no policy** |
| `hybrid_rrf` | fused lexical + vector, **no policy** |
| `hybrid_rrf_version_filter` | fused lexical + vector, then keep only the task's declared environment id; no applicability/schema/reliability policy |
| `outcome_aware` | RECAP's full selection policy |
| `bm25_chunk` / `vector_chunk` / `hybrid_rrf_chunk` | the same three retrieval baselines run over **event-aware** chunk documents (best-chunk RRF per memory), still **no policy** |
| `outcome_aware_chunk` | event-aware chunk retrieval fed into RECAP's full selection policy |
| `*_naive_chunk` | the same four chunk baselines over `naive_fixed` chunks, on the same tasks and embedder |

Both chunk strategies are indexed side by side: `chunk_strategy` is stored on every chunk and
is a query filter. Chunk queries are also scoped to the request's environment fingerprint,
because a chunk compiled for one API version is not a valid candidate for another. Card
retrieval still returns incompatible memories so the decision can explain the rejection.

The three card retrieval baselines deliberately skip applicability, compatibility, and utility
policy. That is the ablation: it isolates what the policy adds. The `_chunk` and
`*_naive_chunk` baselines repeat the comparison at chunk granularity on identical tasks, so the
report separates "does the policy help?" from "does chunking help?". See
[`retrieval.md` §8](retrieval.md#8-ablation-design-why-the-baselines-are-fair).

**v2 limitation:** the v2 dev split seeds one correct memory per environment version plus
stale, wrong-family, and bad-track-record distractors, but no same-environment near-duplicate
or noise set. Chunk retrieval is environment-scoped, so every chunk baseline still sees the
same correct candidate and succeeds. The chunk-strategy matrix is wired and reproducible, but
it is degenerate on v2 until the dataset adds same-environment distractors. No "event-aware
beats naive" claim is made from the current numbers. v3 adds a 2,000-memory synthetic decoy
store and field-scoped compatibility, but still does not claim a chunking win.

`raw_trajectory_rag` is not implemented. The existing `*_chunk` modes index chunks derived
from memories and run evidence, not raw full trajectories.

## 4. Metrics

All metrics are computed from recorded outcomes. None are hardcoded.

| Metric | Definition |
|---|---|
| `task_success` | the agent completed the task and verified its postcondition |
| `recall_at_3` | relevant memories present in top 3 ranked candidates / total relevant |
| `mrr` | reciprocal rank of the first relevant candidate |
| `ndcg_at_3` | binary-gain nDCG at 3 |
| `tool_calls` | tool invocations used by the agent |
| `harmful_transfer` | a memory was used and the task still failed |
| `incompatible_selected` | selected memories that fail an independent post-hoc compatibility/applicability/schema check |
| `hard_negative_selected` | any selected memory that is not relevant to the task |
| `hard_negative_outranked_correct` | a hard negative ranked above the first relevant memory |
| `false_abstention` | `outcome_aware` selected nothing even though relevant memory existed |

### Confidence intervals

`bootstrap_ci` is a percentile bootstrap over the per-task values (2000 resamples, seed 7).
With 2 tasks per split the interval is necessarily degenerate; that is reported honestly
rather than hidden by pretending the sample is larger.

`paired_difference_ci` exists for paired baseline-vs-treatment comparisons on identical
tasks. It is implemented and unit-tested; the current report surfaces per-baseline CIs.

### The safety audit

`incompatible_selected` is the discriminating metric. It is computed *after* the episode:

1. take the memory ids the baseline actually selected
2. rebuild the task's true environment context
3. re-evaluate compatibility, applicability, and tool-schema hashes independently
4. count selections that fail any of those checks

The audit never feeds back into selection. A baseline cannot be penalized or rewarded by a
metric it influenced, which is what makes the comparison meaningful.

## 5. Reported results

### Current v3 held-out test

Deterministic reference agent, 24 Tier C tasks, 2,000 synthetic decoys, 256-d hashing
embedder. `outcome_aware` succeeds on 15/24 tasks (62.5%; 15/18 selectable tasks) with 12.5%
harmful transfer, 0% incompatible selection, and 0% false abstention. `no_memory` is 0/24,
so the acceptance gate passes. The full matrix, CIs, and retained failures are in
[`results-v3.md`](results-v3.md). This is a pipeline/safety result, not a live-model claim.

### Legacy v2 dev

From a real run with real nomic embeddings (`recap eval run --suite v2 --split dev`), 24 tasks:

| baseline | success | 95% CI | recall@3 | MRR | harmful transfer | incompatible selection |
|---|---|---|---|---|---|---|
| `no_memory` | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 |
| `retry_only` | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 |
| `bm25` | 0.500 | [0.292, 0.708] | 1.000 | 0.750 | 0.500 | 1.000 |
| `vector` | 0.500 | [0.292, 0.708] | 1.000 | 0.750 | 0.500 | 1.000 |
| `hybrid_rrf` | 0.500 | [0.292, 0.708] | 1.000 | 0.750 | 0.500 | 1.000 |
| `hybrid_rrf_version_filter` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 |
| `outcome_aware` | 1.000 | [1.000, 1.000] | 1.000 | 0.750 | 0.000 | 0.000 |

How to read this honestly:

- Policy-free card retrieval applies an incompatible/stale memory in half the tasks and
  selects one in every task. The version filter and `outcome_aware` both remove that failure.
- This split does **not** show a success advantage of the full policy over the fair version
  filter. The policy's MRR is lower (0.750 vs 1.000), so it is not a clean win even on ranking.
  Harder same-environment noise and subtler drift are needed before the extra policy checks
  can be distinguished.
- The chunk rows are all 100% because chunk retrieval is environment-scoped and the split has
  no same-environment distractor set. That is a degenerate ablation, not evidence that
  event-aware chunking beats naive chunking.
- The intervals resample tasks within one run. They are not seed-paired and do not establish
  statistical significance.
- `false_abstention` is 0.000 for `outcome_aware`; abstention precision/recall is not reported
  because there are no negative-control tasks.

### Legacy v2 live-model run

A separate v2 run used the same 24 dev tasks and real nomic embeddings with
`gemma-4-e4b-uncensored-hauhaucs-aggressive` through LM Studio (temperature 0, prompt
`llm-tool-agent-v1`, tool budget 6, seed 1):

| baseline | success | recall@3 | MRR | harmful transfer | incompatible selection | input tokens |
|---|---|---|---|---|---|---|
| `no_memory` | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 21,282 |
| `bm25` | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 54,556 |
| `hybrid_rrf_version_filter` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 50,863 |
| `outcome_aware` | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 50,073 |

The model solved every task without memory. `bm25` selected an incompatible memory on all 24
tasks and still succeeded, which means task success alone cannot validate selection safety for
this model.

The held-out test split (36 tasks; `required_parameter`, `enum_change`, `date_format`) shows
the same result:

| baseline | success | recall@3 | MRR | harmful transfer | incompatible selection | input tokens |
|---|---|---|---|---|---|---|
| `no_memory` | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 36,108 |
| `bm25` | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 80,605 |
| `hybrid_rrf_version_filter` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 69,641 |
| `outcome_aware` | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 69,636 |

This is one model, one seed, and four baselines per split; it is not the full Phase 3 matrix
and does not show that memory is useless. It shows that both current splits are a ceiling for
this model: the schema and error text contain enough information to adapt without memory.
Full provenance and limits:
[`evidence/phase3-llm-dev-core/summary.md`](evidence/phase3-llm-dev-core/summary.md) and
[`evidence/phase3-llm-test-core/summary.md`](evidence/phase3-llm-test-core/summary.md). The
deterministic hashing comparison remains in
[`evidence/phase2-summary.md`](evidence/phase2-summary.md).

## 6. Protocol metadata

Every JSON report includes:

```json
"protocol": {
  "memory_setting": "frozen_memory",
  "agent": "deterministic-reference-agent-v1",
  "provider": "deterministic",
  "environment": "mock-api",
  "note": "Deterministic mock pipeline test. Not evidence of real LLM performance."
}
```

Plus `generated_at`, `split`, `dataset_hash`, the task definitions, per-task rows, and the
summary. The Markdown report repeats the caveat in its header.

Live reports add `agent.kind: "llm"`, the exact model, prompt version, temperature, tool
budget, token usage, provider errors, and checkpoint metadata. A live report also includes the
retrieval/embedding fingerprint and the active index, so a model result cannot be attributed
to the wrong embedding space.

## 7. Reproducing

```bash
docker compose start postgres elasticsearch
recap db migrate
recap search init
recap eval run --suite v2 --split dev
recap eval run --suite v2 --split test
```

For real nomic embeddings, export the `EMBEDDING_*` variables and `RECAP_VECTOR_DIMENSION=768`
as shown in [`benchmark-v2.md`](benchmark-v2.md), then initialize a separate index generation.

For a live-model run, configure `LLM_*` and add `--agent llm`. The CLI resumes from a matching
checkpoint by default; use `--no-resume` for an independent fresh run:

```bash
recap eval run --suite v2 --split dev --agent llm --seed 1 \
  --baselines no_memory,bm25,hybrid_rrf_version_filter,outcome_aware \
  --output docs/evidence/phase3-llm-dev-core
```

Outputs:

```
benchmark/reports/benchmark_<split>_<timestamp>.json
benchmark/reports/benchmark_<split>_<timestamp>.csv
benchmark/reports/benchmark_<split>_<timestamp>.md
```

Normal reports are gitignored except for `.gitkeep`; regenerate them rather than trusting a
committed number. The Phase 2 evidence copies under `docs/evidence/phase2-benchmark-v2-dev*`
are committed deliberately as provenance for this milestone.

## 8. What would make this a stronger evaluation

In rough priority order:

1. **Improve the live agent's execution, not just its selection.** On v3 the policy selected
   the correct gold on 18/18 selectable tasks, but the small model completed only 8. A better
   tool-use prompt or a larger model is needed before the live result measures selection
   rather than execution.
2. **Repeated seeds and paired CIs.** Run each `(task, baseline, seed)` at least three times
   and report paired intervals, not only task resampling.
3. **Scale and cluster behavior.** The current store is 2,003 synthetic memories and one
   single-node Elasticsearch. Add realistic corpus sizes, shard layouts, and latency budgets.
4. **Ablations beyond retrieval mode.** Compatibility-only, utility-only, reliability-only,
   and reranker-on/off, so the contribution of each policy component is separable.
5. **Adversarial memories.** Same-environment near-duplicates and noise beyond the current
   stale, wrong-family, and bad-track-record decoys.
6. **`raw_trajectory_rag`.** Standard RAG over full past trajectories, with no repair
   compilation, as a baseline.

## 9. Non-claims

- No claim that RECAP improves general LLM capability. The live v3 run completes 8/18
  selectable tasks; its failures are retained.
- No claim of statistical significance at one seed and one model.
- No claim that the full policy beats the fair `hybrid_rrf_version_filter` on the v2 dev
  split; it matches success and safety there, with lower MRR.
- No claim that the v3 live result matches the deterministic 15/18 selectable-task result.
  It is 8/18, and 10 failures are execution failures after correct selection.
- No claim that the utility weights are optimal; they are interpretable defaults tuned on
  dev.
- No claim that the mock environment represents production APIs or that the synthetic store
  represents production memories.
- No claim that the chunk ablation shows event-aware chunking is better; it is degenerate on
  the current split.
- No scale, multi-tenant security, or production-traffic claim.
