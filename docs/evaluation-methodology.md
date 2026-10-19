# Evaluation Methodology

## 1. What this benchmark is

A **deterministic pipeline benchmark**. It measures whether the retrieval and selection
machinery behaves as designed, using:

- a deterministic mock API with two schema versions
- a deterministic reference agent that follows memory-mapped parameter names
- a deterministic, non-semantic hashing embedding provider
- a small, hand-built task suite

It is **not** a measurement of LLM agent performance. The reports embed that statement, and
this document exists so the claim cannot drift.

What it *can* show, and does show:

- a policy-free retrieval baseline selects schema-incompatible memories
- the outcome-aware policy does not
- removing memory prevents recovery entirely
- retrying the same failing call prevents recovery entirely
- the promotion gate actually gates (memories are created through the real compile ->
  validate -> promote path)

## 2. Task suite

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

### Splits and leakage

- **train** — available for developing the policy and weights.
- **dev** — used for weight tuning and for the demo benchmark.
- **test** — held out; run explicitly and reported separately.

`check_leakage(tasks)` hashes normalized instructions and flags any instruction appearing in
more than one split. The suite is grouped by template so a scenario is not duplicated across
splits.

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
| `outcome_aware` | RECAP's full selection policy |

The three retrieval baselines deliberately skip applicability, compatibility, and utility
policy. That is the ablation: it isolates what the policy adds. See
[`retrieval.md` §8](retrieval.md#8-ablation-design-why-the-baselines-are-fair).

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

## 5. Reported results (dev split)

From a real run (`recap eval run --split dev`):

| baseline | success | 95% CI | recall@3 | MRR | nDCG@3 | tool calls | harmful transfer | incompatible selection |
|---|---|---|---|---|---|---|---|---|
| no_memory | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 1.00 | 0.000 | 0.000 |
| retry_only | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 1.00 | 0.000 | 0.000 |
| bm25 | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 4.00 | 0.000 | 1.000 |
| vector | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 4.00 | 0.000 | 1.000 |
| hybrid_rrf | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 4.00 | 0.000 | 1.000 |
| outcome_aware | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 4.00 | 0.000 | 0.000 |

How to read this honestly:

- **Success is not the interesting column here.** The reference agent can recover from the
  v2 failure even with the wrong (v1) memory, because it eventually hits the right parameter
  by trying. That is a property of this tiny environment, not evidence that retrieval
  quality is irrelevant.
- **Incompatible selection is the interesting column.** Every policy-free baseline selected
  a memory that its own declared schema says does not apply. `outcome_aware` selected none.
- **The intervals are degenerate** because n=2. They are reported as computed, not padded.
- **`no_memory` and `retry_only` at 0.000** confirm the environment genuinely requires the
  recovery knowledge; the task is not trivially solvable.

The test split shows the same qualitative pattern on held-out tasks.

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

## 7. Reproducing

```bash
docker compose up -d postgres elasticsearch
recap db migrate
recap search init
recap eval run --split dev
recap eval run --split test
```

Outputs:

```
benchmark/reports/benchmark_<split>_<timestamp>.json
benchmark/reports/benchmark_<split>_<timestamp>.csv
benchmark/reports/benchmark_<split>_<timestamp>.md
```

Reports are gitignored except for `.gitkeep`; regenerate them rather than trusting a
committed number.

## 8. What would make this a real evaluation

In rough priority order:

1. **More tasks.** Dozens per family, several families, multiple seeds. Confidence intervals
   only become informative with n >> 2.
2. **A real agent.** An LLM agent with a fixed model, temperature, and tool budget. The
   reference agent is a correctness harness, not a capability measurement.
3. **A real embedding model.** The hashing provider captures token overlap only; the vector
   baseline understates what a neural embedder does. Configure `EMBEDDING_PROVIDER=openai_compatible`
   and record the model, dimension, and index generation in the report. A real-model run must
   never be mixed with a deterministic run in the same comparison.
4. **Ablations beyond retrieval mode.** Compatibility-only, utility-only, reliability-only,
   and reranker-on/off, so the contribution of each policy component is separable.
5. **Adversarial memories.** Deliberately misleading but highly similar memories, to measure
   whether the policy resists them rather than only whether it filters stale ones.
6. **Cost/latency reporting.** Latency is captured per decision
   (`stage_latency_ms`, `total_latency_ms`) but not yet aggregated into the report.

## 9. Non-claims

- No claim that RECAP improves LLM task success.
- No claim of statistical significance at the current sample size.
- No claim that the utility weights are optimal; they are interpretable defaults tuned on
  dev.
- No claim that the mock environment represents production APIs.
