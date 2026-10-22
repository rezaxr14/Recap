# Results — benchmark v3 held-out test

**Question:** can a memory policy distinguish relevant memory from *safe* memory, and fail
closed when no safe memory exists?

**Answer on this run:** partially, and honestly not perfectly. The outcome-aware policy
reaches 62.5% verified task success and 0% incompatible selection on the held-out test split,
while every relevance-only retrieval baseline transfers harmful memory on at least 41.7% of
tasks. The remaining nine outcomes are reported: three wrong-family field-strategy transfers
and six correct `no_safe_memory` abstentions on negative controls.

This page is the short version. Design, acceptance gate, and threats to validity are in
[`benchmark-v3.md`](benchmark-v3.md). Exact command and environment are in
[`evidence/phase3b-v3-deterministic-test.md`](evidence/phase3b-v3-deterministic-test.md).
The live-model run is reported separately below; it does not retroactively make the
deterministic headline a live result.

## Headline

Deterministic reference agent. Live PostgreSQL 17 + Elasticsearch 8.19.3. Synthetic
2,000-memory store (3 gold fixtures + 2,000 decoys). Held-out test split, 24 Tier C tasks,
one seed. Deterministic 256-d hashing embedder. **Not a live-LLM or semantic-embedder
result.**

| Baseline | Verified success | Harmful transfer | Incompatible selection | False abstention |
|---|---:|---:|---:|---:|
| `no_memory` | 0.0% | 0.0% | 0.0% | 0.0% |
| `bm25` | 41.7% | 58.3% | 29.2% | 0.0% |
| `vector` | 20.8% | 41.7% | 75.0% | 0.0% |
| `hybrid_rrf` | 29.2% | 58.3% | 58.3% | 0.0% |
| `hybrid_rerank` | 29.2% | 54.2% | 62.5% | 0.0% |
| `hybrid_rrf_version_filter` | 12.5% | 0.0% | 12.5% | 66.7% |
| `fingerprint_filter_oracle` | 12.5% | 0.0% | 12.5% | 66.7% |
| `outcome_aware` | **62.5%** | **12.5%** | **0.0%** | **0.0%** |

Tier C `no_memory` acceptance gate: **0/24 (0.0%)**, threshold ≤ 50%, **PASS**.

On the **18 selectable tasks** (excluding the six `novel_failure` negative controls):
`outcome_aware` **15/18**, `bm25` 10/18, `no_memory` 0/18. The six controls are safe
abstentions for the policy, not task successes.

## What this shows

1. **The suite is memory-required.** A control arm with no memory succeeds on zero held-out
   Tier C tasks. Silent drift is not an error message, so completing an API call is not
   success.
2. **Relevance is not safety.** BM25 and hybrid retrieval transfer harmful memory on more
   than half the tasks. Vector retrieval selects an independently-incompatible memory on 75%.
3. **The outcome-aware policy is materially safer.** It eliminates incompatible selection and
   false abstention on this split and reduces harmful transfer from 58.3% to 12.5%.
4. **Exact-environment filtering is too strict.** The oracle false-abstains on 66.7% of tasks
   because the whole fingerprint changed even when the strategy's required schema paths did
   not. Field-scoped compatibility is why the policy can still act.
5. **Ranking and selection are different questions.** `outcome_aware` has lower recall@3
   (0.458) and MRR (0.324) than the exact-filter baseline (0.750 / 0.625). It still selects
   better. Do not claim the policy ranks better than BM25 or the oracle.

## Live-model run (same held-out split)

One local model, `gemma-4-e4b-uncensored-hauhaucs-aggressive`, temperature `0.0`, streaming,
`2048` max output tokens, `600 s` read / `900 s` overall / `180 s` no-progress timeout, seed 1,
nomic 768-d embeddings, and 2,000 synthetic decoys. Exact command and environment:
[`evidence/phase3b-v3-live-test.md`](evidence/phase3b-v3-live-test.md).

| Baseline | Success (24) | Selectable tasks (18) | Harmful transfer | Incompatible selection | False abstention |
|---|---:|---:|---:|---:|---:|
| `no_memory` | 6/24 (25.0%) | **0/18** | 0.0% | 0.0% | 0.0% |
| `bm25` | 10/24 (41.7%) | 4/18 | 58.3% | 29.2% | 0.0% |
| `outcome_aware` | 14/24 (58.3%) | **8/18** | 41.7% | **0.0%** | **0.0%** |

The six `no_memory` successes are all `novel_failure` negative controls where it selected
nothing. On tasks that require a memory it is `0/18`, so the aggregate 25% is not evidence
that no-memory solves silent drift. The recorded acceptance gate passes **6/24 (25.0%)**
against a 50% threshold.

`outcome_aware` selected the gold memory on **18/18** selectable tasks (selection precision
and recall `1.0`) but the model failed to execute 10 of them. Its `harmful_transfer = 41.7%`
is therefore mostly **execution failure after a correct selection**, not wrong-memory
selection. Field-scoped strategy was ignored on all six `silent_field` tasks; `silent_enum`
succeeded 6/6, `silent_account` 2/6. Read the live result as “the policy selects correctly;
this small model often cannot apply the selected strategy”, not as proof that the memory was
harmful.

## What this does not show

- The **headline** is not a live-model result. The separate live run above uses a different
  embedder and agent and records its own exact model, timeout, and seed; it neither validates
  nor invalidates the deterministic policy numbers.
- The policy is **not perfect**. It succeeds on 15/24 tasks. Of the other nine, three are
  harmful transfers (wrong-family field strategy that remained internally compatible) and
  six are the `novel_failure` negative controls: the policy selected nothing and said
  `no_safe_memory`, which is the correct safety action but is not counted as task success.
  False abstention is 0%, and abstention precision/recall are both 1.0.
- This is **one seed, one deterministic split**, with a scenario-clustered bootstrap interval,
  not a multi-seed significance test.
- The environment and the store are **synthetic fixtures**, not production traffic.
- It is **not** a scale or Elasticsearch-cluster claim. The stack is single-node.

## Reproduce

```powershell
$env:RERANK_PROVIDER = "cross_encoder"
$env:RERANK_MODEL_PATH = (Resolve-Path "models\ms-marco-MiniLM-L-6-v2").Path
recap eval run --suite v3 --split test --agent reference `
  --fixture-count 2000 --seed 1 --no-resume
```

Full per-task report: generated under `benchmark/reports/` (ignored by git by policy) and
summarized in [`evidence/phase3b-v3-deterministic-test.md`](evidence/phase3b-v3-deterministic-test.md).
The live report is under `.tmp/live-v3-test/` and summarized in
[`evidence/phase3b-v3-live-test.md`](evidence/phase3b-v3-live-test.md); it is also exported as
a selectable dataset in the [`web/`](../web/) replay app.
