# Benchmark v3 — memory-required, leak-resistant evaluation

> **Status:** v3 is the current evaluation suite. It is deliberately separate from
> [benchmark v2](benchmark-v2.md), which is kept as the legacy ceiling/regression suite.
> v2 is not used for the memory claim: `no_memory` succeeds on every v2 task with the
> tested live model, so its success rate cannot show whether memory helps.
>
> The numbers below are from a **deterministic reference-agent run** against live
> PostgreSQL and Elasticsearch with the synthetic 2,000-memory store. They measure
> pipeline, selection, and safety behavior. They are not a live-LLM capability claim.
> Live-model results are published separately and labeled with the exact model.

## 1. Why v3 exists

The v2 suite was useful as a vertical-slice regression test, but a code review found
that its instructions and error messages leaked the fix, its retrieval request carried
oracle metadata, and `no_memory` solved every task. A benchmark where the control arm
succeeds cannot demonstrate the project's thesis.

v3 changes the question from “can the agent recover?” to:

> **Can a policy tell the difference between a relevant memory and a safe memory, and
> can it fail closed when no safe memory exists?**

The suite therefore has three tiers:

| Tier | What the environment does | What it tests |
|---|---|---|
| **A — self-explanatory** | Returns a normal, actionable error or no error. | Harness and control. A plain loop should work. |
| **B — opaque but discoverable** | Rejects the request with an opaque error and a request id. | Can the agent probe, and does memory save tool calls? |
| **C — memory-required** | Accepts the old request but silently changes behavior, or requires a hidden convention. | Can a stale memory cause a wrong answer that no error reveals? |

Tier C is the thesis. A successful HTTP response is not success. The runner verifies
the collected records against ground truth, after the agent stops.

## 2. Task suite

`build_benchmark_v3_tasks()` in `src/recap/evals/datasets.py` creates 48 deterministic
tasks. Each task carries `tier`, `scenario_id`, `gold_memory_ids`, `should_select`,
`silent_drift`, `tenant`, and a schema variant.

| Split | Tasks | Composition |
|---|---:|---|
| `train` | 6 | Tier A controls |
| `dev` | 18 | Tier B opaqueness + silent-cursor, irrelevant-drift, and relevant-drift variants |
| `test` | 24 | Held-out Tier C: silent field, enum, account, and novel failure families |

The test split has 18 tasks where selecting a memory is expected and 6
`should_select=false` negative controls where `no_safe_memory` is the correct answer.

Instructions are generic (“Collect all records for tenant X and report the count”).
They never name the drift, the missing parameter, the hidden enum, or the fix. Tier A
keeps the old self-explanatory error text; Tier B and C use realistic opaque failures.

### Silent drift

`src/recap/environments/v3_api.py` implements the failure families:

- `silent_cursor`: the legacy `cursor` parameter is accepted but ignored, so the
  client receives page one forever and silently truncates the result set.
- `silent_field`: the old response field is accepted but returns an empty list.
- `silent_enum`: an invalid enum silently falls back to a different order; the final
  record order is wrong even though every record is present.
- `silent_account`: the request succeeds for the wrong account scope unless the
  hidden account id is supplied.
- `novel_failure`: the current environment changes in a way no gold memory covers;
  the only safe action is `no_safe_memory`.

The exact completion check is external: the environment exposes the expected records,
and the runner compares the agent's collected ids with them. The model's own claim is
never used as success.

## 3. Memory store and retrieval difficulty

The v3 store is built by `src/recap/evals/v3_seed.py`:

- gold memories are explicit synthetic fixtures with typed actions and preconditions;
- 2,000 synthetic decoys include same-vocabulary, stale, wrong-family, and
  bad-track-record distractors;
- all memories are bulk-indexed through the same Elasticsearch repository used by the
  API;
- ids are namespaced by workspace so two benchmark processes sharing one Postgres
  database cannot collide on global primary keys.

Gold memories carry field-scoped schema-path hashes. Compatibility is evaluated only
against the paths the strategy actually depends on. An irrelevant field change remains
compatible; a relevant change is incompatible. This is tested directly:

```text
irrelevant change -> COMPATIBLE
relevant change   -> INCOMPATIBLE
no path evidence  -> fall back to the stricter whole-schema check
```

Pure retrieval baselines receive a natural-language query. They do **not** receive
`environment_id`, `error_code`, `task_family`, or another gold label in the retrieval
request. The outcome-aware policy may use the current tool-schema hashes for the
compatibility gate, but it does not receive the gold family label.

Always-retry and no-memory remain controls. An always-abstain policy is not a win: the
report includes selection recall, false-abstention rate, and abstention precision/recall,
so abstaining everywhere looks bad.

## 4. Baselines and metrics

The v3 report can include the full ladder:

`no_memory`, `retry_only`, `bm25`, `vector`, `hybrid_rrf`, `hybrid_rrf+rerank`,
`hybrid_rrf_version_filter`, `fingerprint_filter_oracle`, `outcome_aware`.

`fingerprint_filter_oracle` is named an oracle because it filters on the exact
environment fingerprint. It is an upper bound on what an exact-match filter can do, not
a fair policy baseline.

Metrics are computed from recorded outcomes:

| Metric | Meaning |
|---|---|
| `task_success` | externally verified completion |
| `recall@3`, `MRR`, `nDCG@3` | pre-policy retrieval ranking |
| `selection_precision`, `selection_recall` | post-policy selection quality |
| `harmful_transfer` | a memory was applied and the verified result was still wrong |
| `incompatible_selection` | independently re-evaluated selected memory fails compatibility/applicability |
| `false_abstention` | no memory selected even though a relevant one existed |
| `abstention_precision`, `abstention_recall` | quality of the no-safe-memory decision |

Confidence intervals use a **scenario-clustered bootstrap**. Multiple paraphrases of one
scenario are nuisance variation, not independent samples. The report prints
`n_scenarios` and `n_runs` next to the interval.

### Acceptance gate

A v3 run fails review if Tier C `no_memory` succeeds on more than 50% of tasks. The
gate is written into the report payload and rendered in the Markdown report. A broken
suite must be redesigned, not reported with a footnote.

### Non-discrimination guard

If two baselines have identical metrics on every field, the report prints a warning.
This is intentional: a benchmark where every arm is the same is not showing a
difference, even if the table looks tidy.

## 5. Deterministic held-out test result

Command (local services, real Elasticsearch + Postgres, synthetic decoys):

```powershell
$env:RERANK_PROVIDER = "cross_encoder"
$env:RERANK_MODEL_PATH = (Resolve-Path "models\ms-marco-MiniLM-L-6-v2").Path
recap eval run --suite v3 --split test --agent reference --fixture-count 2000 --no-resume
```

Dataset hash: `sha256:8438ccf598486dbdb44f8b29593187720e769b1389666fc1ff6dc568716697eb`.
Store: 3 gold fixtures + 2,000 synthetic decoys. Tool budget: 6. Seed: 1. Agent:
deterministic `reference-agent-v1`. Reranker: local `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Embedder: deterministic `recap-hashing-embedding-v1`, 256-d. This run is a pipeline and
safety test, **not** a claim about nomic or another semantic embedder; the live run uses
`text-embedding-nomic-embed-text-v1.5` (768-d) and reports that separately.

| Baseline | Success | 95% CI | Recall@3 | MRR | Harmful transfer | Incompatible selection | False abstention |
|---|---:|---|---:|---:|---:|---:|---:|
| `no_memory` | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bm25` | 0.417 | [0.250, 0.625] | 0.750 | 0.576 | 0.583 | 0.292 | 0.000 |
| `vector` | 0.208 | [0.083, 0.375] | 0.292 | 0.275 | 0.417 | 0.750 | 0.000 |
| `hybrid_rrf` | 0.292 | [0.125, 0.458] | 0.625 | 0.451 | 0.583 | 0.583 | 0.000 |
| `hybrid_rerank` | 0.292 | [0.125, 0.458] | 0.625 | 0.443 | 0.542 | 0.625 | 0.000 |
| `hybrid_rrf_version_filter` | 0.125 | [0.000, 0.292] | 0.750 | 0.625 | 0.000 | 0.125 | 0.667 |
| `fingerprint_filter_oracle` | 0.125 | [0.000, 0.292] | 0.750 | 0.625 | 0.000 | 0.125 | 0.667 |
| `outcome_aware` | **0.625** | [0.417, 0.833] | 0.458 | 0.324 | **0.125** | **0.000** | **0.000** |

**Acceptance gate:** Tier C `no_memory` success `0/24 (0.000)`; threshold ≤ 0.50;
**PASS**.

A separate live-model run on the same split uses
`gemma-4-e4b-uncensored-hauhaucs-aggressive` with nomic 768-d embeddings, streaming, seed 1,
and a 600 s read / 900 s overall timeout. Its aggregate `no_memory` is 6/24, but all six are
`novel_failure` controls; on the 18 selectable tasks it is 0/18. `outcome_aware` selects the
correct gold on 18/18 selectable tasks but completes only 8; the 10 remaining failures are
execution failures after correct selection, which the report records as harmful transfer.
Exact environment, matrix, and caveat:
[`evidence/phase3b-v3-live-test.md`](evidence/phase3b-v3-live-test.md).

What the table supports:

- The suite is memory-required: `no_memory` succeeds on zero Tier C tasks.
- Policy-free retrieval transfers harmful memory often (`bm25` 58.3%, `hybrid_rrf`
  58.3%) and selects incompatible memories (`vector` 75%, `hybrid_rrf` 58.3%).
- The outcome-aware policy eliminates incompatible selection and false abstention on
  this split and reduces harmful transfer to 12.5%.
- The exact-filter oracle is **too strict**: it false-abstains on 66.7% of the test
  tasks because the whole fingerprint changed even when the relevant strategy paths
  did not. Field-scoped compatibility is the difference.
- `outcome_aware` has lower recall@3 and MRR than the exact-filter oracle. Ranking and
  selection are different questions. The policy can still choose correctly after a
  lower-ranked candidate is retrieved; this report does not claim the policy ranks
  better.

What it does **not** support:

- `outcome_aware` is not perfect. It succeeds on 15/24 held-out tasks and harmful
  transfer is 12.5% (3 tasks), because a wrong-family field-strategy can still be
  compatible with its own declared paths and satisfy the policy. The other six failures
  are the `novel_failure` negative controls, where the policy selected nothing and said
  `no_safe_memory`; that is the correct safety action, but task success intentionally does
  not count abstention. False abstention is 0%, and abstention precision/recall are both
  1.0. This is reported as a real failure, not hidden in an average.
- The test split contains one run per task and one seed. The clustered CI is not a
  multi-seed significance claim.
- These are deterministic numbers. The live model can behave differently. Live runs
  are published separately with their exact model, prompt, timeout, and seed.
- `hybrid_rrf_version_filter` and `fingerprint_filter_oracle` are identical on this
  split; the report emits a non-discrimination warning. On v3 test, the environment id
  and the whole fingerprint happen to filter the same candidates, so the extra oracle
  is not adding anything here.

## 6. Reproducing and extending

```bash
# deterministic regression
recap eval run --suite v3 --split test --agent reference --fixture-count 2000 --no-resume

# live model (one seed; record the exact model and timeout in the report)
EMBEDDING_PROVIDER=openai_compatible \
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5 \
EMBEDDING_DIMENSIONS=768 \
LLM_PROVIDER=openai_compatible \
LLM_MODEL=<exact id from /v1/models> \
recap eval run --suite v3 --split test --agent llm_verifying \
  --baselines no_memory,bm25,outcome_aware --fixture-count 2000 --seed 1 --no-resume
```

The live run needs a dedicated index prefix and aliases if the default index was built
with a different embedding dimension. Do not re-point a healthy production alias to a
benchmark index; use benchmark-only names.

Tests:

- `tests/unit/test_benchmark_v3.py` — dataset determinism, tier balance, tier A/B
  behavior, field-scoped compatibility.
- `tests/integration/test_benchmark_v3.py` — live Postgres + Elasticsearch smoke that
  proves silent drift needs memory.
- `tests/unit/test_benchmark_v2.py` — v2 regression suite remains unchanged.

## 7. Threats to validity

- **Synthetic environment.** The mock API is deterministic and small. It is designed to
  isolate selection safety, not to estimate production traffic performance.
- **Synthetic store.** Gold and decoy memories are fixtures. The report records the
  provenance; do not cite them as production memories.
- **One small local model.** Live numbers are one configured model on one machine, not
  a model-family claim.
- **One seed for the headline deterministic run.** The cluster bootstrap describes
  task-level uncertainty, not run-to-run model variance.
- **Single-node Elasticsearch.** No shard, replica, or scale claim is made here.
- **Task success excludes safe abstention.** A correct `no_safe_memory` is not counted
  as task success. It is measured by abstention precision/recall and false abstention,
  which is why “always abstain” still looks bad in the table.
