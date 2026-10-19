# Retrieval

## 1. The claim being tested

Retrieval relevance answers "which memory looks like this failure?". It does not answer
"which memory is safe and useful to apply *here*?". Those are different questions, and
conflating them is how memory systems cause harm: a highly similar memory from a previous
schema version is exactly the kind of candidate a pure vector search will confidently
return.

RECAP therefore separates:

- **retrieval** — produce a ranked candidate set (BM25, vector, hybrid RRF)
- **selection** — decide whether any candidate is operationally applicable

and records both.

## 2. Branches

### 2.1 BM25 (lexical)

`multi_match` over weighted text fields:

| Field | Boost |
|---|---|
| `failure_signature` | 3.0 |
| `failure_code` | 2.5 |
| `title` | 2.0 |
| `summary` | 1.5 |
| `strategy_text` | 1.2 |
| `preconditions_text` | 1.0 |
| `postconditions_text` | 0.8 |
| `limitations_text` | 0.5 |

The failure signature is a normalized `ErrorType|CODE|tool` string, so exact-failure
matching dominates when present.

### 2.2 Dense vector

`knn` over `content_embedding` (cosine, HNSW; dimension comes from
`embeddings.dimension` and must match the mapping).

**Important:** semantic retrieval requires a real embedding endpoint. Configure
`EMBEDDING_PROVIDER=openai_compatible` with `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and
`EMBEDDING_DIMENSIONS`; RECAP calls an already-running API and never downloads weights. The
default local embedder is `HashingEmbeddingProvider` — a deterministic lexical hash
projection, not a neural model. It exists so the pipeline runs offline, captures token
overlap rather than semantics, and is reported as `semantic_retrieval=false`. A local
`sentence-transformers` directory can be used with `EMBEDDING_MODEL_PATH`, but it must already
exist; downloads are disabled.

Changing the model or dimension requires a new index generation and a full reconcile. A
mismatch fails before the index is created or written; vectors are never truncated or padded.

### 2.3 Hybrid fusion

Default: application-side Reciprocal Rank Fusion.

```
score(d) = sum over branches b of 1 / (k + rank_b(d)),  k = 60
```

RRF is used rather than score normalization because BM25 scores and cosine similarities are
not comparable and their distributions shift per query. Rank-based fusion needs no
calibration.

Native Elasticsearch RRF (`es_rrf`) is implemented in the DSL and selectable via
`elasticsearch.hybrid_strategy`, but is not the default: the application-side path gives
per-branch ranks and scores that are persisted in the decision record.

### 2.4 Reranking (`hybrid_rerank`)

Fusion is rank-based and deliberately conservative. `hybrid_rerank` adds a second pass over
the fused candidates:

- `recall.reranker=lexical` (default) uses `LexicalReranker`
  (`lexical-overlap-v1`): a deterministic query-coverage/document-precision score
  (`0.7 * coverage + 0.3 * precision`).
- `recall.reranker=none` disables the stage.

The reranker can only change candidate **order**. Eligibility, applicability,
compatibility, and utility are evaluated afterwards, so reranking cannot make an unsafe
memory selectable. Its rank is recorded as `branch_ranks["rerank"]`, relevance is taken from
the rerank score, and `config_snapshot.rerank_applied` records whether the stage ran. If the
reranker raises, the engine logs a warning, keeps fusion order, and marks the decision
`degraded`.

This is intentionally a lexical reranker rather than a cross-encoder: it is deterministic,
dependency-free, and makes the mode genuinely distinct from fusion. A cross-encoder can
implement the same `Reranker` protocol later.

### 2.5 Chunk-granule retrieval

The card index stores one document per memory. The chunk index (`recap-chunks-v1`) stores
deterministic chunks derived from the same memories and their evidence runs (see
`docs/adr/0005-event-aware-chunking.md`). Chunk modes run the same BM25/vector branches over
chunk documents, fuse at chunk level with application RRF, then group by `memory_id:version`:

- the memory's group score is the **best supporting chunk** RRF score;
- every supporting chunk id, branch rank, and event id is preserved on the candidate;
- the grouped memory is then fed into the **unchanged** applicability/compatibility/utility
  policy.

Chunk modes: `bm25_chunk`, `vector_chunk`, `hybrid_rrf_chunk`, and `outcome_aware_chunk`.
The card modes (`bm25`, `vector`, `hybrid_rrf`, `outcome_aware`) remain, so the report can
compare granule choice on identical tasks. Chunk similarity never overrides safety: only
`outcome_aware_chunk` applies the policy, and the decision records
`retrieval_granule=chunk` plus `supporting_chunk_ids`/`supporting_event_ids`.

There is no silent fallback: if the chunk index is empty or unavailable, chunk retrieval
reports a degraded/empty result rather than returning card hits under a chunk label.

## 3. Hard filters

Only two constraints are pushed into Elasticsearch:

- `workspace_id` (tenant boundary)
- `status` in the operational set, plus `must_not` for known-incompatible documents

Everything else is evaluated in the application **on purpose**:

| Signal | Why application-side |
|---|---|
| task family | a mismatch should be retrieved and explained, not hidden |
| tool names | same |
| environment fingerprint | needs schema-hash comparison, not equality |
| failure code | needs compatibility reasoning |
| preconditions | requires runtime state |
| reliability | requires posterior computation |

If these were ES filters, the decision record would show an empty candidate set and no
explanation. The safety audit would also lose the ability to observe that a baseline
selected something incompatible.

When a request carries `run_id`, the engine also resolves the run's `memory_snapshot_id` and
drops candidates that are not in that frozen set (`not_in_memory_snapshot`), pin a different
version (`snapshot_version_mismatch`), or have drifted from the recorded content hash
(`snapshot_content_mismatch`). This is an app-side filter so the decision record still shows
the candidate and the reason. Snapshots are monotonic for normal lifecycle changes: newly
promoted memories cannot enter, while a memory quarantined/retired after the snapshot is still
rejected by live eligibility. An unknown run or a missing referenced snapshot fails closed.

## 4. Applicability

Preconditions are predicates evaluated against a bounded state snapshot built from the
request context plus environment facts:

```
environment_id, environment_fingerprint, environment_version, environment_type,
task_family, error_type, error_code, tool_names, tool_versions,
tool_schema_fingerprints, <caller state>
```

Result is `satisfied`, `violated`, or `unknown`. **`unknown` is not satisfied.**
`is_operationally_applicable` requires `satisfied`, which is what makes the system fail
closed when the caller cannot supply enough state.

The predicate language is intentionally tiny (`eq`, `neq`, `in`, `exists`, numeric
comparisons). It cannot call functions, traverse attributes, or evaluate expressions.

## 5. Compatibility

Compatibility is resolved from two sources:

1. stored `MemoryShift` compatibility for `(memory, environment fingerprint)`
2. live comparison of the memory's origin fingerprint and declared tool schema hashes

Resolution order (`combine_compatibility`):

- stored `incompatible` -> `incompatible`
- fingerprint differs but all declared tool schema hashes match -> `suspect`
- fingerprint differs -> `suspect`
- stored `suspect` -> `suspect`
- fingerprint matches -> `compatible`
- no fingerprint on the memory -> `unknown`

`incompatible` is a hard rejection. `suspect` is a hard rejection by default
(`allow_suspect_operational_memories=false`) and also carries a large negative utility
weight.

## 6. Utility policy (`utility-v1`)

Features are logged per candidate; the score is a transparent weighted sum, normalized to
`[0,1]`.

| Feature | Weight |
|---|---|
| relevance (fused, normalized) | +0.24 |
| exact failure-signature match | +0.18 |
| error-code match | +0.10 |
| tool match | +0.05 |
| task-family match | +0.07 |
| environment match | +0.06 |
| precondition satisfaction ratio | +0.12 |
| reliability posterior mean | +0.10 |
| evidence strength (saturates at 5 verified outcomes) | +0.03 |
| negative evidence (failure share) | -0.10 |
| validation recency (90-day half-life) | +0.03 |
| side-effect risk | -0.06 |
| compatibility penalty | -0.25 |
| staleness penalty | -0.05 |
| uncertainty penalty | -0.06 |

Selection rule:

```
if applicability != satisfied:        reject "preconditions_not_satisfied"
if compatibility == incompatible:      reject "incompatible_environment"
if total < no_safe_memory_threshold:   reject "below_utility_threshold"
otherwise:                             select
```

Then a diversity cap (`max_per_strategy`, default 2) and `final_k` are applied. Candidates
dropped by those caps are recorded as `duplicate_strategy` / `final_k_exceeded`, so a
"missing" memory is never unexplained.

Weights are starting defaults, tuned on the **dev** split only. The test split is never used
for tuning. They are deliberately interpretable rather than learned: with six tasks, a
learned ranker would be overfitting dressed up as science.

## 7. Reliability

Per-memory Beta-Binomial posterior:

```
alpha = prior_alpha + verified_successes
beta  = prior_beta  + verified_failures
mean  = alpha / (alpha + beta)
sd    = sqrt(alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)))
```

Only `verified_success` and `verified_failure` outcomes move the counters. `partial` and
`unknown` are recorded but do not change reliability. This prevents an agent's
self-reported "it worked" from laundering a bad memory into a good one.

## 8. Ablation design (why the baselines are fair)

This is the part most easy to get wrong, so it is explicit in code:

```python
strict_policy = request.mode in {RetrievalMode.OUTCOME_AWARE, RetrievalMode.OUTCOME_AWARE_CHUNK}
```

- `bm25`, `vector`, `hybrid_rrf` (and their `_chunk` variants) rank by relevance and **skip**
  applicability, compatibility, and utility policy. They are pure retrieval baselines.
- `outcome_aware` and `outcome_aware_chunk` apply the full policy.

That is a deliberate ablation: it measures exactly what the policy adds. If the retrieval
baselines also applied the policy, the comparison would be policy-vs-policy and would say
nothing about selection. The `_chunk` vs card comparison isolates granule choice.

The safety metric is computed independently, after the fact, from the recorded selection and
the task's true schema. It never influences selection. In the dev split:

| baseline | success | incompatible selection |
|---|---|---|
| no_memory | 0.000 | 0.000 |
| retry_only | 0.000 | 0.000 |
| bm25 | 1.000 | 1.000 |
| vector | 1.000 | 1.000 |
| hybrid_rrf | 1.000 | 1.000 |
| outcome_aware | 1.000 | 0.000 |

The retrieval baselines "succeed" on the dev split because the v2 task's memory happens to
be rankable, but they select an incompatible memory to get there — the reference agent
recovers via the wrong path. `outcome_aware` succeeds without ever selecting an incompatible
memory. That contrast is the result the project is built to demonstrate, and it is computed
from recorded outcomes, not asserted. The chunk-granule baselines are reported alongside the
card baselines in every benchmark JSON/Markdown report; numbers are read back from the
persisted run and decision records.

## 9. Decision record

Every retrieval persists:

- query (redacted), mode, policy version, index generation, embedding model id
- status and reason (`selected N` or `no_safe_memory: reason=count, ...`)
- per candidate: branch ranks, branch scores, fused score/rank, hard-filter result,
  applicability result and reason, compatibility status and reason, all utility features,
  score components, final score, selected flag
- rejected candidates with a primary reason and detail
- stage latencies, degraded flag, warnings

This is what makes "why did it return nothing?" a queryable question rather than a mystery.

## 10. Failure and degradation

| Failure | Behaviour |
|---|---|
| embedding provider raises | warn, degrade to BM25; vector-only mode returns `degraded` with no selection |
| Elasticsearch unreachable | `SearchUnavailable` -> API 503, decision not fabricated |
| no candidates match | `no_safe_memory` with `no candidates matched the query` |
| all candidates filtered | `no_safe_memory` with the dominant rejection reasons |
| memory in index but missing in DB | candidate recorded as `memory_not_found_in_source_of_truth`, not selected |

There is no path where an unavailable dependency produces a confident answer.
