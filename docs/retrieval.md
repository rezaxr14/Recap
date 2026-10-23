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

For ablations, `build_lexical_query(..., pure=True)` uses equal-weight text fields and
disables the exact-signature phrase and metadata boosts. This is the mode used by
`scripts/retrieval_depth.py`; it keeps a retrieval comparison from quietly winning on
metadata that a natural-language query would not have.

Identifier analysis is opt-in (`ELASTICSEARCH_IDENTIFIER_ANALYSIS=true`). It adds a
`recap_identifier` subfield to `title`, `summary`, `failure_signature`, and `failure_code`,
built with `word_delimiter_graph`, case-change splitting, and `preserve_original`. That lets
a prose query such as `invalid cursor` match `INVALID_CURSOR` without changing the stored
value. The analyzer cannot be added to an existing mapping in place: enabling it against an
old index raises `ConfigurationError` and requires a new generation plus
`recap search reindex`.

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

kNN pre-filters on `workspace_id` and the allowed statuses **inside** the `knn` query. The
pre-filter is not a post-processing step: a foreign-workspace vector is never a candidate.
`elasticsearch.knn_num_candidates` controls the per-shard candidate pool (default 200). The
mapping also excludes `content_embedding` from `_source`: the vector is an index structure,
not document content, and keeping a second copy only inflates the store.

`elasticsearch.dense_vector_quantization` is `none` (float32 `hnsw`) by default or
`int8_hnsw`. Elasticsearch retains the raw float vectors on disk for rescoring and future
quantization changes, so `int8` is a **search-memory** option, not a disk-size win. It also
changes ranking, so it requires a new index generation and its own recall measurement.

### 2.3 Hybrid fusion

Default: application-side Reciprocal Rank Fusion.

```
score(d) = sum over branches b of 1 / (k + rank_b(d)),  k = 60
```

RRF is used rather than score normalization because BM25 scores and cosine similarities are
not comparable and their distributions shift per query. Rank-based fusion needs no
calibration.

Native Elasticsearch RRF (`es_rrf`) is implemented as an `rrf` retriever with `standard` and
`knn` children and selectable via `elasticsearch.hybrid_strategy`. It is not the default for
two reasons: the application-side path persists per-branch ranks and scores, and native RRF
is licence-gated. On the measured Basic 8.19.3 node it returned
`403 security_exception: current license is non-compliant for [Reciprocal Rank Fusion (RRF)]`.
The configured path fails closed with `SearchUnavailable`; it never silently switches to
application-side fusion. `app_rrf` therefore remains the only usable fusion path in this
deployment.

### 2.4 Reranking (`hybrid_rerank`)

Fusion is rank-based and deliberately conservative. `hybrid_rerank` adds a second pass over
the fused candidates:

- `recall.reranker=lexical` (default) uses `LexicalReranker`
  (`lexical-overlap-v1`): a deterministic query-coverage/document-precision score
  (`0.7 * coverage + 0.3 * precision`).
- `recall.reranker=cross_encoder` uses a local `sentence-transformers` CrossEncoder
  (`RERANK_PROVIDER=cross_encoder`, `RERANK_MODEL_PATH=models/ms-marco-MiniLM-L-6-v2`).
  The path must already exist; the adapter passes `local_files_only=True` and never downloads
  weights. Install the optional dependency with `pip install -e ".[rerank]"`.
- `recall.reranker=none` disables the stage.

All implementations satisfy the `RerankerProvider` port. The reranker can only change
candidate **order**. Eligibility, applicability, compatibility, and utility are evaluated
afterwards, so reranking cannot make an unsafe memory selectable. Its rank is recorded as
`branch_ranks["rerank"]`, relevance is taken from the rerank score, and
`config_snapshot.rerank_applied` / `reranker` / `reranker_model_path` record what ran. The
wall-clock for the stage is stored as `stage_latency_ms["rerank"]` on every decision and is
aggregated in the benchmark's separate `performance` section. If the reranker raises, the
engine logs a warning, keeps fusion order, and marks the decision `degraded`.

The lexical reranker remains the default because it is deterministic and dependency-free.
The cross-encoder is an opt-in quality comparison, not a claim that it improves every
dataset; it must be measured on the same tasks and reported with its exact model path.

### 2.5 Chunk-granule retrieval

The card index stores one document per memory. The chunk index (`recap-chunks-*`, currently
generation 2) stores deterministic chunks derived from the same memories and their evidence
runs (see `docs/adr/0005-event-aware-chunking.md`). Both `event_aware` and `naive_fixed` chunks
can coexist: `chunk_strategy` is stored on the document and is a query filter. Chunk queries
are also scoped to the request's environment fingerprint, because a chunk compiled for one API
version must not be retrieved for another. Chunk modes run the same BM25/vector branches over
chunk documents, fuse at chunk level with application RRF, then group by `memory_id:version`:

- the memory's group score is the **best supporting chunk** RRF score;
- every supporting chunk id, branch rank, and event id is preserved on the candidate;
- the grouped memory is then fed into the **unchanged** applicability/compatibility/utility
  policy.

Chunk modes: `bm25_chunk`, `vector_chunk`, `hybrid_rrf_chunk`, and `outcome_aware_chunk`
for `event_aware` chunks; the benchmark runs the same four over `naive_fixed` chunks as
`*_naive_chunk`. The card modes (`bm25`, `vector`, `hybrid_rrf`, `outcome_aware`) remain, so
the report can compare granule and strategy choice on identical tasks. Chunk similarity never
overrides safety: only `outcome_aware_chunk` applies the policy, and the decision records
`retrieval_granule=chunk` plus `supporting_chunk_ids`/`supporting_event_ids`.

Current caveat: the dev dataset has one memory per environment version. Environment scoping
therefore leaves a single candidate, so the `naive_fixed` vs `event_aware` comparison is
degenerate and is not presented as evidence of a chunking win.

There is no silent fallback: if the chunk index is empty or unavailable, chunk retrieval
reports a degraded/empty result rather than returning card hits under a chunk label.

### 2.6 Measured depth (Phase 4.2)

`scripts/retrieval_depth.py` builds the v3 gold memories and 10,000 decoys in memory, indexes
them into temporary indices, and measures each retrieval layer against live Elasticsearch.
The run below used the deterministic hashing embedder (256-d), not the live nomic model; it is
an Elasticsearch-mechanics measurement, not a semantic-quality claim. Full provenance, raw
numbers, and limitations: [`docs/evidence/phase4-retrieval-depth.md`](evidence/phase4-retrieval-depth.md).

The table reports means over three repeats; the evidence page keeps the min-max range. HNSW is
approximate, so a few candidate counts return slightly different rankings on repeated queries.

| Experiment | Measured result |
|---|---|
| Identifier analyzer, 33 v3 natural-language queries | Recall@3 `0.818 -> 0.818`, MRR `0.500 -> 0.500` |
| Identifier analyzer, 10 diagnostic identifier probes | MRR `0.933 -> 1.000`; recall@3 `1.000` both |
| Pre-filtered kNN, `num_candidates` 50/100/200/500/1000 | `200`: recall@3 `0.394`; `500`: `0.424`; `1000`: `0.424`; search p50 ~`50 ms` throughout |
| Application RRF | recall@3 `0.556`, recall@10 `1.000`, MRR `0.472`, p50 `130 ms` |
| Native ES RRF | not usable: Basic licence `403` |
| float32 `hnsw` vector branch | recall@3 `0.404`, MRR `0.322`, index `21.7 MB` |
| `int8_hnsw` vector branch | recall@3 `0.212`, MRR `0.106`, index `24.2 MB` |

The decisions follow the measurements rather than the feature list:

- `app_rrf` stays the default. Native RRF cannot run on this licence; the flag fails closed
  instead of falling back.
- `knn_num_candidates=200` stays the default. On this 10k corpus, 500 gave slightly better
  recall@3 (`0.424` vs `0.394`) with flat search latency, but 200 is the value used in the
  100k latency run. Changing it needs a 100k sweep, not an inference from a 10k index.
- `identifier_analysis` stays **off by default**. It is a precision tool for identifier-heavy
  queries and needs a new index generation; it did not lift v3 natural-language recall.
- `dense_vector_quantization=none` stays the default. `int8_hnsw` lost recall and is *larger*
  on disk here because Elasticsearch retains raw float vectors for rescoring; it is a
  search-memory trade-off, not a free win.

These defaults are a negative result about feature maximalism, not a claim that the features
are useless. Each is implemented, tested, and selectable with an explicit reindex path.

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
