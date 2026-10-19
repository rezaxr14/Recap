# ADR 0002: Hybrid retrieval with application-side RRF and application-level safety filtering

- **Status:** Accepted
- **Date:** 2026-10-19
- **Deciders:** RECAP maintainers

## Context

Memory retrieval must find a recovery lesson from a failure description. Two signals matter:

- **Lexical overlap.** Failure signatures are semi-structured strings
  (`ValidationError|INVALID_CURSOR|list_records`). Exact token and phrase matches are strong
  evidence.
- **Semantic similarity.** A memory phrased differently ("the page token expired") should
  still match a query about "invalid cursor".

BM25 alone misses paraphrase; dense vectors alone miss exact error codes and can be
distractingly confident on near-duplicates. Hybrid fusion is the standard answer. The
questions are *how* to fuse and *where* safety filtering belongs.

## Decision

### 1. Run both branches, fuse by rank

- BM25 over weighted text fields, with `failure_signature` boosted highest.
- kNN over a `dense_vector` field (cosine, HNSW).
- Fuse with Reciprocal Rank Fusion, `k=60`, **in the application** by default.

RRF is chosen over score normalization because BM25 scores and cosine similarities live on
incomparable scales whose distributions shift per query. Rank-based fusion requires no
calibration and is robust to one branch being much more confident than the other.

### 2. Application-side RRF is the default; native ES RRF is available

Elasticsearch 8.19 supports RRF natively. The DSL is implemented and selectable via
`elasticsearch.hybrid_strategy=es_rrf`, but the default is `app_rrf` because the application
needs **per-branch ranks and scores** for the decision record. Fusion in the application
keeps those values available for explanation and for the benchmark's ablation.

### 3. Push only tenant and operational status into Elasticsearch

Hard ES filters:

- `workspace_id` (tenant boundary — non-negotiable)
- `status` in the operational set
- `must_not` known-incompatible documents

Everything else — task family, tool names, environment fingerprint, failure code,
preconditions, reliability — is evaluated in the application.

### 4. Selection is a separate stage from retrieval

`outcome_aware` mode applies eligibility, applicability, compatibility, and a utility
policy. Pure retrieval modes (`bm25`, `vector`, `hybrid_rrf`) deliberately skip those stages
so they function as honest baselines.

## Rationale for application-level filtering

This is the decision most likely to be questioned, so the reasoning is explicit.

**Explanation.** If task family were an ES filter, a mismatched memory would simply not
appear. The decision record would show an empty candidate list and no reason. The system
could not answer "was there a similar memory that we rejected, and why?" — which is the
central product question.

**Safety auditing.** The benchmark's `incompatible_selection_rate` requires observing that a
policy-free baseline selected a memory that is incompatible. If incompatibility were filtered
in ES, that baseline would silently become safe and the metric would be meaningless.

**Correctness of compatibility.** Compatibility is not equality. A changed environment
fingerprint can still be `suspect` rather than `incompatible` if tool schema hashes match.
That requires comparing declared and observed schemas — application logic, not a term query.

**Preconditions require runtime state.** Applicability is evaluated against a state snapshot
supplied by the caller. No index can answer it.

The cost is that Elasticsearch may return candidates that are immediately rejected. That
cost is bounded (`max_candidates=100`) and is the price of an explainable decision.

## Consequences

### Positive

- **Explainable decisions.** Every rejection has a reason string; nothing silently vanishes.
- **Honest ablations.** Retrieval quality and selection policy are separately measurable.
- **Schema-drift correctness.** Suspect-vs-incompatible is computed from hashes, not equality.
- **No calibration.** RRF avoids tuning a score-normalization scheme per corpus.
- **Branch provenance.** Per-branch ranks and scores are persisted for every candidate.

### Negative

- **Wasted work.** ES may rank candidates the policy will reject.
- **Two fusion paths.** `app_rrf` and `es_rrf` must both be maintained.
- **No index-time policy.** Some filtering could be cheaper in ES; it is deliberately not
  done.
- **Application RRF requires both branches to return ids**, which means an extra round trip
  is avoided only by issuing both queries in one request.

### Neutral

- The default embedder is a deterministic hashing provider, not a neural model. This is an
  implementation detail of the current slice, documented prominently; it does not change the
  fusion design. A neural provider can be enabled without touching retrieval code.

## Alternatives considered

**Vector-only retrieval.** Rejected: exact failure codes and signatures are high-precision
signals that embeddings smear. The demo's v1/v2 distinction is carried by a schema hash and a
parameter name — lexical evidence is essential.

**BM25-only retrieval.** Rejected: misses paraphrase and produces brittle matching on
normalized signatures.

**Score normalization (min-max or z-score) then weighted sum.** Rejected: requires per-query
calibration, is sensitive to outliers, and produces scores that are hard to explain.

**Elasticsearch-side RRF as the default.** Rejected for now: it hides per-branch ranks needed
for the decision record and the benchmark. Kept as an option.

**Filtering task family/schema in ES.** Rejected for the reasons above.

**Learning a ranker.** Rejected at this scale: six tasks would make any learned model an
overfitting demonstration. The weighted utility policy is interpretable and tunable on dev.

## References

- `src/recap/infrastructure/elasticsearch/hybrid_queries.py`
- `src/recap/infrastructure/elasticsearch/search_repository.py`
- `src/recap/recall/rrf.py`
- `src/recap/recall/engine.py`
- `src/recap/recall/eligibility.py`
- `docs/retrieval.md`
