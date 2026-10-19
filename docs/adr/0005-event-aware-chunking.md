# ADR 0005: Deterministic event-aware chunking and chunk-level retrieval

- **Status:** Accepted
- **Date:** 2026-10-19
- **Deciders:** RECAP maintainers

## Context

The memory index currently stores one Elasticsearch document per memory card. That is
enough to retrieve a lesson, but it is thin RAG: there is no chunk corpus, no chunk-level
evidence, and no way to compare card retrieval against event-aware retrieval. Raw run events
are deliberately not indexed one-document-per-event. The missing layer is a deterministic
chunking strategy over the events and validation evidence that already back a memory.

The chunker must be pure, offline, reproducible, and safe. It must never split an event,
never index unredacted text, and never let chunk similarity override the existing
applicability, compatibility, or reliability policy.

## Decision

### 1. Four chunk types

| Type | Contents | Source |
|---|---|---|
| `memory_card` | Existing canonical memory text: title, failure signature, preconditions, strategy, postconditions, limitations, evidence summary. | `ExperienceMemory` |
| `failure_window` | The failing event, up to N preceding events, and following events up to the first recovery or terminal outcome. | Run event stream |
| `action_observation` | One tool call and its result as an atomic pair: tool, normalized arguments, status/error code, bounded observation. | Run event stream |
| `validation_evidence` | Sandbox result: reproduced failure, negative guard result, postcondition check. | `validation_results` plus supporting run |

A memory card chunk always exists for a memory. The other types exist only when the source
data exists. No free-form LLM summarization is used.

### 2. Pure, deterministic function

`build_chunks(...)` takes domain objects and configuration, performs no I/O, and returns
immutable chunk records. Same input plus same `chunker_version` yields byte-identical text
and IDs. Dict ordering, timestamps, UUIDs, cursors, and request IDs are normalized before
embedding; exact values remain in keyword fields.

Chunk ID:

```
sha256(workspace_id | memory_id | memory_version | chunk_type |
       seq_start | seq_end | chunker_version | text_hash)[:32]
```

### 3. Atomic units and packing

1. Segment events into atomic units: action+observation pair, standalone state transition, or
   standalone error. An event is never split.
2. Render each unit with a fixed field order and normalized whitespace.
3. Normalize volatile values (`<UUID>`, `<CURSOR>`, `<TIMESTAMP>`, `<REQUEST_ID>`, ...).
   Preserve originals in the chunk's `exact_ids`.
4. Greedy-pack consecutive units up to `target_tokens`, hard cap `max_tokens`.
5. Overlap is measured in units (default 1), so overlap cannot cut an event.
6. Chunks never cross a run boundary. A failure-window chunk starts at the failure event.
7. A unit larger than `max_tokens` keeps head and tail with an explicit
   `[...truncated N chars...]` marker and `truncated=true`; content is never silently dropped.
8. Prepend a deterministic contextual header:
   `Tool: list_records | Error: INVALID_CURSOR | Env: mock-api v1 | Chunk: failure_window`.
9. Cap `max_chunks_per_memory`; prefer failure/validation evidence, and record truncation.

### 4. Token counting without downloads

Use `token_counter: approx_regex`: split on words/punctuation with a fixed regex and apply a
conservative multiplier (1.3 by default). Do not use `tiktoken` or any tokenizer that fetches
vocabulary files. The counter name is stored in the chunk record and is part of the chunker
version contract.

### 5. Redaction before chunking

The chunker consumes already-redacted event payloads and re-runs the redactor's detector as
a last line of defence. If any configured secret pattern survives, chunk building fails
closed for that memory instead of indexing it. `redaction_rule_version` is stored on every
chunk.

### 6. Storage

New Postgres table `memory_chunks` (source of truth):

```
chunk_id PK, workspace_id, memory_id, memory_version, run_id nullable,
chunk_type, seq_start, seq_end, event_ids jsonb, text, text_hash,
token_estimate, truncated, chunker_version, redaction_rule_version, created_at
```

New Elasticsearch index `recap-chunks-v1` with `recap-chunks-read` / `recap-chunks-write`
aliases and `dynamic: strict`.

Keyword fields: `chunk_id`, `memory_id`, `memory_version`, `workspace_id`, `run_id`,
`chunk_type`, `failure_code`, `tool_names`, `tool_versions`, `environment_fingerprint`,
`task_families`, `chunker_version`, `embedding_fingerprint`, `exact_ids`.
Text: `chunk_text`, plus context fields. Numbers: `seq_start`, `seq_end`, `token_estimate`.
Boolean: `active_for_retrieval`. Dense vector: `content_embedding` with the configured
dimension, cosine, HNSW.

Postgres remains authoritative. Chunk documents are rebuildable with
`recap search reconcile --chunks`.

### 7. Chunk-level retrieval, memory-level decision

- BM25 and kNN run over chunk documents.
- ES-level filters stay minimal: `workspace_id`, `active_for_retrieval`, and
  `must_not` incompatible. Everything else remains application-side so mismatches can be
  observed and explained.
- Fuse at chunk level with application RRF (`k=60`).
- Group by `memory_id:version`; memory score is the best chunk RRF score (sum-of-top-2 is an
  ablation). Preserve every supporting chunk ID, branch ranks, and event IDs.
- Feed grouped memories into the existing policy unchanged. Chunk similarity never overrides
  applicability, compatibility, or reliability.
- Add modes `bm25_chunk`, `vector_chunk`, `hybrid_rrf_chunk`, `outcome_aware_chunk`.
  `card` modes remain to provide a fair baseline.

### 8. Status sync and frozen evaluation

When a memory transitions or is superseded, its chunk documents' `active_for_retrieval` must
follow through the outbox. `recap search reconcile` rebuilds chunk documents from Postgres
and reports drift. Run-level snapshots freeze chunk membership too: chunks whose memory/version
is not in the snapshot are rejected, and evaluation never writes chunks.

### 9. No silent fallback

If chunk retrieval is requested but the chunk index is empty or unavailable, the decision is
an explicit error or a labeled degraded result. It must not silently return card results
while claiming chunk retrieval.

## Consequences

### Positive

- Retrieval returns the parent memory plus traceable evidence chunks and event IDs.
- Chunking and embedding choices can be ablated on the same tasks.
- A pure chunker is cheap to test and cannot perform I/O or execute tools.

### Negative

- More storage, more index documents, and a second indexing path to keep consistent.
- Chunk status synchronization and reconcile add operational work.
- Token estimates are approximate; budgets must stay below provider limits.

## Alternatives considered

**Fixed character windows over a serialized trace.** Rejected as the default because it cuts
events and destroys evidence traceability. Kept only as a labeled `naive_fixed` baseline.

**One ES document per raw event.** Rejected: event spam, no memory-level context, and no
clear lifecycle for retrieval.

**LLM-generated chunk summaries.** Rejected for the core path: non-deterministic, adds cost,
and weakens evidence traceability. Could be a later optional enrichment, clearly labeled.

**Chunk score decides selection.** Rejected: similarity is not safety. The existing policy
remains the only selector in outcome-aware mode.
