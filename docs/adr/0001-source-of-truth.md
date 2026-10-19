# ADR 0001: PostgreSQL is the source of truth; Elasticsearch is a rebuildable index

- **Status:** Accepted
- **Date:** 2026-10-19
- **Deciders:** RECAP maintainers

## Context

RECAP needs two very different access patterns:

1. **Transactional, ordered, auditable state.** Runs and events arrive in sequence, must be
   idempotent, must never lose order, and must support "what happened?" queries with strong
   consistency. Memory lifecycle transitions and audit records are legal/compliance-shaped
   data.
2. **Relevance-ranked retrieval.** Memories must be found by lexical similarity, dense
   vector similarity, or both, over tens to millions of documents, with sub-second latency.

A single store can do both, but not well. A relational database doing vector search is
awkward and slow at scale; a search engine as the primary store loses transactions, ordering
guarantees, and cheap relational integrity. The naive dual-write approach (write to both,
hope) produces silent divergence.

## Decision

**PostgreSQL is the authoritative store.** Every entity — workspaces, environments, runs,
events, memories, outcomes, decisions, candidates, validations, compatibility, revalidation
jobs, audit events — lives in Postgres. All lifecycle rules and policy decisions are made
against Postgres state.

**Elasticsearch is a derived, rebuildable index.** It contains one document per memory
version, built by a single canonical projection function. It is never the source of a
business decision: retrieval returns candidate ids and scores, and the engine re-loads each
candidate from Postgres before evaluating eligibility, applicability, compatibility, or
utility.

**Synchronization uses a transactional outbox.** A DB write and an outbox row are committed
in the same transaction. Indexing happens after commit. If Elasticsearch is unavailable, the
durable state is already correct, `index_status` is marked `failed`, and the outbox row
remains for reconciliation.

**The index is versioned and aliased.** `recap-memories-v<N>` is the physical index;
`recap-memories-read` and `recap-memories-write` are aliases. A new generation is built
alongside the old one and the read alias is cut over atomically.

## Consequences

### Positive

- **Recoverability.** A lost or corrupted index is rebuilt with `recap search reconcile`.
  There is no scenario where index loss means data loss.
- **Correctness.** Retrieval cannot "invent" a memory that does not exist in Postgres. A
  document whose memory is missing is recorded as
  `memory_not_found_in_source_of_truth` and rejected.
- **Auditability.** Decisions are persisted with the state they were made against, not
  reconstructed later.
- **Schema evolution.** The index can be rebuilt with a new mapping or embedding model
  without a data migration.
- **Testability.** Integration tests can drop and recreate the index freely.

### Negative

- **Dual-store complexity.** Outbox, reconciliation, and index-status tracking are real
  costs. A single store would not need them.
- **Read amplification.** Each candidate is re-fetched from Postgres. At 100 candidates this
  is fine; at 10,000 it would need batching.
- **Eventual consistency.** A memory promoted a millisecond ago may not yet be searchable.
  Reconciliation and refresh interval bound the window; the system does not pretend it is
  zero.
- **Operational surface.** Two stores to back up, monitor, and upgrade.

### Neutral

- The outbox has a pull-based consumer (`recap search drain`) with retry/backoff, but it is
  not an always-on thread; an external scheduler is expected. Indexing that succeeds on the
  request path completes its row immediately, so the queue only holds unsatisfied intent.

## Alternatives considered

**Elasticsearch as primary store.** Rejected: no transactions, weak ordering guarantees,
awkward relational joins for evidence chains, and audit data would be at the mercy of index
corruption or reindex mistakes.

**Postgres only, with `pgvector`.** Rejected for this iteration: the project explicitly
requires a real Elasticsearch integration with BM25 + dense vector + hybrid fusion. Postgres
full-text search plus pgvector is a legitimate future option and would remove the dual-store
complexity, at the cost of weaker hybrid retrieval primitives.

**Dual write without an outbox.** Rejected: it is the classic source of silent divergence.
A failed second write leaves no record that anything is missing.

**Index as authoritative for status.** Rejected: status transitions are policy events with
audit requirements. They belong in the transactional store.

## References

- `src/recap/container.py` (`IndexSynchronizer`)
- `src/recap/infrastructure/db/models.py` (`OutboxRow`)
- `migrations/versions/0002_outbox_active_dedupe.py`
- `src/recap/infrastructure/elasticsearch/index_manager.py`
- `docs/architecture.md`, `docs/data-model.md`
