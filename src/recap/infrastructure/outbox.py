"""Outbox drain worker.

Indexing intent is written transactionally with the database change (see
``IndexSynchronizer``). This worker is the consumer side: it claims due rows,
applies them to Elasticsearch, and either completes them or schedules a retry
with exponential backoff.

Guarantees:

- a row is only marked ``done`` after Elasticsearch accepted the write
- a transport failure leaves the row retryable and marks the memory
  ``index_status=failed``; durable database state is never rolled back
- exhausted retries move the row to ``dead`` (visible, not silently dropped)
- claiming uses ``FOR UPDATE SKIP LOCKED`` on PostgreSQL so two workers can run
  concurrently without processing the same row

The worker is pull-based and synchronous. It is invoked by
``recap search drain`` (and can be called from a scheduler or a cron job); there
is no always-on background thread inside the API process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from recap.domain.enums import MemoryStatus
from recap.errors import SearchUnavailable
from recap.logging import get_logger

if TYPE_CHECKING:
    from recap.container import Container, Services
    from recap.infrastructure.db.models import OutboxRow

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_BATCHES = 100


class OutboxWorker:
    def __init__(
        self,
        container: Container,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_batches: int = DEFAULT_MAX_BATCHES,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_batches < 1:
            raise ValueError("max_batches must be >= 1")
        self.container = container
        self.batch_size = batch_size
        self.max_batches = max_batches

    def drain(self) -> dict[str, int]:
        """Process due rows until the queue is empty or the batch cap is hit.

        Returns counts for observability. Each batch runs in its own
        transaction so a poison row cannot prevent other rows from committing.
        """
        counts = {
            "claimed": 0,
            "indexed": 0,
            "deleted": 0,
            "skipped": 0,
            "failed": 0,
            "dead": 0,
            "chunk_indexed": 0,
            "chunk_deleted": 0,
            "chunk_skipped": 0,
            "lifecycle_applied": 0,
        }
        for _ in range(self.max_batches):
            progressed = 0
            with self.container.services() as services:
                rows = services.outbox.claim_due(self.batch_size)
                if not rows:
                    break
                for row in rows:
                    counts["claimed"] += 1
                    progressed += 1
                    self._process(services, row, counts)
            if progressed == 0:
                break
        if counts["claimed"]:
            logger.info(
                "outbox drained",
                extra={"event": "outbox_drained", **counts},
            )
        return counts

    def _process(self, services: Services, row: OutboxRow, counts: dict[str, int]) -> None:
        try:
            entity_type = row.entity_type or "memory"
            if entity_type == "chunk":
                self._process_chunk(services, row, counts)
            elif entity_type == "chunk_lifecycle":
                self._process_chunk_lifecycle(services, row, counts)
            elif entity_type in {"memory", "memory_card"}:
                self._process_memory(services, row, counts)
            else:
                services.outbox.mark_failed(row, f"unknown outbox entity_type {entity_type!r}")
                counts["failed"] += 1
                if row.state == "dead":
                    counts["dead"] += 1
        except SearchUnavailable as exc:
            self._mark_chunk_failure(services, row)
            services.outbox.mark_failed(row, exc.message)
            counts["failed"] += 1
            if row.state == "dead":
                counts["dead"] += 1
        except Exception as exc:  # worker boundary: never crash the drain loop
            logger.warning(
                "outbox row failed",
                extra={
                    "event": "outbox_row_failed",
                    "outbox_id": row.id,
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            self._mark_chunk_failure(services, row)
            services.outbox.mark_failed(row, f"{type(exc).__name__}: {exc}")
            counts["failed"] += 1
            if row.state == "dead":
                counts["dead"] += 1

    def _mark_chunk_failure(self, services: Services, row: OutboxRow) -> None:
        if row.entity_type == "chunk" and row.operation == "index":
            services.chunk_store.mark_indexed([row.entity_id], success=False)

    def _process_memory(self, services: Services, row: OutboxRow, counts: dict[str, int]) -> None:
        memory = services.memories.get_memory(row.entity_id, row.workspace_id)
        if memory is None:
            # The entity is gone. There is nothing to index and nothing to
            # delete; completing the row is the honest outcome.
            services.outbox.mark_done(row)
            counts["skipped"] += 1
            return

        if row.operation == "delete":
            services.container.search_repository.delete_memory(memory.id, memory.memory_version)
            services.outbox.mark_done(row)
            counts["deleted"] += 1
            return

        if services.index_sync(memory):
            services.outbox.mark_done(row)
            counts["indexed"] += 1
        else:
            services.outbox.mark_failed(
                row, services.index_sync.last_error or "index_memory reported failure"
            )
            counts["failed"] += 1
            if row.state == "dead":
                counts["dead"] += 1

    def _process_chunk(self, services: Services, row: OutboxRow, counts: dict[str, int]) -> None:
        chunk = services.chunk_store.get(row.entity_id)
        if chunk is None:
            # Postgres is authoritative for indexing, but a delete intent still
            # owns the search document id. Remove it even if the row is gone so
            # an ES orphan cannot outlive a completed delete row.
            if row.operation == "delete":
                services.chunk_sync.search.delete_chunks([row.entity_id])
                services.outbox.mark_done(row)
                counts["chunk_deleted"] += 1
            else:
                services.outbox.mark_done(row)
                counts["chunk_skipped"] += 1
            return
        if row.operation == "delete":
            services.chunk_sync.search.delete_chunks([chunk.chunk_id])
            services.outbox.mark_done(row)
            counts["chunk_deleted"] += 1
            return
        embedding = services.chunk_sync.embeddings.embed_documents([chunk.text])[0]
        services.chunk_sync.search.index_chunks(
            [chunk],
            [embedding],
            active_for_retrieval=chunk.active_for_retrieval,
        )
        services.chunk_store.mark_indexed(
            [chunk.chunk_id],
            success=True,
            embedding_fingerprint=getattr(
                services.chunk_sync.embeddings, "embedding_fingerprint", ""
            ),
            generation=services.settings.elasticsearch.chunks_index_generation,
        )
        services.outbox.mark_done(row)
        counts["chunk_indexed"] += 1

    def _process_chunk_lifecycle(
        self, services: Services, row: OutboxRow, counts: dict[str, int]
    ) -> None:
        memory = services.memories.get_memory(row.entity_id, row.workspace_id)
        if memory is None:
            services.outbox.mark_done(row)
            counts["skipped"] += 1
            return
        # Re-read the current status: a stale row must never resurrect chunks of
        # a memory that has since been retired.
        active = memory.status is MemoryStatus.ACTIVE
        services.chunk_store.set_active_for_memory(memory.id, memory.memory_version, active=active)
        services.chunk_sync.search.set_active_for_memory(
            memory.id, memory.memory_version, active=active
        )
        services.outbox.mark_done(row)
        counts["lifecycle_applied"] += 1
