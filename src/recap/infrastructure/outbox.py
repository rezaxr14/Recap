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
        except SearchUnavailable as exc:
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
                    "entity_id": row.entity_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            services.outbox.mark_failed(row, f"{type(exc).__name__}: {exc}")
            counts["failed"] += 1
            if row.state == "dead":
                counts["dead"] += 1
