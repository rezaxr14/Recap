"""Outbox worker tests.

These use SQLite and a fake search repository so the retry/backoff and
completion semantics are testable without services. The real Elasticsearch path
is covered by ``tests/integration/test_elasticsearch.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from sqlalchemy import select
from tests.support import make_memory

from recap.config import load_settings
from recap.container import Container
from recap.domain.memories import ExperienceMemory
from recap.errors import SearchUnavailable
from recap.infrastructure.db.models import OutboxRow
from recap.infrastructure.db.session import Database, build_engine
from recap.infrastructure.embeddings.fake import FakeEmbeddingProvider
from recap.infrastructure.outbox import OutboxWorker
from recap.repair_compiler.validation import build_default_registry

WORKSPACE = "ws-outbox"


class FakeSearchRepository:
    def __init__(self) -> None:
        self.indexed: dict[str, ExperienceMemory] = {}
        self.deleted: list[tuple[str, int]] = []
        self.fail = False

    def index_memory(
        self, memory: ExperienceMemory, embedding: list[float], *, refresh: bool = True
    ) -> None:
        if self.fail:
            raise SearchUnavailable("fake outage")
        self.indexed[memory.id] = memory

    def delete_memory(self, memory_id: str, memory_version: int, *, refresh: bool = True) -> None:
        if self.fail:
            raise SearchUnavailable("fake outage")
        self.deleted.append((memory_id, memory_version))


def _container() -> tuple[Container, FakeSearchRepository]:
    settings = load_settings()
    database = Database(build_engine("sqlite+pysqlite:///:memory:"))
    database.create_all()
    search = FakeSearchRepository()
    container = Container(
        settings=settings,
        database=database,
        es_client=cast(Any, None),
        embeddings=FakeEmbeddingProvider(dimension=settings.embeddings.dimension),
        index_manager=cast(Any, None),
        search_repository=cast(Any, search),
        validators=build_default_registry(),
    )
    return container, search


def _memory(container: Container, *, suffix: str = "") -> ExperienceMemory:
    return make_memory(
        container,
        workspace_id=WORKSPACE,
        title=f"Recover from INVALID_CURSOR {suffix}".strip(),
        summary="Discard the expired cursor and start from the first page.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
        index=False,
    )


def _enqueue(container: Container, memory: ExperienceMemory, operation: str = "index") -> None:
    with container.services() as services:
        services.outbox.enqueue(
            workspace_id=WORKSPACE,
            entity_id=memory.id,
            operation=operation,
            dedupe_key=f"{operation}:{memory.id}:v{memory.memory_version}",
        )


def _row(container: Container) -> OutboxRow:
    with container.services() as services:
        row = services.session.scalar(select(OutboxRow))
        assert row is not None
        return row


def test_drain_indexes_pending_rows_and_marks_them_done() -> None:
    container, search = _container()
    memory = _memory(container)
    _enqueue(container, memory)

    counts = OutboxWorker(container, batch_size=10, max_batches=5).drain()

    assert counts == {
        "claimed": 1,
        "indexed": 1,
        "deleted": 0,
        "skipped": 0,
        "failed": 0,
        "dead": 0,
    }
    assert memory.id in search.indexed
    row = _row(container)
    assert row.state == "done"
    assert row.attempts == 1
    with container.services() as services:
        refreshed = services.memories.get_memory(memory.id)
        assert refreshed is not None
        assert refreshed.index_generation == container.settings.elasticsearch.index_generation


def test_transport_failure_is_retryable_and_visible() -> None:
    container, search = _container()
    memory = _memory(container)
    _enqueue(container, memory)
    search.fail = True

    counts = OutboxWorker(container, batch_size=10, max_batches=5).drain()

    assert counts["failed"] == 1
    assert counts["dead"] == 0
    row = _row(container)
    assert row.state == "pending"
    assert row.attempts == 1
    assert row.last_error == "fake outage"
    # SQLite drops tzinfo; normalize before comparing.
    scheduled = row.next_attempt_at
    now = datetime.now(UTC)
    if scheduled.tzinfo is None:
        scheduled = scheduled.replace(tzinfo=UTC)
    assert scheduled > now
    with container.services() as services:
        refreshed = services.memories.get_memory(memory.id)
        assert refreshed is not None
        assert refreshed.index_generation == 0

    # Backoff means a second immediate drain must not hammer Elasticsearch.
    assert OutboxWorker(container, batch_size=10, max_batches=5).drain()["claimed"] == 0


def test_retry_recovers_after_transient_outage() -> None:
    container, search = _container()
    memory = _memory(container)
    _enqueue(container, memory)
    search.fail = True
    OutboxWorker(container).drain()

    # Simulate the backoff elapsing.
    with container.services() as services:
        row = services.session.scalar(select(OutboxRow))
        assert row is not None
        row.next_attempt_at = datetime.now(UTC)
    search.fail = False

    counts = OutboxWorker(container).drain()

    assert counts["indexed"] == 1
    assert memory.id in search.indexed
    assert _row(container).state == "done"
    with container.services() as services:
        refreshed = services.memories.get_memory(memory.id)
        assert refreshed is not None
        assert refreshed.index_generation == container.settings.elasticsearch.index_generation


def test_exhausted_retries_become_dead_not_lost() -> None:
    container, search = _container()
    memory = _memory(container)
    _enqueue(container, memory)
    with container.services() as services:
        row = services.session.scalar(select(OutboxRow))
        assert row is not None
        row.max_attempts = 1
    search.fail = True

    counts = OutboxWorker(container).drain()

    assert counts["failed"] == 1
    assert counts["dead"] == 1
    row = _row(container)
    assert row.state == "dead"
    assert row.attempts == 1
    assert row.last_error == "fake outage"


def test_delete_operation_removes_the_document() -> None:
    container, search = _container()
    memory = _memory(container)
    _enqueue(container, memory, operation="delete")

    counts = OutboxWorker(container).drain()

    assert counts["deleted"] == 1
    assert search.deleted == [(memory.id, memory.memory_version)]
    assert _row(container).state == "done"


def test_successful_direct_index_does_not_leave_pending_intent() -> None:
    container, search = _container()
    memory = _memory(container)

    with container.services() as services:
        stored = services.memories.get_memory(memory.id, WORKSPACE)
        assert stored is not None
        assert services.index_sync(stored) is True
        assert services.outbox.pending_count() == 0

    row = _row(container)
    assert row.state == "done"
    assert row.dedupe_key == f"index:{memory.id}:v{memory.memory_version}"


def test_failed_direct_index_leaves_intent_for_the_worker() -> None:
    container, search = _container()
    memory = _memory(container)
    search.fail = True

    with container.services() as services:
        stored = services.memories.get_memory(memory.id, WORKSPACE)
        assert stored is not None
        assert services.index_sync(stored) is False
        assert services.outbox.pending_count() == 1
        assert services.index_sync.last_error == "fake outage"

    row = _row(container)
    assert row.state == "pending"


def test_missing_entity_is_skipped_not_retried() -> None:
    container, search = _container()
    with container.services() as services:
        services.outbox.enqueue(
            workspace_id=WORKSPACE,
            entity_id="mem_does_not_exist",
            operation="index",
            dedupe_key="index:mem_does_not_exist:v1",
        )

    counts = OutboxWorker(container).drain()

    assert counts["skipped"] == 1
    assert counts["failed"] == 0
    assert search.indexed == {}
    assert _row(container).state == "done"


def test_batch_size_is_validated() -> None:
    container, _ = _container()
    with pytest.raises(ValueError):
        OutboxWorker(container, batch_size=0)
    with pytest.raises(ValueError):
        OutboxWorker(container, max_batches=0)
