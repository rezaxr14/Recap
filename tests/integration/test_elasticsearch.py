"""Integration tests against real PostgreSQL and Elasticsearch 8.19.

Every test uses a unique workspace id and asserts on real ES responses, not on
mocks. Run with::

    pytest tests/integration --run-integration
"""

from __future__ import annotations

import uuid

import pytest
from elasticsearch import Elasticsearch
from tests.support import context_for, make_memory

from recap.domain.enums import MemoryStatus, RetrievalMode
from recap.domain.retrieval import RetrievalRequest
from recap.errors import SearchUnavailable
from recap.infrastructure.elasticsearch.search_repository import (
    ElasticsearchMemoryRepository,
)

pytestmark = pytest.mark.integration


def _refresh(container) -> None:  # type: ignore[no-untyped-def]
    container.es_client.indices.refresh(index=container.index_manager.write_index)


# ----------------------------------------------------------------- indexing
def test_index_document_shape_is_strict_and_explicit(container, workspace_id: str) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Refresh invalid cursor",
        summary="Restart pagination from page one to obtain a fresh cursor.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records|invalid cursor",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    _refresh(container)
    response = container.es_client.get(
        index=container.index_manager.write_index, id=f"{memory.id}:v1"
    )
    source = response["_source"]
    assert source["workspace_id"] == workspace_id
    assert source["status"] == "active"
    assert source["failure_code"] == "INVALID_CURSOR"
    assert len(source["content_embedding"]) == 256
    assert "api_key" not in source and "authorization" not in source

    mapping = container.es_client.indices.get_mapping(index=container.index_manager.write_index)
    properties = next(iter(mapping.values()))["mappings"]["properties"]
    assert next(iter(mapping.values()))["mappings"]["dynamic"] == "strict"
    assert properties["content_embedding"]["type"] == "dense_vector"
    assert properties["content_embedding"]["dims"] == 256
    assert properties["failure_signature"]["fields"]["keyword"]["type"] == "keyword"


def test_indexing_is_idempotent_by_document_id(container, workspace_id: str) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Idempotent indexing",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    _refresh(container)
    before = container.es_client.count(index=container.index_manager.read_alias)["count"]
    with container.services() as services:
        again = services.memories.get_memory(memory.id, workspace_id)
        assert again is not None
        assert services.index_sync(again) is True
    _refresh(container)
    after = container.es_client.count(index=container.index_manager.read_alias)["count"]
    assert after == before


# ----------------------------------------------------------------- retrieval
@pytest.mark.parametrize(
    "mode", [RetrievalMode.BM25, RetrievalMode.VECTOR, RetrievalMode.HYBRID_RRF]
)
def test_each_branch_retrieves_the_memory(container, workspace_id: str, mode) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Recover from INVALID_CURSOR",
        summary="Discard the expired pagination cursor and start from the first page.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records|cursor is invalid",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    _refresh(container)
    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=workspace_id, environment_version="1"),
                mode=mode,
                final_k=3,
            )
        )
    assert decision.status.value == "selected", decision.reason
    assert [item.memory_id for item in decision.selected_memories] == [memory.id]
    candidate = next(item for item in decision.candidates if item.memory_id == memory.id)
    assert candidate.branch_ranks
    assert candidate.features.relevance > 0


def test_hybrid_rerank_actually_reranks(container, workspace_id: str) -> None:
    """HYBRID_RERANK must run a real rerank stage, not alias HYBRID_RRF."""
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Recover from INVALID_CURSOR",
        summary="Discard the expired pagination cursor and start from the first page.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records|cursor is invalid",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    _refresh(container)
    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=workspace_id, environment_version="1"),
                mode=RetrievalMode.HYBRID_RERANK,
                final_k=3,
            )
        )
    assert decision.status.value == "selected", decision.reason
    assert decision.config_snapshot["rerank_applied"] is True
    assert decision.config_snapshot["reranker"] == "lexical-overlap-v1"
    assert "rerank" in decision.stage_latency_ms
    candidate = next(item for item in decision.candidates if item.memory_id == memory.id)
    assert candidate.branch_ranks.get("rerank") == 1


def test_wrong_workspace_is_never_returned(container, workspace_id: str) -> None:
    make_memory(
        container,
        workspace_id=workspace_id,
        title="Secret cursor memory",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    other_workspace = f"other-{uuid.uuid4().hex[:10]}"
    _refresh(container)
    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=other_workspace, environment_version="1"),
                mode=RetrievalMode.HYBRID_RRF,
            )
        )
    assert decision.candidates == []
    assert decision.status.value == "no_safe_memory"


def test_incompatible_memory_is_retrieved_but_rejected(container, workspace_id: str) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="v1 cursor recovery",
        summary="Restart pagination from page one using the cursor parameter.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records|cursor is invalid",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    _refresh(container)
    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(
                    workspace_id=workspace_id,
                    environment_version="2",
                    error_code="UNKNOWN_PARAMETER",
                    query="API renamed cursor to page_token",
                ),
                mode=RetrievalMode.OUTCOME_AWARE,
            )
        )
    candidate = next((item for item in decision.candidates if item.memory_id == memory.id), None)
    assert candidate is not None, (
        "the incompatible memory must still be retrievable for explanation"
    )
    assert candidate.selected is False
    assert any("tool_schema_mismatch" in reason for reason in candidate.rejection_reasons)
    assert decision.status.value == "no_safe_memory"


def test_retired_memory_is_not_returned(container, workspace_id: str) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Retired memory",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
        status=MemoryStatus.RETIRED,
    )
    with container.services() as services:
        services.index_sync(memory)
    _refresh(container)
    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=workspace_id, environment_version="1"),
                mode=RetrievalMode.BM25,
            )
        )
    assert decision.candidates == []


# ------------------------------------------------------------------ aliases
def test_read_alias_points_at_write_index(container) -> None:
    aliases = container.es_client.indices.get_alias(name=container.index_manager.read_alias)
    assert container.index_manager.write_index in aliases
    write_aliases = container.es_client.indices.get_alias(name=container.index_manager.write_alias)
    assert container.index_manager.write_index in write_aliases


def test_initialize_is_idempotent(container) -> None:
    first = container.init_search()
    second = container.init_search()
    assert second["index"] == first["index"]
    assert second["created"] is False


def test_reconcile_rebuilds_documents_from_postgres(container, workspace_id: str) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Rebuild me",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    container.es_client.delete(
        index=container.index_manager.write_index,
        id=f"{memory.id}:v1",
        refresh=True,
    )
    with container.services() as services:
        stored = services.memories.get_memory(memory.id, workspace_id)
        assert stored is not None
        assert services.index_sync(stored) is True
    _refresh(container)
    assert container.es_client.exists(
        index=container.index_manager.write_index, id=f"{memory.id}:v1"
    )


def test_outbox_worker_indexes_a_pending_memory(container, workspace_id: str) -> None:
    """The consumer side of the outbox applies intent written by the DB layer."""
    from recap.infrastructure.outbox import OutboxWorker

    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Outbox recovery",
        summary="Restart pagination after an expired cursor.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
        index=False,
    )
    with container.services() as services:
        services.outbox.enqueue(
            workspace_id=workspace_id,
            entity_id=memory.id,
            operation="index",
            dedupe_key=f"index:{memory.id}:v{memory.memory_version}",
        )

    counts = OutboxWorker(container).drain()

    # The queue is global; other tests may leave retryable rows. Assert on this
    # memory's outcome rather than on a global count.
    assert counts["indexed"] >= 1
    assert counts["failed"] == 0
    assert container.es_client.exists(
        index=container.index_manager.write_index, id=f"{memory.id}:v1"
    )
    with container.services() as services:
        stored = services.memories.get_memory(memory.id, workspace_id)
        assert stored is not None
        assert stored.index_generation == container.settings.elasticsearch.index_generation
        from sqlalchemy import select

        from recap.infrastructure.db.models import OutboxRow

        row = services.session.scalar(
            select(OutboxRow).where(
                OutboxRow.dedupe_key == f"index:{memory.id}:v{memory.memory_version}"
            )
        )
        assert row is not None
        assert row.state == "done"


# ------------------------------------------------------------------ failure
def test_es_outage_raises_search_unavailable(live_settings) -> None:  # type: ignore[no-untyped-def]
    settings = live_settings.model_copy(deep=True)
    settings.elasticsearch.url = "http://127.0.0.1:9299"
    settings.elasticsearch.request_timeout_seconds = 0.25
    client = Elasticsearch(
        hosts=[settings.elasticsearch.url],
        request_timeout=0.25,
        max_retries=0,
        retry_on_timeout=False,
    )
    repository = ElasticsearchMemoryRepository(client, settings.elasticsearch)
    try:
        with pytest.raises(SearchUnavailable):
            repository.ensure_ready()
        with pytest.raises(SearchUnavailable):
            repository.count()
    finally:
        client.close()


def test_es_outage_does_not_lose_durable_state(container, workspace_id: str) -> None:
    """A search failure must leave the DB row intact and mark indexing failed."""
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Survives outage",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    broken_settings = container.settings.model_copy(deep=True)
    broken_settings.elasticsearch.url = "http://127.0.0.1:9299"
    broken_settings.elasticsearch.request_timeout_seconds = 0.25
    broken_client = Elasticsearch(
        hosts=[broken_settings.elasticsearch.url],
        request_timeout=0.25,
        max_retries=0,
    )
    broken_repository = ElasticsearchMemoryRepository(broken_client, broken_settings.elasticsearch)
    try:
        with container.services() as services:
            services.index_sync.search = broken_repository
            stored = services.memories.get_memory(memory.id, workspace_id)
            assert stored is not None
            assert services.index_sync(stored) is False
            reloaded = services.memories.get_memory(memory.id, workspace_id)
            assert reloaded is not None
            assert reloaded.content.title == memory.content.title
            assert services.outbox.pending_count() >= 1
    finally:
        broken_client.close()
