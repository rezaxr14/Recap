"""Chunk materialization, indexing, and lifecycle against real services."""

from __future__ import annotations

import pytest
from tests.support import context_for, seed_promoted_memory

from recap.domain.enums import MemoryStatus, RetrievalMode
from recap.domain.retrieval import RetrievalRequest
from recap.environments.mock_api import PaginatedApiEnvironment

pytestmark = pytest.mark.integration


def test_promoted_memory_chunks_are_built_indexed_and_deterministic(
    container, workspace_id: str
) -> None:
    environment = PaginatedApiEnvironment(version="2")
    fingerprint = environment.fingerprint()
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)

    _candidate, memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="2"
    )
    assert memory.status is MemoryStatus.ACTIVE

    with container.services() as services:
        report = services.chunk_service.rebuild(memory)
        assert report.indexed, report.error
        chunks = services.chunk_store.list_for_memory(memory.id, memory.memory_version)

    assert chunks
    types = {str(chunk.chunk_type) for chunk in chunks}
    assert "memory_card" in types
    assert "failure_window" in types
    card = next(chunk for chunk in chunks if str(chunk.chunk_type) == "memory_card")
    assert "Strategy:" in card.text
    assert all(chunk.token_estimate > 0 for chunk in chunks)
    assert all(chunk.chunker_version == container.settings.chunking.version for chunk in chunks)

    # Rebuilding is idempotent: same chunk ids, same count.
    with container.services() as services:
        second = services.chunk_service.rebuild(memory)
        after = services.chunk_store.list_for_memory(memory.id, memory.memory_version)
    assert second.indexed
    assert {chunk.chunk_id for chunk in after} == {chunk.chunk_id for chunk in chunks}

    container.es_client.indices.refresh(index=container.index_manager.chunks_write_index)
    response = container.es_client.search(
        index=container.index_manager.chunks_read_alias,
        query={"term": {"memory_id": memory.id}},
        size=100,
    )
    documents = response["hits"]["hits"]
    assert len(documents) == len(chunks)
    assert all(hit["_source"]["active_for_retrieval"] is True for hit in documents)
    assert all(
        hit["_source"]["embedding_fingerprint"]
        == container.chunk_search_repository.embedding_fingerprint
        for hit in documents
    )

    # Lifecycle follows the memory: retiring deactivates chunk documents.
    with container.services() as services:
        current = services.memories.get_memory(memory.id)
        assert current is not None
        retired = current.transition(MemoryStatus.RETIRED, "chunk lifecycle test")
        services.memories.update_status(retired)
        assert services.index_sync(retired) is True

    container.es_client.indices.refresh(index=container.index_manager.chunks_write_index)
    response = container.es_client.search(
        index=container.index_manager.chunks_read_alias,
        query={"term": {"memory_id": memory.id}},
        size=100,
    )
    documents = response["hits"]["hits"]
    assert documents
    assert all(hit["_source"]["active_for_retrieval"] is False for hit in documents)


def test_chunk_level_retrieval_selects_parent_memory_with_evidence(
    container, workspace_id: str
) -> None:
    environment = PaginatedApiEnvironment(version="2")
    fingerprint = environment.fingerprint()
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)

    _candidate, memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="2"
    )
    with container.services() as services:
        assert services.chunk_service.rebuild(memory).indexed
        container.es_client.indices.refresh(index=container.index_manager.chunks_write_index)
        request = RetrievalRequest(
            context=context_for(workspace_id=workspace_id, environment_version="2"),
            mode=RetrievalMode.OUTCOME_AWARE_CHUNK,
            final_k=3,
        )
        decision = services.recall.retrieve(request)
        baseline_request = RetrievalRequest(
            context=context_for(workspace_id=workspace_id, environment_version="2"),
            mode=RetrievalMode.OUTCOME_AWARE,
            final_k=3,
        )
        baseline = services.recall.retrieve(baseline_request)

    assert decision.status.value == "selected"
    selected = decision.selected_memories[0]
    assert selected.memory_id == memory.id
    assert selected.retrieval_granule == "chunk"
    assert selected.supporting_chunk_ids
    assert selected.supporting_event_ids
    assert decision.retrieval_granule == "chunk"
    assert decision.config_snapshot["chunker_version"] == container.settings.chunking.version

    # The card baseline still works and is labeled as card retrieval.
    assert baseline.status.value == "selected"
    assert baseline.selected_memories[0].retrieval_granule == "memory_card"
    assert baseline.selected_memories[0].memory_id == memory.id
