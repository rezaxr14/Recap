"""Chunk materialization, indexing, lifecycle, and integrity against real services."""

from __future__ import annotations

import pytest
from tests.support import context_for, seed_promoted_memory

from recap.domain.chunks import MemoryChunk
from recap.domain.enums import ChunkType, MemoryStatus, RetrievalMode
from recap.domain.retrieval import RetrievalRequest
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.errors import ConfigurationError
from recap.infrastructure.db.models import MemoryChunkRow
from recap.infrastructure.elasticsearch.chunk_repository import (
    ElasticsearchChunkRepository,
)
from recap.infrastructure.elasticsearch.index_manager import IndexManager
from recap.infrastructure.elasticsearch.mappings import chunk_index_body
from recap.maintenance import prune_orphan_chunks, reset_workspace

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

    # Rebuilding is idempotent: same chunk ids, same count, and the stored
    # vector identity means an unchanged chunk is not re-embedded.
    embedded_batches: list[int] = []
    original_embed = container.embeddings.embed_documents

    def counting_embed(texts):  # type: ignore[no-untyped-def]
        embedded_batches.append(len(texts))
        return original_embed(texts)

    with container.services() as services:
        services.chunk_service.sync.embeddings.embed_documents = counting_embed
        second = services.chunk_service.rebuild(memory)
        after = services.chunk_store.list_for_memory(memory.id, memory.memory_version)
    assert second.indexed
    assert {chunk.chunk_id for chunk in after} == {chunk.chunk_id for chunk in chunks}
    assert embedded_batches == [], "unchanged chunks must reuse their stored embeddings"

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


def test_orphan_chunk_hits_are_excluded_and_pruned(container, workspace_id: str) -> None:
    environment = PaginatedApiEnvironment(version="2")
    fingerprint = environment.fingerprint()
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)

    _candidate, memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="2"
    )
    with container.services() as services:
        assert services.chunk_service.rebuild(memory).indexed

    orphan = MemoryChunk(
        chunk_id="chk_orphan_parent_gone",
        workspace_id=workspace_id,
        memory_id="mem_parent_was_deleted",
        memory_version=1,
        chunk_type=ChunkType.FAILURE_WINDOW,
        seq_start=1,
        seq_end=2,
        event_ids=["evt-orphan"],
        text="Action: list_records\nError: INVALID_CURSOR cursor is invalid",
        text_hash="sha256:orphan",
        token_estimate=12,
        chunker_version=container.settings.chunking.version,
        environment_fingerprint=fingerprint.fingerprint,
    )
    embedding = container.embeddings.embed_documents([orphan.text])
    container.chunk_search_repository.index_chunks([orphan], embedding, active_for_retrieval=True)
    container.es_client.indices.refresh(index=container.index_manager.chunks_write_index)

    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(
                    workspace_id=workspace_id,
                    environment_version="2",
                    query="INVALID_CURSOR cursor is invalid",
                ),
                mode=RetrievalMode.BM25_CHUNK,
                final_k=3,
            )
        )

    assert decision.config_snapshot["orphan_chunk_hits"] >= 1
    assert any("orphan_chunk_hits" in warning for warning in decision.warnings)
    assert all(candidate.memory_id != orphan.memory_id for candidate in decision.candidates)

    pruned = prune_orphan_chunks(container, workspace_id)
    assert orphan.chunk_id in pruned["orphan_ids"]
    assert pruned["orphans_removed"] >= 1
    container.es_client.indices.refresh(index=container.index_manager.chunks_write_index)
    response = container.es_client.search(
        index=container.index_manager.chunks_read_alias,
        query={"term": {"chunk_id": orphan.chunk_id}},
        size=10,
    )
    assert response["hits"]["hits"] == []


def test_embedding_fingerprint_isolation_in_queries(container, workspace_id: str) -> None:
    environment = PaginatedApiEnvironment(version="2")
    fingerprint = environment.fingerprint()
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)

    _candidate, memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="2"
    )
    with container.services() as services:
        assert services.chunk_service.rebuild(memory).indexed

    other = ElasticsearchChunkRepository(
        container.es_client,
        container.settings.elasticsearch,
        embedding_fingerprint="sha256:different-model",
        embedding_model_id="different-model",
    )
    payload = other.search(
        context=context_for(workspace_id=workspace_id, environment_version="2"),
        mode="bm25_chunk",
        candidate_k=10,
    )

    assert payload["hits"] == {}
    assert payload["degraded"] is True
    assert any("embedding fingerprint" in warning for warning in payload["warnings"])


def test_rebuild_reembeds_chunks_with_a_stale_fingerprint(container, workspace_id: str) -> None:
    environment = PaginatedApiEnvironment(version="2")
    fingerprint = environment.fingerprint()
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)

    _candidate, memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="2"
    )
    with container.services() as services:
        assert services.chunk_service.rebuild(memory).indexed

    # Simulate chunks indexed by an older embedding model. Reconcile/rebuild
    # must not trust ``index_status='indexed'`` without comparing the identity.
    with container.services() as services:
        for chunk in services.chunk_store.list_for_memory(memory.id, memory.memory_version):
            row = services.session.get(MemoryChunkRow, chunk.chunk_id)
            assert row is not None
            row.embedding_fingerprint = "sha256:old-model"
            row.index_status = "indexed"
        services.session.flush()

    # The embedding cache is content-addressed and would correctly return the
    # vector from the first rebuild. Clear it here so this test observes the
    # provider call that a cold cache must make for a stale fingerprint.
    if container.embedding_cache is not None:
        container.embedding_cache.clear()

    embedded_batches: list[int] = []
    with container.services() as services:
        original_embed = services.chunk_service.sync.embeddings.embed_documents

        def counting_embed(texts):  # type: ignore[no-untyped-def]
            embedded_batches.append(len(texts))
            return original_embed(texts)

        services.chunk_service.sync.embeddings.embed_documents = counting_embed
        report = services.chunk_service.rebuild(memory)
        after = services.chunk_store.list_for_memory(memory.id, memory.memory_version)

    assert report.indexed, report.error
    assert embedded_batches and sum(embedded_batches) == len(after)
    assert all(
        chunk.embedding_fingerprint == container.chunk_search_repository.embedding_fingerprint
        for chunk in after
    )


def test_empty_chunk_index_is_labelled_degraded(container, workspace_id: str) -> None:
    payload = container.chunk_search_repository.search(
        context=context_for(workspace_id=workspace_id, environment_version="2"),
        mode="bm25_chunk",
        candidate_k=5,
    )

    assert payload["hits"] == {}
    assert payload["degraded"] is True
    assert any("chunk_index_empty" in warning for warning in payload["warnings"])


def test_existing_chunk_index_with_wrong_dimension_fails_fast(container) -> None:
    settings = container.settings.elasticsearch.model_copy(deep=True)
    settings.chunks_index_generation = 991
    settings.chunks_read_alias = "recap-test-chunks-read-991"
    settings.chunks_write_alias = "recap-test-chunks-write-991"
    index = settings.chunks_write_index
    container.es_client.indices.create(
        index=index, **chunk_index_body(dimension=8, meta={"embedding_fingerprint": "sha256:x"})
    )
    try:
        manager = IndexManager(container.es_client, settings, embedding_fingerprint="sha256:x")
        with pytest.raises(ConfigurationError) as excinfo:
            manager.initialize_chunks()
        assert "dimension" in excinfo.value.message
    finally:
        container.es_client.indices.delete(index=index, ignore_unavailable=True)


def test_existing_chunk_index_with_wrong_fingerprint_fails_fast(container) -> None:
    settings = container.settings.elasticsearch.model_copy(deep=True)
    settings.chunks_index_generation = 992
    settings.chunks_read_alias = "recap-test-chunks-read-992"
    settings.chunks_write_alias = "recap-test-chunks-write-992"
    index = settings.chunks_write_index
    container.es_client.indices.create(
        index=index,
        **chunk_index_body(
            dimension=settings.vector_dimension,
            meta={"embedding_fingerprint": "sha256:old-model"},
        ),
    )
    try:
        manager = IndexManager(
            container.es_client, settings, embedding_fingerprint="sha256:new-model"
        )
        with pytest.raises(ConfigurationError) as excinfo:
            manager.initialize_chunks()
        assert "fingerprint" in excinfo.value.message
    finally:
        container.es_client.indices.delete(index=index, ignore_unavailable=True)


def test_reset_workspace_clears_chunk_rows_and_documents_twice(
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
        assert services.chunk_store.list_for_workspace(workspace_id)
    container.es_client.indices.refresh(index=container.index_manager.chunks_write_index)
    assert container.chunk_search_repository.count(workspace_id=workspace_id) > 0

    first = reset_workspace(container, [workspace_id])
    second = reset_workspace(container, [workspace_id])

    assert first["postgres"]["memory_chunks"] > 0
    assert second["postgres"]["memory_chunks"] == 0
    with container.services() as services:
        assert services.chunk_store.list_for_workspace(workspace_id) == []
        assert services.memories.list_memories(workspace_id) == []
    assert container.chunk_search_repository.count(workspace_id=workspace_id) == 0
