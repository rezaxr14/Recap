"""Memory index generation rebuild against real Postgres and Elasticsearch."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from tests.support import make_memory

from recap.config import Settings
from recap.container import Container
from recap.errors import ConfigurationError
from recap.infrastructure.db.models import MemoryRow
from recap.reindex import MemoryReindexService

pytestmark = pytest.mark.integration


def _isolated_settings(live_settings: Settings) -> Settings:
    settings = live_settings.model_copy(deep=True)
    suffix = uuid.uuid4().hex[:10]
    settings.elasticsearch.index_prefix = f"recap-test-reindex-{suffix}"
    settings.elasticsearch.memories_read_alias = f"recap-test-reindex-{suffix}-read"
    settings.elasticsearch.memories_write_alias = f"recap-test-reindex-{suffix}-write"
    settings.elasticsearch.chunks_read_alias = f"recap-test-reindex-{suffix}-chunks-read"
    settings.elasticsearch.chunks_write_alias = f"recap-test-reindex-{suffix}-chunks-write"
    return settings


@pytest.fixture()
def isolated_container(live_settings: Settings) -> Iterator[tuple[Container, Settings]]:
    settings = _isolated_settings(live_settings)
    container = Container.build(settings)
    container.migrate()
    container.init_search()
    try:
        yield container, settings
    finally:
        existing = container.es_client.indices.get(
            index=f"{settings.elasticsearch.index_prefix}-*",
            ignore_unavailable=True,
        )
        for name in existing:
            container.es_client.indices.delete(index=name, ignore_unavailable=True)
        container.es_client.close()


def test_reindex_populates_new_generation_and_atomically_swaps_aliases(
    isolated_container: tuple[Container, Settings], workspace_id: str
) -> None:
    container, settings = isolated_container
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Refresh invalid cursor",
        summary="Refresh the cursor after INVALID_CURSOR.",
        failure_signature="ValidationError INVALID_CURSOR",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    old_index = container.index_manager.active_index_name()
    assert old_index is not None

    result = MemoryReindexService(container).reindex(2, workspace_id=workspace_id)

    target = f"{settings.elasticsearch.index_prefix}-memories-v2"
    assert result.source_index == old_index
    assert result.target_index == target
    assert result.memories == 1
    assert result.vector_dimension == settings.elasticsearch.vector_dimension
    assert container.index_manager.active_index_name() == target
    assert container.es_client.count(index=target)["count"] == 1

    read_aliases = container.es_client.indices.get_alias(
        name=settings.elasticsearch.memories_read_alias
    )
    write_aliases = container.es_client.indices.get_alias(
        name=settings.elasticsearch.memories_write_alias
    )
    assert set(read_aliases) == {target}
    assert set(write_aliases) == {target}
    with container.services() as services:
        stored = services.memories.get_memory(memory.id)
        row = services.session.get(MemoryRow, memory.id)
    assert stored is not None
    assert stored.index_generation == 2
    assert row is not None
    assert row.index_status == "indexed"

    # A fresh process may still have the bootstrap generation (v1) in its
    # static config. The alias must remain authoritative, not re-point to v1.
    reopened = Container.build(settings)
    try:
        state = reopened.init_search()
        assert reopened.index_manager.active_index_name() == target
        assert reopened.index_manager.write_target == target
        assert state["index"] == target
        read_aliases = reopened.es_client.indices.get_alias(
            name=settings.elasticsearch.memories_read_alias
        )
        assert set(read_aliases) == {target}
    finally:
        reopened.es_client.close()


def test_reindex_failure_leaves_aliases_and_source_index_untouched(
    isolated_container: tuple[Container, Settings], workspace_id: str, monkeypatch
) -> None:
    container, settings = isolated_container
    make_memory(
        container,
        workspace_id=workspace_id,
        title="Refresh invalid cursor",
        summary="Refresh the cursor after INVALID_CURSOR.",
        failure_signature="ValidationError INVALID_CURSOR",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    old_index = container.index_manager.active_index_name()
    assert old_index is not None
    target = f"{settings.elasticsearch.index_prefix}-memories-v2"

    def fail_embed(_texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding outage during reindex")

    monkeypatch.setattr(container.embeddings, "embed_documents", fail_embed)

    with pytest.raises(RuntimeError, match="embedding outage"):
        MemoryReindexService(container).reindex(2, workspace_id=workspace_id)

    assert container.index_manager.active_index_name() == old_index
    assert not container.es_client.indices.exists(index=target)
    read_aliases = container.es_client.indices.get_alias(
        name=settings.elasticsearch.memories_read_alias
    )
    assert set(read_aliases) == {old_index}


def test_reindex_rejects_wrong_embedding_dimension_before_creating_index(
    isolated_container: tuple[Container, Settings], workspace_id: str, monkeypatch
) -> None:
    container, settings = isolated_container
    target = f"{settings.elasticsearch.index_prefix}-memories-v2"
    monkeypatch.setattr(container.embeddings, "embed_query", lambda _text: [0.0, 0.0, 0.0])

    with pytest.raises(ConfigurationError, match="unexpected dimension"):
        MemoryReindexService(container).reindex(2, workspace_id=workspace_id)

    assert not container.es_client.indices.exists(index=target)
