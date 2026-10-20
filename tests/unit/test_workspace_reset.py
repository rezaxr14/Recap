"""Workspace reset must clear both stores or repeat runs silently drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from tests.support import make_memory

from recap.config import load_settings
from recap.container import Container
from recap.domain.chunks import MemoryChunk
from recap.domain.enums import ChunkType
from recap.infrastructure.db.session import Database, build_engine
from recap.infrastructure.embeddings.fake import FakeEmbeddingProvider
from recap.maintenance import reset_workspace
from recap.repair_compiler.validation import build_default_registry

WORKSPACE = "ws-reset-unit"


class FakeElasticsearch:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def delete_by_query(
        self, *, index: str, query: dict[str, Any], refresh: bool, conflicts: str
    ) -> dict[str, int]:
        assert query["terms"]["workspace_id"] == [WORKSPACE]
        assert refresh is True
        self.calls.append(index)
        return {"deleted": 4}


@dataclass
class FakeIndexManager:
    read_alias: str = "memories-read"
    write_index: str = "memories-v1"
    chunks_read_alias: str = "chunks-read"
    chunks_write_index: str = "chunks-v1"


def _container() -> tuple[Container, FakeElasticsearch]:
    settings = load_settings()
    database = Database(build_engine("sqlite+pysqlite:///:memory:"))
    database.create_all()
    es = FakeElasticsearch()
    container = Container(
        settings=settings,
        database=database,
        es_client=cast(Any, es),
        embeddings=FakeEmbeddingProvider(dimension=settings.embeddings.dimension),
        index_manager=cast(Any, FakeIndexManager()),
        search_repository=cast(Any, None),
        validators=build_default_registry(),
    )
    return container, es


def _seed(container: Container) -> None:
    memory = make_memory(
        container,
        workspace_id=WORKSPACE,
        title="Recover from INVALID_CURSOR",
        summary="Restart pagination from the first page.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
        index=False,
    )
    chunk = MemoryChunk(
        chunk_id="chk_reset_unit",
        workspace_id=WORKSPACE,
        memory_id=memory.id,
        memory_version=memory.memory_version,
        chunk_type=ChunkType.MEMORY_CARD,
        seq_start=0,
        seq_end=0,
        text="Tool: list_records | Error: INVALID_CURSOR",
        text_hash="sha256:reset-unit",
        token_estimate=10,
        chunker_version="1",
    )
    with container.services() as services:
        services.chunk_store.upsert_many([chunk], active_for_retrieval=True)


def test_reset_clears_memories_chunks_and_both_indices() -> None:
    container, es = _container()
    _seed(container)

    with container.services() as services:
        assert services.memories.list_memories(WORKSPACE)
        assert services.chunk_store.list_for_workspace(WORKSPACE)

    result = reset_workspace(container, [WORKSPACE])

    assert result["postgres"]["memory_chunks"] == 1
    assert result["postgres"]["memories"] == 1
    assert set(es.calls) == {"memories-read", "chunks-read"}
    with container.services() as services:
        assert services.memories.list_memories(WORKSPACE) == []
        assert services.chunk_store.list_for_workspace(WORKSPACE) == []


def test_reset_is_idempotent() -> None:
    container, es = _container()
    _seed(container)
    reset_workspace(container, [WORKSPACE])
    es.calls.clear()

    second = reset_workspace(container, [WORKSPACE])

    assert second["postgres"]["memory_chunks"] == 0
    assert set(es.calls) == {"memories-read", "chunks-read"}


def test_reset_with_no_workspaces_does_nothing() -> None:
    container, es = _container()
    result = reset_workspace(container, [])
    assert result == {"workspace_ids": [], "postgres": {}, "elasticsearch": {}}
    assert es.calls == []
