"""Zero-downtime memory index generation rebuild.

The old index stays behind the read alias while a new concrete index is
created, populated from Postgres, and refreshed. A single atomic alias update
then moves both the read and write aliases. If anything fails before that
update, the aliases are untouched and the partial target index is deleted.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from elasticsearch import exceptions as es_exceptions

from recap.config import Settings
from recap.container import Container
from recap.domain.memories import ExperienceMemory, build_embedding_text
from recap.errors import ConfigurationError, SearchUnavailable
from recap.infrastructure.db.repositories import SqlMemoryRepository
from recap.infrastructure.elasticsearch.client import TRANSPORT_ERRORS
from recap.infrastructure.elasticsearch.mappings import memory_index_body
from recap.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReindexResult:
    source_index: str | None
    target_index: str
    generation: int
    vector_dimension: int
    memories: int
    batches: int
    elapsed_ms: int
    old_indices: list[str] = field(default_factory=list)
    deleted_old: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "generation": self.generation,
            "vector_dimension": self.vector_dimension,
            "memories": self.memories,
            "batches": self.batches,
            "elapsed_ms": self.elapsed_ms,
            "old_indices": self.old_indices,
            "deleted_old": self.deleted_old,
        }


class MemoryReindexService:
    def __init__(self, container: Container) -> None:
        self.container = container
        self.settings: Settings = container.settings
        self.client = container.es_client
        self.index_manager = container.index_manager
        self.search = container.search_repository
        self.embeddings = container.embeddings

    def reindex(
        self,
        generation: int,
        *,
        workspace_id: str | None = None,
        include_suspect: bool = True,
        delete_old: bool = False,
        batch_size: int | None = None,
    ) -> ReindexResult:
        if generation < 1:
            raise ConfigurationError("target generation must be >= 1")
        prefix = self.settings.elasticsearch.index_prefix
        target_index = f"{prefix}-memories-v{generation}"
        source_index = self.index_manager.active_index_name()
        if source_index == target_index:
            raise ConfigurationError(
                "target generation is already the active read/write index",
                details={"index": target_index},
            )
        if self.client.indices.exists(index=target_index):
            raise ConfigurationError(
                "target index already exists; choose a new generation or delete it explicitly",
                details={"index": target_index},
            )

        dimension = self._validate_dimension()
        started = time.perf_counter()
        meta = {
            "embedding_fingerprint": getattr(self.embeddings, "embedding_fingerprint", ""),
            "embedding_model_id": getattr(self.embeddings, "model_id", ""),
            "vector_dimension": dimension,
            "index_generation": generation,
        }
        try:
            self.client.indices.create(
                index=target_index,
                **memory_index_body(dimension=dimension, meta=meta),
            )
        except es_exceptions.RequestError as exc:
            raise SearchUnavailable(
                "Elasticsearch rejected the new index definition",
                details={"error": str(exc.info) if hasattr(exc, "info") else str(exc)},
            ) from exc

        swapped = False
        try:
            memories = self._load_memories(workspace_id, include_suspect)
            batch_size = batch_size or self.settings.embeddings.batch_size
            batches = 0
            for offset in range(0, len(memories), batch_size):
                batch = memories[offset : offset + batch_size]
                vectors = self.embeddings.embed_documents(  # type: ignore[attr-defined]
                    [build_embedding_text(memory) for memory in batch]
                )
                if len(vectors) != len(batch):
                    raise ConfigurationError(
                        "embedding provider returned the wrong number of vectors",
                        details={"expected": len(batch), "received": len(vectors)},
                    )
                self.search.bulk_index(
                    list(zip(batch, vectors, strict=True)),
                    index=target_index,
                )
                batches += 1

            self.client.indices.refresh(index=target_index)
            indexed = int(self.client.count(index=target_index)["count"])
            if indexed != len(memories):
                raise SearchUnavailable(
                    "reindex count does not match the Postgres source of truth",
                    details={"postgres": len(memories), "elasticsearch": indexed},
                )

            old_indices = self._alias_indices(
                self.settings.elasticsearch.memories_read_alias,
                self.settings.elasticsearch.memories_write_alias,
            )
            self._swap_aliases(target_index, old_indices)
            swapped = True
            self.index_manager.set_active_index(target_index)
            try:
                self._mark_database_generation(memories, generation)
            except Exception:
                # The alias swap and DB provenance update should be one logical
                # operation. If the DB write fails, put reads/writes back on the
                # old index before surfacing the error.
                self._swap_aliases(source_index, {target_index})
                self.index_manager.set_active_index(source_index or self.index_manager.write_index)
                raise

            deleted: list[str] = []
            if delete_old:
                for index in old_indices:
                    if index != target_index:
                        self.client.indices.delete(index=index, ignore_unavailable=True)
                        deleted.append(index)

            elapsed = int((time.perf_counter() - started) * 1000)
            logger.info(
                "memory index reindexed",
                extra={
                    "event": "reindex_completed",
                    "source_index": source_index,
                    "target_index": target_index,
                    "memory_count": len(memories),
                },
            )
            return ReindexResult(
                source_index=source_index,
                target_index=target_index,
                generation=generation,
                vector_dimension=dimension,
                memories=len(memories),
                batches=batches,
                elapsed_ms=elapsed,
                old_indices=sorted(old_indices),
                deleted_old=deleted,
            )
        except Exception:
            if swapped:
                # A post-swap failure is rare; leave the new active index in
                # place if it is already serving, but never delete it silently.
                raise
            with contextlib.suppress(Exception):
                self.client.indices.delete(index=target_index, ignore_unavailable=True)
            raise

    def _validate_dimension(self) -> int:
        expected = self.settings.embeddings.dimension
        mapping_dimension = self.settings.elasticsearch.vector_dimension
        if expected != mapping_dimension:
            raise ConfigurationError(
                "embeddings.dimension and elasticsearch.vector_dimension must match",
                details={"embedding": expected, "mapping": mapping_dimension},
            )
        probe = self.embeddings.embed_query("recap reindex dimension probe")  # type: ignore[attr-defined]
        if len(probe) != expected:
            raise ConfigurationError(
                "embedding provider returned an unexpected dimension",
                details={
                    "expected": expected,
                    "received": len(probe),
                    "provider": self.settings.embeddings.provider,
                    "model": self.settings.embeddings.model_id,
                },
            )
        return expected

    def _load_memories(
        self, workspace_id: str | None, include_suspect: bool
    ) -> list[ExperienceMemory]:
        with self.container.database.session() as session:
            return SqlMemoryRepository(session).list_for_reindex(
                workspace_id=workspace_id,
                include_suspect=include_suspect,
            )

    def _mark_database_generation(self, memories: list[ExperienceMemory], generation: int) -> None:
        if not memories:
            return
        with self.container.database.session() as session:
            repository = SqlMemoryRepository(session)
            for memory in memories:
                repository.mark_indexed(memory.id, generation, success=True)

    def _alias_indices(self, *aliases: str) -> set[str]:
        indices: set[str] = set()
        for alias in aliases:
            try:
                indices.update(dict(self.client.indices.get_alias(name=alias)))
            except es_exceptions.NotFoundError:
                continue
            except TRANSPORT_ERRORS as exc:
                raise SearchUnavailable("cannot read Elasticsearch aliases") from exc
        return indices

    def _swap_aliases(self, target_index: str | None, old_indices: set[str]) -> None:
        read_alias = self.settings.elasticsearch.memories_read_alias
        write_alias = self.settings.elasticsearch.memories_write_alias
        actions: list[dict[str, Any]] = []
        for index in sorted(old_indices):
            for alias in (read_alias, write_alias):
                actions.append({"remove": {"index": index, "alias": alias}})
        if target_index is not None:
            for alias in (read_alias, write_alias):
                actions.append({"add": {"index": target_index, "alias": alias}})
        if not actions:
            return
        try:
            self.client.indices.update_aliases(actions=actions)
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot swap Elasticsearch aliases") from exc
        except es_exceptions.RequestError as exc:
            raise SearchUnavailable(
                "Elasticsearch rejected the alias swap",
                details={"error": str(exc.info) if hasattr(exc, "info") else str(exc)},
            ) from exc
