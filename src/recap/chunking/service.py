"""Chunk materialization and synchronization.

Builds deterministic chunks for a memory from its evidence runs, persists them
to Postgres (authoritative), and indexes them through the outbox. Elasticsearch
failures leave retryable outbox work instead of losing durable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recap.chunking.chunker import build_chunks
from recap.config import ChunkingSettings, Settings
from recap.domain.chunks import MemoryChunk
from recap.domain.enums import ChunkType, MemoryStatus
from recap.domain.memories import ExperienceMemory
from recap.domain.runs import utc_now
from recap.errors import (
    EmbeddingUnavailable,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderUnavailable,
    SearchUnavailable,
)
from recap.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkBuildReport:
    memory_id: str
    memory_version: int
    chunks: list[MemoryChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    removed_stale: int = 0
    indexed: bool = False
    error: str = ""

    @property
    def count(self) -> int:
        return len(self.chunks)


class ChunkService:
    def __init__(
        self,
        *,
        settings: Settings,
        memory_repository: Any,
        run_repository: Any,
        repair_repository: Any,
        chunk_repository: Any,
        chunk_index_sync: Any,
    ) -> None:
        self.settings = settings
        self.memories = memory_repository
        self.runs = run_repository
        self.repairs = repair_repository
        self.chunks = chunk_repository
        self.sync = chunk_index_sync

    # ------------------------------------------------------------- building
    def build_for_memory(
        self,
        memory: ExperienceMemory,
        *,
        run_ids: list[str] | None = None,
        config: ChunkingSettings | None = None,
    ) -> ChunkBuildReport:
        report = ChunkBuildReport(memory_id=memory.id, memory_version=memory.memory_version)
        evidence_runs = run_ids if run_ids is not None else list(memory.evidence_run_ids)
        seen: dict[str, MemoryChunk] = {}
        built_at = utc_now()
        chunk_settings = config or self.settings.chunking

        def add(chunks: tuple[MemoryChunk, ...] | list[MemoryChunk]) -> None:
            for chunk in chunks:
                if chunk.chunk_id not in seen:
                    seen[chunk.chunk_id] = chunk

        events_by_run = {run_id: self.runs.list_events(run_id) for run_id in evidence_runs}
        for run_id, events in events_by_run.items():
            result = build_chunks(
                memory,
                events=events,
                run_id=run_id,
                config=chunk_settings,
                created_at=built_at,
            )
            add(result.chunks)
            report.warnings.extend(result.warnings)
        if not events_by_run:
            result = build_chunks(memory, config=chunk_settings, created_at=built_at)
            add(result.chunks)
            report.warnings.extend(result.warnings)

        validation_chunks: list[MemoryChunk] = []
        anchor_run = evidence_runs[-1] if evidence_runs else None
        for validation_id in memory.validation_record_ids:
            validation = self.repairs.get_validation(validation_id)
            if validation is None:
                report.warnings.append(f"validation record {validation_id} not found")
                continue
            result = build_chunks(
                memory,
                events=(),
                validation=validation,
                run_id=anchor_run,
                config=chunk_settings,
                created_at=built_at,
            )
            validation_chunks.extend(
                chunk
                for chunk in result.chunks
                if chunk.chunk_type is ChunkType.VALIDATION_EVIDENCE
            )
        add(validation_chunks)

        report.chunks = list(seen.values())
        stale = self.chunks.delete_stale(
            memory.id,
            memory.memory_version,
            set(seen),
            chunk_strategy=chunk_settings.strategy,
        )
        report.removed_stale = len(stale)
        if stale and getattr(self.sync, "search", None) is not None:
            try:
                self.sync.search.delete_chunks(stale)
            except SearchUnavailable as exc:
                # Orphan pruning in reconcile is the safety net; do not fail a
                # rebuild because the cleanup write failed.
                report.warnings.append(f"stale chunk documents not removed: {exc.message}")
        return report

    # ------------------------------------------------------------- indexing
    def rebuild(
        self,
        memory: ExperienceMemory,
        *,
        run_ids: list[str] | None = None,
        config: ChunkingSettings | None = None,
    ) -> ChunkBuildReport:
        report = self.build_for_memory(memory, run_ids=run_ids, config=config)
        active = memory.status is MemoryStatus.ACTIVE
        report.chunks = self.chunks.upsert_many(report.chunks, active_for_retrieval=active)
        report.indexed = self.sync.sync(memory, report.chunks, active=active)
        if not report.indexed:
            report.error = self.sync.last_error
        return report


class ChunkIndexSynchronizer:
    """Embed and index chunks, recording intent in the outbox first."""

    def __init__(
        self,
        chunk_search: Any,
        embeddings: Any,
        chunk_store: Any,
        outbox: Any,
        settings: Settings,
    ) -> None:
        self.search = chunk_search
        self.embeddings = embeddings
        self.store = chunk_store
        self.outbox = outbox
        self.settings = settings
        self.last_error = ""

    def sync(self, memory: ExperienceMemory, chunks: list[MemoryChunk], *, active: bool) -> bool:
        if not chunks:
            return True
        fingerprint = getattr(self.embeddings, "embedding_fingerprint", "")
        generation = self.settings.elasticsearch.chunks_index_generation
        # Content-addressed ids mean an unchanged chunk needs no re-embedding.
        # Skip only when the stored vector identity and visibility already match.
        pending = [
            chunk
            for chunk in chunks
            if not (
                chunk.index_status == "indexed"
                and chunk.embedding_fingerprint == fingerprint
                and chunk.indexed_generation == generation
                and chunk.active_for_retrieval == active
            )
        ]
        self.last_error = ""
        if not pending:
            return True
        for chunk in pending:
            self.outbox.enqueue(
                workspace_id=chunk.workspace_id,
                entity_id=chunk.chunk_id,
                operation="index",
                dedupe_key=f"index:chunk:{chunk.chunk_id}",
                entity_type="chunk",
            )
        try:
            embeddings = self.embeddings.embed_documents([chunk.text for chunk in pending])
        except (
            EmbeddingUnavailable,
            ProviderUnavailable,
            ProviderAuthenticationError,
            ProviderResponseError,
        ) as exc:
            self.last_error = exc.message
            for chunk in pending:
                self.outbox.mark_failed_by_key(f"index:chunk:{chunk.chunk_id}", exc.message)
            self.store.mark_indexed([chunk.chunk_id for chunk in pending], success=False)
            return False
        try:
            self.search.index_chunks(pending, embeddings, active_for_retrieval=active)
        except SearchUnavailable as exc:
            self.last_error = exc.message
            for chunk in pending:
                self.outbox.mark_failed_by_key(f"index:chunk:{chunk.chunk_id}", exc.message)
            self.store.mark_indexed([chunk.chunk_id for chunk in pending], success=False)
            return False
        ids = [chunk.chunk_id for chunk in pending]
        self.store.mark_indexed(
            ids,
            success=True,
            embedding_fingerprint=fingerprint,
            generation=generation,
        )
        for chunk_id in ids:
            self.outbox.mark_done_by_key(f"index:chunk:{chunk_id}")
        self.last_error = ""
        return True
