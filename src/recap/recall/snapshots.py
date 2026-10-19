"""Create and resolve run-level memory snapshots."""

from __future__ import annotations

from typing import Any, cast

from recap.config import Settings
from recap.domain.enums import MemoryStatus
from recap.domain.snapshots import (
    MemorySnapshot,
    SnapshotEntry,
    new_snapshot_id,
    utc_now,
)
from recap.errors import ValidationError

MAX_SNAPSHOT_ENTRIES = 10_000


class MemorySnapshotService:
    def __init__(
        self,
        *,
        memory_repository: Any,
        snapshot_repository: Any,
        settings: Settings,
        embedding_provider: Any,
    ) -> None:
        self.memories = memory_repository
        self.snapshots = snapshot_repository
        self.settings = settings
        self.embeddings = embedding_provider

    def create(self, workspace_id: str, *, reason: str = "run_start") -> MemorySnapshot:
        memories = self.memories.list_for_snapshot(workspace_id, limit=MAX_SNAPSHOT_ENTRIES + 1)
        if len(memories) > MAX_SNAPSHOT_ENTRIES:
            raise ValidationError(
                f"workspace {workspace_id!r} has more than {MAX_SNAPSHOT_ENTRIES} "
                "memories; refusing to create an unbounded run snapshot"
            )
        snapshot = MemorySnapshot(
            id=new_snapshot_id(),
            workspace_id=workspace_id,
            created_at=utc_now(),
            index_generation=self.settings.elasticsearch.index_generation,
            embedding_model_id=str(getattr(self.embeddings, "model_id", "")),
            entries=tuple(
                SnapshotEntry(
                    memory_id=memory.id,
                    memory_version=memory.memory_version,
                    content_hash=memory.content_hash,
                    status=memory.status
                    if isinstance(memory.status, MemoryStatus)
                    else MemoryStatus(memory.status),
                )
                for memory in memories
            ),
            reason=reason,
            metadata={"max_entries": MAX_SNAPSHOT_ENTRIES},
        )
        return cast(MemorySnapshot, self.snapshots.create(snapshot))

    def get(self, snapshot_id: str, workspace_id: str) -> MemorySnapshot | None:
        return cast(MemorySnapshot | None, self.snapshots.get(snapshot_id, workspace_id))
