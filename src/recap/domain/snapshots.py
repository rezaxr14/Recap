"""Run-level memory snapshots.

A snapshot freezes the set of memory versions that were present when a run
started. Retrieval tied to that run may only use snapshot members, so a memory
promoted while the run is in progress cannot silently change its context.

The snapshot is not a security bypass: live eligibility checks still run, so a
memory quarantined or retired after the snapshot is still rejected. A snapshot
can therefore only narrow the candidate set, never widen it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from recap.domain.enums import MemoryStatus
from recap.domain.runs import new_id


@dataclass(frozen=True)
class SnapshotEntry:
    memory_id: str
    memory_version: int
    content_hash: str
    status: MemoryStatus


@dataclass(frozen=True)
class MemorySnapshot:
    id: str
    workspace_id: str
    created_at: datetime
    index_generation: int
    embedding_model_id: str
    entries: tuple[SnapshotEntry, ...] = ()
    reason: str = "run_start"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def entry_map(self) -> dict[str, SnapshotEntry]:
        return {entry.memory_id: entry for entry in self.entries}


def new_snapshot_id() -> str:
    return new_id("snap")


def utc_now() -> datetime:
    return datetime.now(UTC)
