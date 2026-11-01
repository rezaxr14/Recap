"""Run snapshots are the boundary that makes held-out conditions honest."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from recap.domain.enums import MemoryStatus
from recap.domain.snapshots import MemorySnapshot
from recap.recall.snapshots import MemorySnapshotService


@dataclass
class _Memory:
    id: str
    memory_version: int = 1
    content_hash: str = ""
    status: MemoryStatus = MemoryStatus.ACTIVE


class _Memories:
    def __init__(self, memories: list[_Memory]) -> None:
        self._memories = memories
        self.limit = 0

    def list_for_snapshot(self, workspace_id: str, *, limit: int) -> list[_Memory]:
        self.limit = limit
        return list(self._memories)


class _Snapshots:
    def __init__(self) -> None:
        self.created: list[MemorySnapshot] = []

    def create(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        self.created.append(snapshot)
        return snapshot

    def get(self, snapshot_id: str, workspace_id: str) -> MemorySnapshot | None:
        return next((item for item in self.created if item.id == snapshot_id), None)


def _service(memories: list[_Memory]) -> tuple[MemorySnapshotService, _Snapshots]:
    snapshots = _Snapshots()
    service = MemorySnapshotService(
        memory_repository=_Memories(memories),
        snapshot_repository=snapshots,
        settings=SimpleNamespace(elasticsearch=SimpleNamespace(index_generation=3)),
        embedding_provider=SimpleNamespace(model_id="fixture-model"),
    )
    return service, snapshots


def test_create_excludes_requested_memory_ids() -> None:
    service, snapshots = _service(
        [
            _Memory("mem_a", content_hash="a"),
            _Memory("mem_b", content_hash="b"),
            _Memory("mem_c", content_hash="c"),
        ]
    )
    snapshot = service.create(
        "ws", reason="held_out", exclude_memory_ids={"mem_b"}
    )
    assert snapshot.reason == "held_out"
    assert [entry.memory_id for entry in snapshot.entries] == ["mem_a", "mem_c"]
    assert snapshot.metadata["excluded_memory_ids"] == ["mem_b"]
    assert snapshots.created == [snapshot]
    assert snapshot.index_generation == 3
    assert snapshot.embedding_model_id == "fixture-model"


def test_create_without_exclusions_keeps_every_entry() -> None:
    service, _ = _service([_Memory("mem_a"), _Memory("mem_b")])
    snapshot = service.create("ws")
    assert [entry.memory_id for entry in snapshot.entries] == ["mem_a", "mem_b"]
    assert snapshot.metadata["excluded_memory_ids"] == []
