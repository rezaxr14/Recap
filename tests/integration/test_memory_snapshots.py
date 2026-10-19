"""Run-level memory snapshot pinning against real Postgres and Elasticsearch."""

from __future__ import annotations

import uuid

import pytest
from tests.support import context_for, make_memory

from recap.domain.enums import RetrievalMode
from recap.domain.retrieval import RetrievalRequest
from recap.domain.runs import AgentRun
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.errors import NotFoundError, ValidationError
from recap.infrastructure.db.models import MemoryRow

pytestmark = pytest.mark.integration


def _start_run(container, workspace_id: str, version: str = "1") -> AgentRun:
    environment = PaginatedApiEnvironment(version=version)
    fingerprint = environment.fingerprint()
    with container.services() as services:
        run = services.recorder.start_run(
            AgentRun(
                workspace_id=workspace_id,
                agent_id="snapshot-test-agent",
                task_id=f"snapshot-{uuid.uuid4().hex[:8]}",
                task_family="api_pagination",
                environment_id=fingerprint.environment_id,
                environment_fingerprint=fingerprint.fingerprint,
                toolset_fingerprint=fingerprint.fingerprint,
            )
        )
        assert run.memory_snapshot_id is not None
        return run


def _request(workspace_id: str, *, run_id: str | None = None) -> RetrievalRequest:
    return RetrievalRequest(
        context=context_for(workspace_id=workspace_id, environment_version="1"),
        mode=RetrievalMode.OUTCOME_AWARE,
        final_k=3,
        run_id=run_id,
    )


def test_empty_snapshot_blocks_a_memory_promoted_after_run_start(
    container, workspace_id: str
) -> None:
    run = _start_run(container, workspace_id)
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Restart pagination",
        summary="Restart from the first page.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )

    with container.services() as services:
        snapshot = services.snapshots.get(run.memory_snapshot_id, workspace_id)
        assert snapshot is not None
        assert snapshot.entry_count == 0
        decision = services.recall.retrieve(_request(workspace_id, run_id=run.id))

        assert decision.memory_snapshot_id == run.memory_snapshot_id
        assert decision.selected_memories == []
        assert memory.id not in [item.memory_id for item in decision.selected_memories]
        rejected = {item.memory_id: item.reason for item in decision.rejected_candidates}
        assert rejected.get(memory.id) == "not_in_memory_snapshot"
        assert decision.config_snapshot["memory_snapshot_entries"] == 0


def test_snapshot_excludes_later_memories_but_keeps_frozen_members(
    container, workspace_id: str
) -> None:
    first = make_memory(
        container,
        workspace_id=workspace_id,
        title="Restart pagination (v1)",
        summary="Restart from the first page.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    run = _start_run(container, workspace_id)
    later = make_memory(
        container,
        workspace_id=workspace_id,
        title="Restart pagination (later promotion)",
        summary="A different active memory promoted after the run started.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )

    with container.services() as services:
        unpinned = services.recall.retrieve(_request(workspace_id))
        unpinned_ids = {item.memory_id for item in unpinned.candidates}
        assert {first.id, later.id} <= unpinned_ids

        pinned = services.recall.retrieve(_request(workspace_id, run_id=run.id))
        candidate_ids = {item.memory_id for item in pinned.candidates}
        assert first.id in candidate_ids
        assert later.id in candidate_ids
        assert [item.memory_id for item in pinned.selected_memories] == [first.id]
        rejected = {item.memory_id: item.reason for item in pinned.rejected_candidates}
        assert rejected.get(later.id) == "not_in_memory_snapshot"
        assert pinned.memory_snapshot_id == run.memory_snapshot_id


def test_snapshot_detects_content_hash_drift(container, workspace_id: str) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Restart pagination",
        summary="Restart from the first page.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    run = _start_run(container, workspace_id)

    with container.services() as services:
        row = services.session.get(MemoryRow, memory.id)
        assert row is not None
        row.content_hash = "sha256:deliberately-tampered"
        services.session.flush()

        decision = services.recall.retrieve(_request(workspace_id, run_id=run.id))
        assert decision.selected_memories == []
        rejected = {item.memory_id: item.reason for item in decision.rejected_candidates}
        assert rejected.get(memory.id) == "snapshot_content_mismatch"


def test_unknown_run_fails_closed(container, workspace_id: str) -> None:
    with container.services() as services, pytest.raises(NotFoundError):
        services.recall.retrieve(_request(workspace_id, run_id="run_does_not_exist"))


def test_explicit_snapshot_from_another_workspace_is_rejected(container, workspace_id: str) -> None:
    with container.services() as services:
        with pytest.raises(ValidationError) as excinfo:
            services.recorder.start_run(
                AgentRun(
                    workspace_id=workspace_id,
                    agent_id="snapshot-test-agent",
                    task_id="cross-workspace",
                    task_family="api_pagination",
                    environment_id="mock-api-v1",
                    environment_fingerprint="sha256:env",
                    memory_snapshot_id="snap_missing",
                )
            )
        assert "not found" in str(excinfo.value)
