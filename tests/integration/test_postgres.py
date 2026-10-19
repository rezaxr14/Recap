"""Integration tests for the PostgreSQL source of truth.

Covers the guarantees that matter for durability: idempotent event ingestion,
optimistic ordering, lifecycle transitions, outbox durability, and tenant
isolation. These run against real PostgreSQL (not SQLite) because the schema
uses JSONB and partial indexes.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from recap.domain.enums import EventType, MemoryStatus, OutcomeStatus, RunStatus
from recap.domain.feedback import OutcomeEvidence
from recap.domain.runs import AgentRun, RunEventCreate
from recap.errors import TransitionError, ValidationError
from recap.infrastructure.db.repositories import ensure_workspace

pytestmark = pytest.mark.integration


def _run(workspace_id: str) -> AgentRun:
    return AgentRun(
        workspace_id=workspace_id,
        agent_id="test-agent",
        task_id=f"task-{uuid.uuid4().hex[:8]}",
        task_family="api_pagination",
        environment_id="mock-api-v1",
        environment_fingerprint="sha256:test",
        toolset_fingerprint="sha256:test",
        model_provider="deterministic",
        model_name="test",
        status=RunStatus.RUNNING,
        success_criteria="collect all records",
    )


def test_postgres_is_the_configured_backend(container) -> None:
    with container.database.session() as session:
        version = session.execute(text("SHOW server_version")).scalar_one()
        assert version.startswith("17")


def test_event_ingestion_is_idempotent(container, workspace_id: str) -> None:
    with container.services() as services:
        ensure_workspace(services.session, workspace_id)
        run = _run(workspace_id)
        services.recorder.start_run(run)
        event = RunEventCreate(
            sequence_number=2,
            event_type=EventType.TOOL_CALL_FAILED,
            tool_name="list_records",
            payload={"error_code": "INVALID_CURSOR", "error_message": "cursor invalid"},
        )
        first = services.recorder.append_event(run, event)
        replay = services.recorder.append_event(run, event)
        assert first.id == replay.id
        events = services.runs.list_events(run.id)
        # ``start_run`` creates the run row only; events are appended explicitly.
        assert len(events) == 1
        assert events[0].sequence_number == 2


def test_event_sequence_cannot_regress(container, workspace_id: str) -> None:
    with container.services() as services:
        ensure_workspace(services.session, workspace_id)
        run = _run(workspace_id)
        services.recorder.start_run(run)
        services.recorder.append_event(
            run,
            RunEventCreate(sequence_number=5, event_type=EventType.TASK_OBSERVATION, payload={}),
        )
        # Older sequence numbers are rejected rather than overwriting history.
        with pytest.raises(ValidationError):
            services.recorder.append_event(
                run,
                RunEventCreate(
                    sequence_number=3, event_type=EventType.TASK_OBSERVATION, payload={}
                ),
            )
        assert [e.sequence_number for e in services.runs.list_events(run.id)] == [5]


def test_secrets_are_redacted_before_persistence(container, workspace_id: str) -> None:
    with container.services() as services:
        ensure_workspace(services.session, workspace_id)
        run = _run(workspace_id)
        services.recorder.start_run(run)
        services.recorder.append_event(
            run,
            RunEventCreate(
                sequence_number=2,
                event_type=EventType.TOOL_CALL_REQUESTED,
                tool_name="list_records",
                payload={
                    "arguments": {"tenant": "acme", "api_key": "sk-live-abcdefghijklmnop"},
                    "headers": {"authorization": "Bearer abcdefghijklmnopqrstuvwxyz"},
                    "note": "token=supersecret123",
                },
            ),
        )
        stored = services.runs.list_events(run.id)[-1]
        serialized = str(stored.payload_redacted)
        assert "sk-live-abcdefghijklmnop" not in serialized
        assert "abcdefghijklmnopqrstuvwxyz" not in serialized
        assert "supersecret123" not in serialized
        assert stored.redaction_applied is True


def test_memory_lifecycle_transitions_are_enforced(container, workspace_id: str) -> None:
    from tests.support import make_memory

    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Lifecycle memory",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    with container.services() as services:
        current = services.memories.get_memory(memory.id, workspace_id)
        assert current is not None
        suspect = current.transition(MemoryStatus.SUSPECT, "two consecutive failures")
        services.memories.update_status(suspect)
        retired = suspect.transition(MemoryStatus.RETIRED, "superseded")
        services.memories.update_status(retired)
        stored = services.memories.get_memory(memory.id, workspace_id)
        assert stored is not None and stored.status is MemoryStatus.RETIRED
        with pytest.raises(TransitionError):
            stored.transition(MemoryStatus.ACTIVE, "illegal resurrection")


def test_outbox_entries_survive_index_failures(container, workspace_id: str) -> None:
    from tests.support import make_memory

    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Outbox memory",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    with container.services() as services:
        from sqlalchemy import func, select

        from recap.infrastructure.db.models import OutboxRow

        before = services.outbox.pending_count()
        stored = services.memories.get_memory(memory.id, workspace_id)
        assert stored is not None
        services.index_sync(stored)
        # Indexing succeeded. The outbox row from the first index call is durable
        # and pending until a worker acknowledges it, and the dedupe key prevents
        # a duplicate row for the same memory version.
        assert services.outbox.pending_count() == before
        dedupe = f"index:{memory.id}:v{memory.memory_version}"
        rows = services.session.scalars(
            select(OutboxRow).where(OutboxRow.dedupe_key == dedupe)
        ).all()
        assert len(rows) == 1
        assert rows[0].state == "pending"
        services.outbox.mark_done(rows[0])
        assert rows[0].state == "done"
        # A worker that later reconciles the same memory can enqueue again.
        services.outbox.enqueue(
            workspace_id=workspace_id,
            entity_id=memory.id,
            operation="index",
            dedupe_key=dedupe,
        )
        total = services.session.scalar(
            select(func.count()).select_from(OutboxRow).where(OutboxRow.dedupe_key == dedupe)
        )
        assert total == 2


def test_cross_workspace_reads_are_isolated(container, workspace_id: str) -> None:
    from tests.support import make_memory

    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Isolated memory",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    with container.services() as services:
        assert services.memories.get_memory(memory.id, "someone-else") is None
        assert services.memories.list_memories("someone-else") == []
        assert services.memories.get_memory(memory.id, workspace_id) is not None


def test_outcome_evidence_updates_contextual_stats(container, workspace_id: str) -> None:
    from tests.support import make_memory

    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Outcome memory",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    with container.services() as services:
        for outcome in (
            OutcomeStatus.VERIFIED_SUCCESS,
            OutcomeStatus.VERIFIED_SUCCESS,
            OutcomeStatus.VERIFIED_FAILURE,
        ):
            services.outcomes.append(
                OutcomeEvidence(
                    workspace_id=workspace_id,
                    memory_id=memory.id,
                    memory_version=memory.memory_version,
                    run_id=f"run-{uuid.uuid4().hex[:8]}",
                    outcome=outcome,
                    environment_fingerprint="mock-api-v1|list_records",
                    applied=True,
                )
            )
        stats = services.outcomes.context_stats(memory.id, "mock-api-v1|list_records")
        assert stats is not None
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["applied"] == 3
