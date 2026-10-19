from __future__ import annotations

import pytest

from recap.domain.enums import EventType, RunStatus
from recap.domain.runs import AgentRun, RunEventCreate
from recap.errors import ConflictError, ValidationError
from recap.infrastructure.db.repositories import SqlRunRepository
from recap.infrastructure.db.session import Database, build_engine
from recap.recorder.recorder import RunRecorder


@pytest.fixture()
def recorder() -> RunRecorder:
    database = Database(build_engine("sqlite+pysqlite:///:memory:"))
    database.create_all()
    session = database.session_factory()
    repository = SqlRunRepository(session)
    yield RunRecorder(repository)  # type: ignore[misc]
    session.close()


def _run() -> AgentRun:
    return AgentRun(
        workspace_id="ws-1",
        agent_id="agent-1",
        task_id="task-1",
        task_family="api_pagination",
        environment_id="mock-api-v1",
        environment_fingerprint="sha256:env",
    )


def test_events_are_appended_in_order(recorder: RunRecorder) -> None:
    run = recorder.start_run(_run())
    first = recorder.append_event(
        run,
        RunEventCreate(
            sequence_number=1,
            event_type=EventType.RUN_STARTED,
            payload={"task": "x"},
        ),
    )
    second = recorder.append_event(
        run,
        RunEventCreate(
            sequence_number=2,
            event_type=EventType.TOOL_CALL_REQUESTED,
            tool_name="list_records",
            payload={"arguments": {"cursor": "x"}},
        ),
    )
    events = recorder.repository.list_events(run.id)
    assert [event.sequence_number for event in events] == [1, 2]
    assert first.id != second.id


def test_duplicate_event_submission_is_idempotent(recorder: RunRecorder) -> None:
    run = recorder.start_run(_run())
    payload = RunEventCreate(
        sequence_number=1,
        event_type=EventType.RUN_STARTED,
        payload={"task": "x"},
    )
    first = recorder.append_event(run, payload)
    duplicate = recorder.append_event(run, payload)
    assert duplicate.id == first.id
    assert len(recorder.repository.list_events(run.id)) == 1


def test_conflicting_payload_for_same_sequence_is_rejected(recorder: RunRecorder) -> None:
    run = recorder.start_run(_run())
    recorder.append_event(
        run,
        RunEventCreate(sequence_number=1, event_type=EventType.RUN_STARTED, payload={"task": "x"}),
    )
    with pytest.raises(ConflictError):
        recorder.append_event(
            run,
            RunEventCreate(
                sequence_number=1,
                event_type=EventType.RUN_STARTED,
                payload={"task": "different"},
            ),
        )


def test_sequence_numbers_must_increase(recorder: RunRecorder) -> None:
    run = recorder.start_run(_run())
    recorder.append_event(
        run,
        RunEventCreate(sequence_number=5, event_type=EventType.RUN_STARTED, payload={}),
    )
    with pytest.raises(ValidationError):
        recorder.append_event(
            run,
            RunEventCreate(sequence_number=3, event_type=EventType.TASK_OBSERVATION, payload={}),
        )


def test_secrets_are_redacted_before_persistence(recorder: RunRecorder) -> None:
    run = recorder.start_run(_run())
    recorder.append_event(
        run,
        RunEventCreate(
            sequence_number=1,
            event_type=EventType.TOOL_CALL_FAILED,
            tool_name="list_records",
            payload={
                "error_code": "UNAUTHORIZED",
                "headers": {"authorization": "Bearer supersecrettoken123456"},
                "api_key": "sk-live-abcdefghijklmnop",
            },
        ),
    )
    stored = recorder.repository.list_events(run.id)[0]
    serialized = str(stored.payload_redacted)
    assert "supersecrettoken123456" not in serialized
    assert "sk-live-abcdefghijklmnop" not in serialized
    assert stored.redaction_applied is True


def test_cannot_append_to_finished_run(recorder: RunRecorder) -> None:
    run = recorder.start_run(_run())
    finished = recorder.finish_run(run, status=RunStatus.FAILED, failure_type="x")
    with pytest.raises(ConflictError):
        recorder.append_event(
            finished,
            RunEventCreate(sequence_number=1, event_type=EventType.RUN_STARTED, payload={}),
        )


def test_failure_context_is_derived_from_events(recorder: RunRecorder) -> None:
    run = recorder.start_run(_run())
    recorder.append_event(
        run,
        RunEventCreate(
            sequence_number=1,
            event_type=EventType.TOOL_CALL_FAILED,
            tool_name="list_records",
            payload={
                "error_type": "ValidationError",
                "error_code": "INVALID_CURSOR",
                "error_message": "cursor 'start' is invalid",
            },
        ),
    )
    context = recorder.build_failure_context(run, recorder.repository.list_events(run.id))
    assert context["error_code"] == "INVALID_CURSOR"
    assert "INVALID_CURSOR" in context["failure_signature"]
