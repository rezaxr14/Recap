from __future__ import annotations

import pytest

from recap.domain.enums import (
    ActionType,
    EventType,
    ValidationStatus,
)
from recap.domain.predicates import Predicate
from recap.domain.strategies import StrategyAction
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.errors import ValidationError
from recap.repair_compiler.candidate_generation import (
    CandidateDraft,
    DeterministicGenerator,
    validate_draft,
)
from recap.repair_compiler.failure_windows import extract_failure_window
from recap.repair_compiler.validation import MockSandboxValidator, StaticValidator


def _events(failure_payload: dict | None = None):
    from recap.domain.runs import RunEvent

    payload = failure_payload or {
        "error_type": "ValidationError",
        "error_code": "INVALID_CURSOR",
        "error_message": "cursor 'start' is invalid or expired",
    }
    return [
        RunEvent(
            run_id="run_1",
            sequence_number=1,
            event_type=EventType.RUN_STARTED,
            payload_redacted={"task": "collect"},
        ),
        RunEvent(
            run_id="run_1",
            sequence_number=2,
            event_type=EventType.TOOL_CALL_REQUESTED,
            tool_name="list_records",
            payload_redacted={"arguments": {"cursor": "start"}},
        ),
        RunEvent(
            run_id="run_1",
            sequence_number=3,
            event_type=EventType.TOOL_CALL_FAILED,
            tool_name="list_records",
            payload_redacted=payload,
        ),
    ]


def test_failure_window_extraction_finds_invalid_cursor() -> None:
    window = extract_failure_window(run_id="run_1", events=_events())
    assert window is not None
    assert window.failure_category == "invalid_pagination_cursor"
    assert window.error_code == "INVALID_CURSOR"
    assert window.start_sequence <= 3
    assert window.end_event_id


def test_failure_window_returns_none_without_failures() -> None:
    from recap.domain.runs import RunEvent

    events = [
        RunEvent(
            run_id="run_1",
            sequence_number=1,
            event_type=EventType.RUN_STARTED,
            payload_redacted={},
        )
    ]
    assert extract_failure_window(run_id="run_1", events=events) is None


def test_generator_produces_typed_candidate_for_cursor_failure() -> None:
    window = extract_failure_window(run_id="run_1", events=_events())
    assert window is not None
    drafts = DeterministicGenerator().generate(
        window=window, environment={"environment_version": "1", "tool_schema_hash": "sha256:abc"}
    )
    assert len(drafts) == 1
    assert drafts[0].actions[0].action_type is ActionType.REFRESH_PAGINATION_CURSOR


def test_generator_declines_insufficient_evidence() -> None:
    window = extract_failure_window(run_id="run_1", events=_events())
    assert window is not None
    window.incomplete = True
    assert DeterministicGenerator().generate(window=window, environment={}) == []


def test_validate_draft_rejects_invented_event_ids() -> None:
    from recap.config import RepairCompilerSettings

    window = extract_failure_window(run_id="run_1", events=_events())
    assert window is not None
    draft = CandidateDraft(
        summary="s",
        hypothesis="h",
        evidence_event_ids=["evt_does_not_exist"],
        preconditions=[Predicate(field="error_code", operator="eq", value="INVALID_CURSOR")],
        actions=[
            StrategyAction(
                action_type=ActionType.REFRESH_PAGINATION_CURSOR, tool_name="list_records"
            )
        ],
        postconditions=[Predicate(field="records_collected", operator="gte", value=1)],
    )
    with pytest.raises(ValidationError):
        validate_draft(
            draft,
            window=window,
            settings=RepairCompilerSettings(),
            allowed_event_ids={"evt_real"},
        )


def test_static_validator_passes_safe_candidate() -> None:
    from recap.repair_compiler.candidate_generation import build_candidate

    window = extract_failure_window(run_id="run_1", events=_events())
    assert window is not None
    environment = PaginatedApiEnvironment(version="1").fingerprint()
    draft = DeterministicGenerator().generate(
        window=window,
        environment={
            "environment_version": "1",
            "tool_schema_hash": environment.tool_schema_hashes["list_records"],
        },
    )[0]
    candidate = build_candidate(
        draft,
        window=window,
        workspace_id="ws-1",
        environment_id=environment.environment_id,
        environment_fingerprint=environment.fingerprint,
        generated_by="test",
    )
    result = StaticValidator().validate(candidate, environment=environment)
    assert result.status is ValidationStatus.PASSED
    assert all(check["passed"] for check in result.checks)


def test_mock_sandbox_validates_cursor_repair() -> None:
    from recap.repair_compiler.candidate_generation import build_candidate

    window = extract_failure_window(run_id="run_1", events=_events())
    assert window is not None
    environment = PaginatedApiEnvironment(version="1").fingerprint()
    draft = DeterministicGenerator().generate(
        window=window,
        environment={
            "environment_version": "1",
            "tool_schema_hash": environment.tool_schema_hashes["list_records"],
        },
    )[0]
    candidate = build_candidate(
        draft,
        window=window,
        workspace_id="ws-1",
        environment_id=environment.environment_id,
        environment_fingerprint=environment.fingerprint,
        generated_by="test",
    )
    result = MockSandboxValidator().validate(candidate, environment=environment)
    assert result.status is ValidationStatus.PASSED, result.detail
    names = {check["check"]: check["passed"] for check in result.checks}
    assert names["failure_reproduced"] is True
    assert names["postconditions_satisfied"] is True
    assert names["rejects_mismatched_schema"] is True


def test_mock_sandbox_fails_candidate_with_wrong_schema_precondition() -> None:
    from recap.repair_compiler.candidate_generation import build_candidate

    window = extract_failure_window(run_id="run_1", events=_events())
    assert window is not None
    environment = PaginatedApiEnvironment(version="1").fingerprint()
    draft = DeterministicGenerator().generate(
        window=window,
        environment={
            "environment_version": "1",
            "tool_schema_hash": "sha256:not-the-real-schema",
        },
    )[0]
    candidate = build_candidate(
        draft,
        window=window,
        workspace_id="ws-1",
        environment_id=environment.environment_id,
        environment_fingerprint=environment.fingerprint,
        generated_by="test",
    )
    result = MockSandboxValidator().validate(candidate, environment=environment)
    assert result.status is ValidationStatus.FAILED
