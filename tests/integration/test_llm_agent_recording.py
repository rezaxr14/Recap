"""Real PostgreSQL trace of the configured-LLM agent loop.

The model is a scripted test double here; the point is that run/event recording,
model metadata, token accounting, and failure typing are real database writes.
Real inference is covered only by the opt-in live-provider suite.
"""

from __future__ import annotations

import pytest
from tests.support import (
    ScriptedChatProvider,
    make_memory,
    pagination_script,
)

from recap.domain.enums import EventType, RunStatus
from recap.environments.llm_agent import MemoryAdvisory
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.run_recording import record_llm_run
from recap.providers.base import ChatResult, ToolCall

pytestmark = pytest.mark.integration


def test_llm_agent_run_persists_model_metadata_and_real_tool_events(
    container, workspace_id: str
) -> None:
    provider = ScriptedChatProvider(pagination_script(version="2"))
    environment = PaginatedApiEnvironment(version="2")

    run_id, result = record_llm_run(
        container,
        environment,
        provider=provider,
        workspace_id=workspace_id,
        task_id="llm-agent-integration",
    )

    assert result.success is True
    with container.services() as services:
        run = services.runs.get_run(run_id)
        assert run is not None
        assert run.status is RunStatus.SUCCEEDED
        assert run.agent_id == "llm-tool-agent-v1"
        assert run.model_provider == "scripted"
        assert run.model_name == "scripted-agent-v1"
        assert run.task_id == "llm-agent-integration"
        assert run.tool_calls_count == 4
        assert run.retry_count == 1
        assert run.input_tokens == 50
        assert run.output_tokens == 25
        assert run.model_config_fingerprint.startswith("sha256:")
        assert run.config_snapshot["prompt_version"] == "llm-tool-agent-v1"
        assert run.config_snapshot["tools"] == ["list_records"]

        events = services.runs.list_events(run_id)
        types = [event.event_type for event in events]
        assert types[0] is EventType.RUN_STARTED
        assert types[-1] is EventType.RUN_FINISHED
        assert types.count(EventType.TOOL_CALL_REQUESTED) == 4
        assert types.count(EventType.TOOL_CALL_SUCCEEDED) == 3
        assert types.count(EventType.TOOL_CALL_FAILED) == 1
        failed = next(event for event in events if event.event_type is EventType.TOOL_CALL_FAILED)
        assert failed.payload_redacted["error_code"] == "INVALID_CURSOR"
        assert failed.tool_name == "list_records"


def test_llm_agent_failure_is_recorded_with_a_specific_failure_type(
    container, workspace_id: str
) -> None:
    def script(messages: list) -> ChatResult:  # type: ignore[type-arg]
        if not any(message.role == "tool" for message in messages):
            return ChatResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="delete_records",
                        arguments={"tenant": "acme"},
                    )
                ],
            )
        return ChatResult(content="I cannot do that")

    run_id, result = record_llm_run(
        container,
        PaginatedApiEnvironment(version="2"),
        provider=ScriptedChatProvider(script),
        workspace_id=workspace_id,
    )

    assert result.success is False
    with container.services() as services:
        run = services.runs.get_run(run_id)
        assert run is not None
        assert run.status is RunStatus.FAILED
        assert run.failure_type == "unknown_tool"
        events = services.runs.list_events(run_id)
        failed = next(event for event in events if event.event_type is EventType.TOOL_CALL_FAILED)
        assert failed.payload_redacted["error_code"] == "UNKNOWN_TOOL"


def test_memory_advisory_is_recorded_as_selected_without_copying_raw_text(
    container, workspace_id: str
) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Restart pagination",
        summary="Start from the first page.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="2",
    )
    advisory = MemoryAdvisory.from_memory(memory)

    run_id, result = record_llm_run(
        container,
        PaginatedApiEnvironment(version="2"),
        provider=ScriptedChatProvider(pagination_script(version="2")),
        workspace_id=workspace_id,
        memory=advisory,
    )

    assert result.success is True
    assert result.used_memory_id == memory.id
    with container.services() as services:
        events = services.runs.list_events(run_id)
        selected = next(event for event in events if event.event_type is EventType.MEMORY_SELECTED)
        assert selected.payload_redacted["memory_id"] == memory.id
        assert selected.payload_redacted["advisory"] is True
        # Only the bounded advisory fields are recorded, not a copied prompt.
        assert "summary" not in selected.payload_redacted
