"""Record a reference-agent run into the durable store.

Shared by the demo, benchmark seeding, and tests so the event shape is identical
everywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from recap.container import Container
from recap.domain.enums import EventType, RunStatus
from recap.domain.runs import AgentRun, RunEventCreate
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.reference_agent import AgentRunResult, ReferenceAgent


def record_reference_run(
    container: Container,
    environment: PaginatedApiEnvironment,
    *,
    workspace_id: str = "demo-workspace",
    task_id: str | None = None,
) -> tuple[str, AgentRunResult]:
    fingerprint = environment.fingerprint()
    agent = ReferenceAgent()
    agent_result = agent.run(environment)

    with container.services() as services:
        from recap.infrastructure.db.repositories import ensure_workspace

        ensure_workspace(services.session, workspace_id)
        run = AgentRun(
            workspace_id=workspace_id,
            agent_id="reference-agent",
            task_id=task_id
            or f"paginate-{environment.version}-{datetime.now(UTC).timestamp():.0f}",
            task_family="api_pagination",
            environment_id=fingerprint.environment_id,
            environment_fingerprint=fingerprint.fingerprint,
            toolset_fingerprint=fingerprint.fingerprint,
            model_provider="deterministic",
            model_name="reference-agent-v1",
            status=RunStatus.RUNNING,
            success_criteria="all records collected",
        )
        services.recorder.start_run(run)
        sequence = 0

        def append(event_type: EventType, payload: dict[str, Any], tool: str | None = None) -> None:
            nonlocal sequence
            sequence += 1
            services.recorder.append_event(
                run,
                RunEventCreate(
                    sequence_number=sequence,
                    event_type=event_type,
                    tool_name=tool,
                    tool_version=environment.version if tool else None,
                    payload=payload,
                ),
            )

        append(
            EventType.RUN_STARTED,
            {"task": "collect all records", "environment_version": environment.version},
        )
        append(
            EventType.TASK_OBSERVATION,
            {"observation": f"tenant={environment.tenant} total={len(environment.records)}"},
        )
        for step in agent_result.steps:
            append(EventType.TOOL_CALL_REQUESTED, {"arguments": step.request}, tool="list_records")
            if step.ok:
                append(EventType.TOOL_CALL_SUCCEEDED, {"result": step.result}, tool="list_records")
            else:
                append(
                    EventType.TOOL_CALL_FAILED,
                    {"request": step.request, **(step.error or {})},
                    tool="list_records",
                )
        append(
            EventType.POSTCONDITION_CHECKED,
            {"passed": agent_result.success, "detail": agent_result.final_error},
        )
        append(
            EventType.RUN_FINISHED,
            {"success": agent_result.success, "error": agent_result.final_error},
        )
        services.recorder.finish_run(
            run,
            status=RunStatus.SUCCEEDED if agent_result.success else RunStatus.FAILED,
            failure_type=None if agent_result.success else "invalid_pagination_cursor",
            tool_calls_count=agent_result.tool_calls,
            retry_count=agent_result.retries,
        )
        return run.id, agent_result
