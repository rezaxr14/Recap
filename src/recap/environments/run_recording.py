"""Record agent runs into the durable store.

Two agents share this recorder so their event shape is identical:

* ``record_reference_run``: the deterministic reference agent used by the demo,
  benchmark, and reproducibility tests.
* ``record_llm_run``: the configured real chat model driving the same mock API
  through tool calls.

The recorded run never derives success from the model's self-report. It comes
from the agent's environment postcondition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from recap.container import Container
from recap.domain.enums import EventType, RunStatus
from recap.domain.environments import canonical_json
from recap.domain.runs import AgentRun, RunEventCreate
from recap.environments.llm_agent import PROMPT_VERSION, LlmAgent, MemoryAdvisory
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.reference_agent import AgentRunResult, ReferenceAgent
from recap.providers.base import ChatProvider


@dataclass(frozen=True)
class AgentRunMetadata:
    """Provenance that makes a run reproducible and auditable."""

    agent_id: str
    model_provider: str
    model_name: str
    task_family: str = "api_pagination"
    success_criteria: str = "all records collected"
    failure_type: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)


def record_reference_run(
    container: Container,
    environment: PaginatedApiEnvironment,
    *,
    workspace_id: str = "demo-workspace",
    task_id: str | None = None,
) -> tuple[str, AgentRunResult]:
    agent_result = ReferenceAgent().run(environment)
    metadata = AgentRunMetadata(
        agent_id="reference-agent",
        model_provider="deterministic",
        model_name="reference-agent-v1",
        failure_type="invalid_pagination_cursor",
    )
    return record_agent_run(
        container,
        environment,
        agent_result,
        metadata=metadata,
        workspace_id=workspace_id,
        task_id=task_id,
    )


def record_llm_run(
    container: Container,
    environment: PaginatedApiEnvironment,
    *,
    provider: ChatProvider,
    workspace_id: str = "demo-workspace",
    task_id: str | None = None,
    memory: MemoryAdvisory | None = None,
    max_tool_calls: int = 6,
) -> tuple[str, AgentRunResult]:
    """Run the configured chat model and persist its real tool-use trace."""

    agent = LlmAgent(provider)
    agent_result = agent.run(environment, memory=memory, max_tool_calls=max_tool_calls)
    metadata = AgentRunMetadata(
        agent_id="llm-tool-agent-v1",
        model_provider=str(getattr(provider, "provider_type", "unknown")),
        model_name=str(getattr(provider, "model", "") or "unknown"),
        config_snapshot={
            "prompt_version": PROMPT_VERSION,
            "max_tool_calls": max_tool_calls,
            "temperature": 0.0,
            "tools": ["list_records"],
            "memory_id": memory.memory_id if memory is not None else None,
        },
    )
    return record_agent_run(
        container,
        environment,
        agent_result,
        metadata=metadata,
        workspace_id=workspace_id,
        task_id=task_id,
        memory=memory,
    )


def record_agent_run(
    container: Container,
    environment: PaginatedApiEnvironment,
    agent_result: AgentRunResult,
    *,
    metadata: AgentRunMetadata,
    workspace_id: str = "demo-workspace",
    task_id: str | None = None,
    memory: MemoryAdvisory | None = None,
) -> tuple[str, AgentRunResult]:
    fingerprint = environment.fingerprint()
    with container.services() as services:
        from recap.infrastructure.db.repositories import ensure_workspace

        ensure_workspace(services.session, workspace_id)
        run = AgentRun(
            workspace_id=workspace_id,
            agent_id=metadata.agent_id,
            task_id=task_id
            or f"paginate-{environment.version}-{datetime.now(UTC).timestamp():.0f}",
            task_family=metadata.task_family,
            environment_id=fingerprint.environment_id,
            environment_fingerprint=fingerprint.fingerprint,
            toolset_fingerprint=fingerprint.fingerprint,
            model_provider=metadata.model_provider,
            model_name=metadata.model_name,
            model_config_fingerprint=_config_fingerprint(metadata.config_snapshot),
            status=RunStatus.RUNNING,
            success_criteria=metadata.success_criteria,
            config_snapshot=dict(metadata.config_snapshot),
        )
        run = services.recorder.start_run(run)
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
            {
                "task": "collect all records",
                "environment_version": environment.version,
                "agent_id": metadata.agent_id,
            },
        )
        append(
            EventType.TASK_OBSERVATION,
            {
                "observation": f"tenant={environment.tenant} total={len(environment.records)}",
                "model_turns": agent_result.model_turns,
                "input_tokens": agent_result.input_tokens,
                "output_tokens": agent_result.output_tokens,
            },
        )
        if memory is not None:
            append(
                EventType.MEMORY_SELECTED,
                {
                    "memory_id": memory.memory_id,
                    "memory_version": memory.memory_version,
                    "title": memory.title,
                    "advisory": True,
                },
            )
        for step in agent_result.steps:
            append(EventType.TOOL_CALL_REQUESTED, {"arguments": step.request}, tool=step.action)
            if step.ok:
                append(EventType.TOOL_CALL_SUCCEEDED, {"result": step.result}, tool=step.action)
            else:
                append(
                    EventType.TOOL_CALL_FAILED,
                    {"request": step.request, **(step.error or {})},
                    tool=step.action,
                )
        append(
            EventType.POSTCONDITION_CHECKED,
            {"passed": agent_result.success, "detail": agent_result.final_error},
        )
        append(
            EventType.RUN_FINISHED,
            {"success": agent_result.success, "error": agent_result.final_error},
        )
        failure_type = None
        if not agent_result.success:
            failure_type = metadata.failure_type or _infer_failure_type(agent_result)
        services.recorder.finish_run(
            run,
            status=RunStatus.SUCCEEDED if agent_result.success else RunStatus.FAILED,
            failure_type=failure_type,
            tool_calls_count=agent_result.tool_calls,
            retry_count=agent_result.retries,
            input_tokens=agent_result.input_tokens,
            output_tokens=agent_result.output_tokens,
            duration_ms=agent_result.duration_ms,
        )
        return run.id, agent_result


def _infer_failure_type(result: AgentRunResult) -> str:
    for step in reversed(result.steps):
        if not step.ok:
            code = (step.error or {}).get("error_code")
            if code:
                return str(code).lower()
    return "tool_failure"


def _config_fingerprint(config: dict[str, Any]) -> str:
    if not config:
        return ""
    payload = canonical_json(config).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
