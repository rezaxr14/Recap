"""Record agent runs into the durable store.

Two agents share this recorder so their event shape is identical:

* ``record_reference_run``: the deterministic reference agent used by the demo,
  benchmark, and reproducibility tests.
* ``record_llm_run``: the configured real chat model driving the same mock API
  through tool calls.

The benchmark uses ``start_recorded_run`` before retrieval and
``complete_recorded_run`` after the agent finishes. That ordering means the
task's retrieval is pinned to the run's memory snapshot, and the metrics are
read back from the persisted run rather than from an in-memory shortcut.

The recorded run never derives success from the model's self-report. It comes
from the agent's environment postcondition.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from recap.container import Container
from recap.domain.enums import EventType, RunStatus
from recap.domain.environments import canonical_json
from recap.domain.runs import AgentRun, RunEventCreate
from recap.environments.llm_agent import PROMPT_VERSION, LlmAgent, MemoryAdvisory
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.reference_agent import AgentRunResult, ReferenceAgent
from recap.errors import (
    NotFoundError,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderUnavailable,
)
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


def start_recorded_run(
    container: Container,
    environment: PaginatedApiEnvironment,
    *,
    metadata: AgentRunMetadata,
    workspace_id: str = "demo-workspace",
    task_id: str | None = None,
) -> AgentRun:
    """Create the run and its frozen memory snapshot before the agent starts."""

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
        services.recorder.append_event(
            run,
            RunEventCreate(
                sequence_number=1,
                event_type=EventType.RUN_STARTED,
                payload={
                    "task": "collect all records",
                    "environment_version": environment.version,
                    "agent_id": metadata.agent_id,
                    "memory_snapshot_id": run.memory_snapshot_id,
                },
            ),
        )
        return run


def complete_recorded_run(
    container: Container,
    run: AgentRun,
    environment: PaginatedApiEnvironment,
    agent_result: AgentRunResult,
    *,
    metadata: AgentRunMetadata,
    memory: Any | None = None,
    memory_source: str = "advisory",
) -> str:
    """Append the observable trace, enforce the postcondition, and finish the run."""

    with container.services() as services:
        current = services.recorder.get_run(run.id)
        if current is None:
            raise NotFoundError(f"run {run.id} not found")
        sequence = 0
        existing = services.runs.list_events(run.id)
        if existing:
            sequence = existing[-1].sequence_number

        def append(event_type: EventType, payload: dict[str, Any], tool: str | None = None) -> None:
            nonlocal sequence
            sequence += 1
            services.recorder.append_event(
                current,
                RunEventCreate(
                    sequence_number=sequence,
                    event_type=event_type,
                    tool_name=tool,
                    tool_version=environment.version if tool else None,
                    payload=payload,
                ),
            )

        append(
            EventType.TASK_OBSERVATION,
            {
                "observation": f"tenant={environment.tenant} total={len(environment.records)}",
                "model_turns": agent_result.model_turns,
                "input_tokens": agent_result.input_tokens,
                "output_tokens": agent_result.output_tokens,
                "provider_attempts": agent_result.provider_attempts,
                "provider_retries": agent_result.provider_retries,
                "first_token_ms": agent_result.first_token_ms,
            },
        )
        memory_id = _memory_id(memory)
        if memory_id is not None:
            memory_payload: dict[str, Any] = {
                "memory_id": memory_id,
                "memory_version": getattr(memory, "memory_version", None),
                "title": getattr(memory, "title", ""),
                "source": memory_source,
            }
            if memory_source == "llm_advisory":
                memory_payload["advisory"] = True
            append(EventType.MEMORY_SELECTED, memory_payload)
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
        if agent_result.used_memory_id is not None:
            append(
                EventType.MEMORY_APPLIED,
                {
                    "memory_id": agent_result.used_memory_id,
                    "attribution": "environment_recovery",
                    "success": agent_result.success,
                },
            )
        append(
            EventType.POSTCONDITION_CHECKED,
            {
                "passed": agent_result.success,
                "detail": None if agent_result.success else agent_result.final_error,
            },
        )
        append(
            EventType.RUN_FINISHED,
            {
                "success": agent_result.success,
                "error": None if agent_result.success else agent_result.final_error,
            },
        )
        failure_type = None
        if not agent_result.success:
            failure_type = metadata.failure_type or _infer_failure_type(agent_result)
        services.recorder.finish_run(
            current,
            status=RunStatus.SUCCEEDED if agent_result.success else RunStatus.FAILED,
            failure_type=failure_type,
            tool_calls_count=agent_result.tool_calls,
            retry_count=agent_result.retries,
            input_tokens=agent_result.input_tokens,
            output_tokens=agent_result.output_tokens,
            duration_ms=agent_result.duration_ms,
        )
        return current.id


def record_reference_run(
    container: Container,
    environment: PaginatedApiEnvironment,
    *,
    workspace_id: str = "demo-workspace",
    task_id: str | None = None,
    failure_family: str | None = None,
) -> tuple[str, AgentRunResult]:
    family = failure_family or environment.failure_family
    explicit_family = failure_family is not None or environment.family_explicit
    metadata = AgentRunMetadata(
        agent_id="reference-agent",
        model_provider="deterministic",
        model_name="reference-agent-v1",
        task_family=family if explicit_family else "api_pagination",
        failure_type=family if explicit_family else "invalid_pagination_cursor",
    )
    run = start_recorded_run(
        container, environment, metadata=metadata, workspace_id=workspace_id, task_id=task_id
    )
    agent_result = ReferenceAgent().run(environment)
    run_id = complete_recorded_run(container, run, environment, agent_result, metadata=metadata)
    return run_id, agent_result


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

    metadata = AgentRunMetadata(
        agent_id="llm-tool-agent-v1",
        model_provider=str(getattr(provider, "provider_type", "unknown")),
        model_name=str(getattr(provider, "model", "") or "unknown"),
        task_family=(
            environment.failure_family if environment.family_explicit else "api_pagination"
        ),
        config_snapshot={
            "prompt_version": PROMPT_VERSION,
            "max_tool_calls": max_tool_calls,
            "temperature": 0.0,
            "tools": ["list_records"],
            "memory_id": memory.memory_id if memory is not None else None,
        },
    )
    run = start_recorded_run(
        container, environment, metadata=metadata, workspace_id=workspace_id, task_id=task_id
    )
    try:
        agent_result = LlmAgent(provider).run(
            environment, memory=memory, max_tool_calls=max_tool_calls
        )
    except (
        ProviderUnavailable,
        ProviderAuthenticationError,
        ProviderResponseError,
    ) as exc:
        # Persist the terminal state, then surface the provider failure to the
        # caller. Never turn an outage into a fake answer or a success.
        failed = AgentRunResult(success=False, final_error=f"provider error: {exc.message}")
        with contextlib.suppress(Exception):
            complete_recorded_run(
                container,
                run,
                environment,
                failed,
                metadata=replace(metadata, failure_type="provider_error"),
                memory=memory,
                memory_source="llm_advisory",
            )
        raise
    run_id = complete_recorded_run(
        container,
        run,
        environment,
        agent_result,
        metadata=metadata,
        memory=memory,
        memory_source="llm_advisory",
    )
    return run_id, agent_result


def record_agent_run(
    container: Container,
    environment: PaginatedApiEnvironment,
    agent_result: AgentRunResult,
    *,
    metadata: AgentRunMetadata,
    workspace_id: str = "demo-workspace",
    task_id: str | None = None,
    memory: Any | None = None,
    memory_source: str = "advisory",
) -> tuple[str, AgentRunResult]:
    run = start_recorded_run(
        container, environment, metadata=metadata, workspace_id=workspace_id, task_id=task_id
    )
    run_id = complete_recorded_run(
        container,
        run,
        environment,
        agent_result,
        metadata=metadata,
        memory=memory,
        memory_source=memory_source,
    )
    return run_id, agent_result


def _memory_id(memory: Any | None) -> str | None:
    if memory is None:
        return None
    value = getattr(memory, "memory_id", None)
    if value is None:
        value = getattr(memory, "id", None)
    return str(value) if value is not None else None


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
