"""Helpers shared by integration/e2e tests.

These build state through the real production services (recorder, compiler,
validators, promotion) rather than hand-constructing rows, so the tests exercise
the same code path as the CLI and API.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from recap.container import Container
from recap.domain.enums import ActionType, MemoryStatus, SideEffectClass
from recap.domain.memories import ExperienceMemory, MemoryContent
from recap.domain.predicates import Predicate
from recap.domain.strategies import RepairStrategy, StrategyAction
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.run_recording import record_reference_run
from recap.infrastructure.db.repositories import ensure_workspace
from recap.providers.base import (
    ChatMessage,
    ChatResult,
    ProviderCapabilities,
    ToolCall,
    Usage,
)


def make_memory(
    container: Container,
    *,
    workspace_id: str,
    title: str,
    summary: str,
    failure_signature: str,
    failure_code: str,
    environment_version: str,
    task_families: list[str] | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    preconditions: list[Predicate] | None = None,
    index: bool = True,
) -> ExperienceMemory:
    environment = PaginatedApiEnvironment(version=environment_version)
    fingerprint = environment.fingerprint()
    content = MemoryContent(
        title=title,
        summary=summary,
        failure_signature=failure_signature,
        failure_code=failure_code,
        task_families=task_families if task_families is not None else ["api_pagination"],
        tool_names=["list_records"],
        tool_schema_fingerprints={"list_records": fingerprint.tool_schema_hashes["list_records"]},
        environment_type="mock_api",
        preconditions=preconditions
        if preconditions is not None
        else [
            Predicate(field="error_code", operator="in", value=[failure_code]),
            Predicate(
                field="tool_schema_fingerprints.list_records",
                operator="eq",
                value=fingerprint.tool_schema_hashes["list_records"],
            ),
        ],
        strategy=RepairStrategy(
            summary=summary,
            actions=[
                StrategyAction(
                    action_type=ActionType.REFRESH_PAGINATION_CURSOR,
                    tool_name="list_records",
                    side_effect_class=SideEffectClass.READ_ONLY,
                )
            ],
        ),
        postconditions=[Predicate(field="records_collected", operator="gte", value=1)],
    )
    memory = ExperienceMemory(
        workspace_id=workspace_id,
        content=content,
        status=status,
        status_reason="integration fixture",
        environment_id=fingerprint.environment_id,
        environment_fingerprint=fingerprint.fingerprint,
    )
    with container.services() as services:
        ensure_workspace(services.session, workspace_id)
        created = services.memories.create_memory(memory)
        if index:
            assert services.index_sync(created) is True
        return created


def seed_promoted_memory(
    container: Container, *, workspace_id: str, environment_version: str
) -> tuple[str, Any]:
    """Compile, validate, and promote a memory through the real gate."""
    environment = PaginatedApiEnvironment(version=environment_version)
    fingerprint = environment.fingerprint()
    run_id, _ = record_reference_run(
        container,
        environment,
        workspace_id=workspace_id,
        task_id=f"integration-seed-{environment_version}",
    )
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)
        candidates = services.repair_compiler.compile(run_id, environment=fingerprint)
        assert candidates, "compiler produced no candidate for a failed run"
        candidate_id = candidates[0].id
        static = services.repair_compiler.validate(
            candidate_id, environment=fingerprint, validator_name="static"
        )
        assert static.status.value == "passed", static.checks
        sandbox = services.repair_compiler.validate(
            candidate_id, environment=fingerprint, validator_name="mock_sandbox"
        )
        assert sandbox.status.value == "passed", sandbox.checks
        refreshed = services.repairs.get(candidate_id)
        memory = services.promotion.promote(refreshed)
        return candidate_id, memory


def context_for(
    *,
    workspace_id: str,
    environment_version: str,
    query: str = "invalid pagination cursor",
    error_code: str = "INVALID_CURSOR",
    task_family: str = "api_pagination",
):
    from recap.domain.retrieval import RetrievalContext

    environment = PaginatedApiEnvironment(version=environment_version)
    fingerprint = environment.fingerprint()
    return RetrievalContext(
        workspace_id=workspace_id,
        query=query,
        task_family=task_family,
        environment_id=fingerprint.environment_id,
        environment_fingerprint=fingerprint.fingerprint,
        environment_version=fingerprint.environment_version,
        environment_type=fingerprint.environment_type,
        tool_names=["list_records"],
        tool_versions=fingerprint.tool_versions,
        tool_schema_fingerprints=fingerprint.tool_schema_hashes,
        error_type="ValidationError",
        error_code=error_code,
        error_message="cursor 'start' is invalid or expired",
        state={"tool_schema_fingerprints": fingerprint.tool_schema_hashes},
    )


class ScriptedChatProvider:
    """Test-only chat provider. Never used as real-model evidence."""

    provider_type = "scripted"
    model = "scripted-agent-v1"
    capabilities = ProviderCapabilities(
        provider_type=provider_type,
        model=model,
        chat=True,
        tool_calling=True,
    )

    def __init__(self, script: Callable[[list[ChatMessage]], ChatResult]) -> None:
        self.script = script
        self.requests: list[list[ChatMessage]] = []
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        snapshot = list(messages)
        self.requests.append(snapshot)
        self.tools_seen.append(list(tools) if tools is not None else None)
        return self.script(snapshot)

    def list_models(self) -> list[str]:
        return [self.model]

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_available": True}

    def generate_structured(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("ScriptedChatProvider does not generate structured output")


SCRIPTED_USAGE = Usage(prompt_tokens=10, completion_tokens=5)


def pagination_script(
    *, version: str = "2", usage: Usage = SCRIPTED_USAGE
) -> Callable[[list[ChatMessage]], ChatResult]:
    """A deterministic stand-in for a competent tool-calling model."""

    pagination_key = "page_token" if version == "2" else "cursor"

    def script(messages: list[ChatMessage]) -> ChatResult:
        last_tool = next(
            (message for message in reversed(messages) if message.role == "tool"), None
        )
        turn = sum(1 for message in messages if message.role == "tool") + 1
        if last_tool is None:
            arguments: dict[str, Any] = {"tenant": "acme", pagination_key: "start"}
        else:
            payload = json.loads(last_tool.content or "{}")
            if not payload.get("ok"):
                arguments = {"tenant": "acme"}
            elif payload.get("has_more"):
                arguments = {"tenant": "acme", pagination_key: payload.get(pagination_key)}
            else:
                return ChatResult(content="all records collected", usage=usage)
        return ChatResult(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"call-{turn}",
                    name="list_records",
                    arguments=arguments,
                )
            ],
            usage=usage,
        )

    return script
