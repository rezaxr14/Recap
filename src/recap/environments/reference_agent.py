"""Reference deterministic agent used by the demo and end-to-end tests.

This is a *real* small policy loop over the mock environment, not an LLM. It is
explicitly labelled deterministic so its results are never presented as evidence
of real LLM agent performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recap.domain.enums import ActionType
from recap.domain.retrieval import SelectedMemory
from recap.environments.mock_api import MockApiSuite, PaginatedApiEnvironment, ToolCallResult


@dataclass
class AgentStep:
    action: str
    request: dict[str, Any]
    result: dict[str, Any]
    ok: bool
    error: dict[str, Any] | None = None


@dataclass
class AgentRunResult:
    success: bool
    steps: list[AgentStep] = field(default_factory=list)
    tool_calls: int = 0
    retries: int = 0
    final_error: str = ""
    used_memory_id: str | None = None


class ReferenceAgent:
    """Executes a pagination task, optionally applying a retrieved memory strategy.

    Without memory it uses a plausible-but-wrong cursor assumption so the first
    attempt fails deterministically. With an applicable memory it follows the
    declared strategy actions.
    """

    def __init__(self, *, tenant: str = "acme") -> None:
        self.tenant = tenant

    def run(
        self,
        environment: PaginatedApiEnvironment,
        *,
        memory: SelectedMemory | None = None,
        max_tool_calls: int = 6,
    ) -> AgentRunResult:
        result = AgentRunResult(success=False)
        collected: list[dict[str, Any]] = []
        token: str | None = None
        first = True
        while result.tool_calls < max_tool_calls:
            parameters: dict[str, Any]
            if first:
                # A naive agent guesses a cursor value on the first call.
                parameters = {"tenant": self.tenant, "cursor": "start"}
                first = False
            else:
                parameters = {"tenant": self.tenant}
                if token is not None:
                    if environment.version == "1":
                        parameters["cursor"] = token
                    else:
                        parameters["page_token"] = token
            call = environment.list_records(**parameters)
            result.tool_calls += 1
            result.steps.append(
                AgentStep(
                    action="list_records",
                    request=parameters,
                    result=call.payload,
                    ok=call.ok,
                    error=call.error,
                )
            )
            if not call.ok:
                result.final_error = (call.error or {}).get("error_message", "tool call failed")
                if memory is not None:
                    recovered, recovery_token = self._recover(
                        environment, memory, parameters, call
                    )
                    if not recovered:
                        return result
                    token = recovery_token
                    result.retries += 1
                    result.used_memory_id = memory.memory_id
                    continue
                return result
            collected.extend(call.payload.get("items", []))
            token = call.payload.get("cursor") or call.payload.get("page_token")
            if not call.payload.get("has_more"):
                result.success = len(collected) == len(environment.records)
                return result
        return result

    def _recover(
        self,
        environment: PaginatedApiEnvironment,
        memory: SelectedMemory,
        failed_parameters: dict[str, Any],
        failed_call: ToolCallResult,
    ) -> tuple[bool, str | None]:
        """Apply only allowlisted typed actions; never execute free-form text.

        Returns ``(recovered, next_token)``. ``recovered`` is False when no safe
        action applies or the strategy explicitly halts.
        """
        error_code = (failed_call.error or {}).get("error_code", "")
        for action in memory.actions:
            action_type = action.get("action_type")
            if action_type == ActionType.RETRY_WITH_BACKOFF.value:
                # Retrying with the same parameters is only safe for transient errors.
                if error_code in {"RATE_LIMITED", "TIMEOUT", "UNAVAILABLE"}:
                    return True, failed_parameters.get("cursor") or failed_parameters.get("page_token")
            if action_type == ActionType.REFRESH_PAGINATION_CURSOR.value:
                # Start a fresh page sequence instead of reusing the invalid cursor.
                return True, None
            if action_type == ActionType.MAP_PARAMETER.value:
                mapping = action.get("parameters", {})
                old = mapping.get("from")
                new = mapping.get("to")
                if old and new and old in failed_parameters:
                    value = failed_parameters[old]
                    if value in {None, "", "start"}:
                        # The value itself is invalid; a fresh token is required.
                        return True, None
                    return True, str(value)
                return True, None
            if action_type in {
                ActionType.REQUEST_MISSING_FIELD.value,
                ActionType.HALT.value,
            }:
                return False, None
        return False, None

    def run_task(
        self,
        suite: MockApiSuite,
        version: str = "1",
        *,
        memory: SelectedMemory | None = None,
        expire_first_cursor: bool = False,
    ) -> AgentRunResult:
        environment = suite.get(version)
        if expire_first_cursor:
            # Force the failure mode used by the seeded demo.
            environment.expire_cursor("cur_1_0001")
        return self.run(environment, memory=memory)
