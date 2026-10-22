"""Reference deterministic agent used by the demo and end-to-end tests.

This is a *real* small policy loop over the mock environment, not an LLM. It is
explicitly labelled deterministic so its results are never presented as evidence
of real LLM agent performance.

The agent starts from a plausible pre-memory request shape and may apply typed
memory actions after a failure. It never reads the environment version or the
answer directly from the environment; the recovery must come from the selected
memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recap.domain.enums import ActionType
from recap.domain.retrieval import SelectedMemory
from recap.environments.mock_api import (
    MockApiSuite,
    PaginatedApiEnvironment,
    ToolCallResult,
    family_error,
)


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
    model_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    # Provider-call accounting is separate from tool retries. A timeout or
    # rate-limit must not be silently reported as a model reasoning failure.
    provider_attempts: int = 0
    provider_retries: int = 0
    first_token_ms: float | None = None
    provider_error_type: str | None = None
    ordered_record_ids: list[str] = field(default_factory=list)


class ReferenceAgent:
    """Executes a collection task, optionally applying a retrieved memory."""

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
        base = environment.naive_parameters()
        base["tenant"] = self.tenant
        token: str | None = None
        response_field = environment.naive_response_field()
        parameter_name = _paginations_parameter(environment)
        # Silent drift has no error to trigger a recovery. A memory-driven agent
        # applies its typed actions before the first call; a memory-free agent
        # cannot know the convention and is expected to fail externally.
        if memory is not None and getattr(environment, "is_v3", False):
            applied, base, response_field, parameter_name = self._apply_proactive(
                memory, base, response_field, parameter_name
            )
            if applied:
                result.used_memory_id = memory.memory_id
        while result.tool_calls < max_tool_calls:
            parameters = dict(base)
            if token is not None:
                parameters[parameter_name] = token
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
                if memory is None:
                    # Tier B: the fix is discoverable by probing. The opaque
                    # message does not name the field, but the schema exposes an
                    # optional account id that can be supplied speculatively.
                    error_code = str((call.error or {}).get("error_code", ""))
                    if (
                        getattr(environment, "is_v3", False)
                        and error_code == "INVALID_REQUEST"
                        and "account_id" not in base
                    ):
                        base["account_id"] = "acct-bench"
                        result.retries += 1
                        continue
                    return result
                recovered, base, token, response_field = self._recover(
                    memory, base, token, response_field, call
                )
                if not recovered:
                    return result
                result.retries += 1
                result.used_memory_id = memory.memory_id
                continue

            items = call.payload.get(response_field)
            if items is None:
                # A successful HTTP response with no expected record field is a
                # response-shape failure, not a successful empty page.
                error = dict(family_error("response_field"))
                result.steps.append(
                    AgentStep(
                        action="list_records",
                        request=parameters,
                        result=call.payload,
                        ok=False,
                        error=error,
                    )
                )
                result.final_error = str(error["error_message"])
                if memory is None:
                    return result
                recovered, base, token, response_field = self._recover(
                    memory,
                    base,
                    token,
                    response_field,
                    ToolCallResult(ok=False, payload=call.payload, error=error),
                )
                if not recovered:
                    return result
                result.retries += 1
                result.used_memory_id = memory.memory_id
                continue

            collected.extend(items)
            result.ordered_record_ids = [str(record.get("id")) for record in collected]
            token = call.payload.get(parameter_name)
            if not call.payload.get("has_more"):
                if getattr(environment, "requires_exact_order", False):
                    expected = getattr(
                        environment, "expected_records", lambda: environment.records
                    )()
                    result.success = collected == expected
                else:
                    result.success = len(collected) == len(environment.records)
                if not result.success:
                    result.final_error = (
                        f"collected {len(collected)}/{len(environment.records)} records"
                    )
                return result
        return result

    def _apply_proactive(
        self,
        memory: SelectedMemory,
        base: dict[str, Any],
        response_field: str,
        parameter_name: str,
    ) -> tuple[bool, dict[str, Any], str, str]:
        """Apply pre-call typed actions for silent-drift scenarios."""
        applied = False
        next_base = dict(base)
        next_response_field = response_field
        next_parameter = parameter_name
        for action in memory.actions:
            action_type = action.get("action_type")
            parameters = action.get("parameters") or {}
            if action_type == ActionType.MAP_PARAMETER.value:
                old = parameters.get("from")
                new = parameters.get("to")
                if old and new and old in next_base:
                    value = next_base.pop(old)
                    if value not in {None, "", "start"}:
                        next_base[new] = value
                elif old:
                    next_base.pop(old, None)
                if isinstance(new, str) and new:
                    next_parameter = new
                applied = True
            elif action_type == ActionType.SUPPLY_REQUIRED_PARAMETER.value:
                for key, value in parameters.items():
                    if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                        next_base[key] = value
                applied = True
            elif action_type == ActionType.MAP_PARAMETER_VALUE.value:
                field = parameters.get("field")
                target = parameters.get("to")
                if isinstance(field, str) and target is not None:
                    next_base[field] = target
                applied = True
            elif action_type == ActionType.NORMALIZE_TIMESTAMP.value:
                field = parameters.get("field", "since")
                value = parameters.get("value", "2026-01-01T00:00:00Z")
                next_base[str(field)] = value
                applied = True
            elif action_type == ActionType.MAP_RESPONSE_FIELD.value:
                next_response_field = str(parameters.get("field", "data"))
                applied = True
        return applied, next_base, next_response_field, next_parameter

    def _recover(
        self,
        memory: SelectedMemory,
        base: dict[str, Any],
        token: str | None,
        response_field: str,
        failed_call: ToolCallResult,
    ) -> tuple[bool, dict[str, Any], str | None, str]:
        """Apply only allowlisted typed actions; never execute free-form text."""
        error = failed_call.error or {}
        error_code = str(error.get("error_code", ""))
        next_base = dict(base)
        next_token = token
        next_response_field = response_field
        for action in memory.actions:
            action_type = action.get("action_type")
            parameters = action.get("parameters") or {}
            if action_type == ActionType.RETRY_WITH_BACKOFF.value and error_code in {
                "RATE_LIMITED",
                "TIMEOUT",
                "UNAVAILABLE",
            }:
                return True, next_base, next_token, next_response_field
            if action_type == ActionType.REFRESH_PAGINATION_CURSOR.value:
                next_base.pop("cursor", None)
                next_base.pop("page_token", None)
                return True, next_base, None, next_response_field
            if action_type == ActionType.MAP_PARAMETER.value:
                old = parameters.get("from")
                new = parameters.get("to")
                if old and new and old in next_base:
                    value = next_base.pop(old)
                    if value not in {None, "", "start"}:
                        next_base[new] = value
                elif old:
                    next_base.pop(old, None)
                return True, next_base, None, next_response_field
            if action_type == ActionType.REFRESH_AUTH_TOKEN.value:
                next_base["auth_token"] = str(parameters.get("token", "fresh-token"))
                return True, next_base, None, next_response_field
            if action_type == ActionType.SUPPLY_REQUIRED_PARAMETER.value:
                for key, value in parameters.items():
                    if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                        next_base[key] = value
                return True, next_base, None, next_response_field
            if action_type == ActionType.MAP_PARAMETER_VALUE.value:
                field = parameters.get("field")
                target = parameters.get("to")
                if isinstance(field, str) and target is not None:
                    next_base[field] = target
                return True, next_base, None, next_response_field
            if action_type == ActionType.NORMALIZE_TIMESTAMP.value:
                field = parameters.get("field", "since")
                value = parameters.get("value", "2026-01-01T00:00:00Z")
                next_base[str(field)] = value
                return True, next_base, None, next_response_field
            if action_type == ActionType.MAP_RESPONSE_FIELD.value:
                target = parameters.get("field", "data")
                next_response_field = str(target)
                return True, next_base, None, next_response_field
            if action_type in {
                ActionType.REQUEST_MISSING_FIELD.value,
                ActionType.HALT.value,
            }:
                return False, next_base, next_token, next_response_field
        return False, next_base, next_token, next_response_field

    def run_task(
        self,
        suite: MockApiSuite,
        version: str = "1",
        *,
        memory: SelectedMemory | None = None,
        expire_first_cursor: bool = False,
        failure_family: str | None = None,
    ) -> AgentRunResult:
        environment = suite.get(version, failure_family)
        if expire_first_cursor:
            # Force the failure mode used by the seeded demo.
            environment.expire_cursor("cur_1_0001")
        return self.run(environment, memory=memory)


def _paginations_parameter(environment: PaginatedApiEnvironment) -> str:
    if environment.failure_family == "renamed_parameter" and environment.version == "2":
        return "page_token"
    return "cursor"
