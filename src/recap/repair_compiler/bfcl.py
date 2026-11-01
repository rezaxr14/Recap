"""BFCL-specific repair generation and validation.

The built-in deterministic generator maps ``page_token`` to ``cursor`` because
the mock pagination API is its only customer. BFCL functions have arbitrary
parameter names, so this generator derives the mapping from the *current*
schema and the recorded failed call: exactly one argument the schema does not
declare, exactly one required argument the call omitted.

The validator replays the recorded failure in the same BFCL environment,
applies the candidate's typed actions, and checks that the repaired call is
schema-valid. It never consults a gold label or the query's target function.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from recap.domain.enums import (
    ActionType,
    EventType,
    SideEffectClass,
    ValidationStatus,
)
from recap.domain.environments import EnvironmentFingerprint
from recap.domain.predicates import Predicate
from recap.domain.repairs import FailureWindow, RepairCandidate, ValidationResult
from recap.domain.strategies import StrategyAction
from recap.environments.bfcl import BFCLEnvironment
from recap.repair_compiler.candidate_generation import CandidateDraft

GENERATOR_NAME = "bfcl-schema-rename-v1"
VALIDATOR_NAME = "bfcl_sandbox"
VALIDATOR_VERSION = "1"


def _failed_request(window: FailureWindow, events: list[Any] | None, tool: str) -> dict[str, Any]:
    for event in events or []:
        if event.event_type is not EventType.TOOL_CALL_FAILED:
            continue
        if tool and event.tool_name and event.tool_name != tool:
            continue
        request = (event.payload_redacted or {}).get("request")
        if isinstance(request, dict):
            return request
    return {}


class SchemaRenameGenerator:
    """Map the undeclared argument to the missing required argument."""

    name = GENERATOR_NAME

    def __init__(self, schemas: dict[str, dict[str, Any]], *, tool_key: str = "name") -> None:
        self.schemas = schemas
        self.tool_key = tool_key

    def generate(
        self,
        *,
        window: FailureWindow,
        environment: dict[str, Any],
        events: list[Any] | None = None,
    ) -> list[CandidateDraft]:
        if window.incomplete or window.failure_category not in {
            "tool_parameter_mismatch",
            "missing_required_parameter",
        }:
            return []
        schema = self.schemas.get(str(environment.get("environment_id") or ""))
        if schema is None:
            return []
        params = schema.get("parameters") or {}
        properties = dict(params.get("properties") or {})
        required = list(params.get("required") or properties)
        tool = window.tool_name or str(schema.get(self.tool_key) or "")
        request = _failed_request(window, events, tool)
        if not request:
            return []
        unknown = sorted(set(request) - set(properties))
        missing = [name for name in required if name not in request]
        if len(unknown) != 1 or len(missing) != 1:
            return []
        old, new = unknown[0], missing[0]
        schema_hash = str(
            (environment.get("tool_schema_hashes") or {}).get(tool)
            or environment.get("tool_schema_hash")
            or ""
        )
        description = str(schema.get("description") or "").strip()
        limited = description[:400]
        return [
            CandidateDraft(
                summary=(
                    f"Tool {tool}: {limited} "
                    f"Map the legacy parameter '{old}' to '{new}' declared by the "
                    "current schema before retrying."
                ).strip(),
                hypothesis=(
                    f"The recorded call sent '{old}', which the current schema does not "
                    f"declare; it requires '{new}'. The remaining arguments already match."
                ),
                evidence_event_ids=window.event_ids[:10],
                preconditions=[
                    Predicate(
                        field="tool_schema_fingerprints." + tool,
                        operator="eq",
                        value=schema_hash,
                        description=f"{tool} schema matches the validated version",
                    ),
                    Predicate(
                        field="error_code",
                        operator="in",
                        value=["UNKNOWN_PARAMETER", "MISSING_REQUIRED_PARAMETER"],
                        description="failure is a parameter/schema mismatch",
                    ),
                ],
                actions=[
                    StrategyAction(
                        action_type=ActionType.MAP_PARAMETER,
                        tool_name=tool,
                        description=f"Rename '{old}' to '{new}' using the verified schema.",
                        parameters={"from": old, "to": new},
                        side_effect_class=SideEffectClass.READ_ONLY,
                    )
                ],
                postconditions=[
                    Predicate(
                        field="tool_call_succeeded",
                        operator="eq",
                        value=True,
                        description="call succeeds after the parameter mapping",
                    )
                ],
                limitations=["Only valid for the schema version cited in the evidence."],
                risk_factors=["schema_dependent", "read_only_operation"],
                confidence_rationale=(
                    "The current schema and the recorded request agree on exactly one rename."
                ),
                validation_steps=["replay_failure_window", "verify_postconditions"],
            )
        ]


class BFCLSandboxValidator:
    """Replay the failure, apply the typed repair, and re-run the call."""

    name = VALIDATOR_NAME

    def __init__(self, environments: dict[str, BFCLEnvironment]) -> None:
        self.environments = environments

    def validate(
        self,
        candidate: RepairCandidate,
        *,
        environment: EnvironmentFingerprint,
        attempt: int = 1,
    ) -> ValidationResult:
        started = datetime.now(UTC)
        checks: list[dict[str, Any]] = []
        sandbox = self.environments.get(environment.environment_id)

        def finish(
            status: ValidationStatus, detail: str, observed: str = "", error: str = ""
        ) -> ValidationResult:
            return ValidationResult(
                candidate_id=candidate.id,
                validator_name=self.name,
                validator_version=VALIDATOR_VERSION,
                status=status,
                attempt=attempt,
                environment_id=environment.environment_id,
                environment_fingerprint=environment.fingerprint,
                candidate_hash=candidate.content_hash or candidate.compute_hash(),
                observed_outcome=observed,
                detail=detail,
                checks=checks,
                started_at=started,
                finished_at=datetime.now(UTC),
                error=error,
            )

        checks.append(
            {
                "check": "environment_available",
                "passed": sandbox is not None,
                "detail": environment.environment_id,
            }
        )
        if sandbox is None:
            return finish(
                ValidationStatus.INCONCLUSIVE,
                "no sandbox environment registered for this fingerprint",
            )

        replay = sandbox.replay_failure()
        checks.append(
            {
                "check": "failure_reproduced",
                "passed": not replay.ok,
                "detail": str((replay.error or {}).get("error_code") or ""),
            }
        )
        if replay.ok:
            return finish(ValidationStatus.INCONCLUSIVE, "failure did not reproduce")

        repaired = dict(sandbox.failing_args)
        applied = 0
        for action in candidate.strategy.actions:
            if action.action_type is ActionType.MAP_PARAMETER:
                old = str(action.parameters.get("from") or "")
                new = str(action.parameters.get("to") or "")
                if old in repaired:
                    repaired[new] = repaired.pop(old)
                    applied += 1
            elif action.action_type is ActionType.SUPPLY_REQUIRED_PARAMETER:
                for key, value in action.parameters.items():
                    repaired.setdefault(str(key), value)
                    applied += 1
        checks.append(
            {
                "check": "typed_actions_applied",
                "passed": applied > 0,
                "detail": f"{applied} action(s)",
            }
        )
        if not applied:
            return finish(ValidationStatus.INCONCLUSIVE, "no applicable typed action")

        outcome = sandbox.call(repaired)
        checks.append(
            {
                "check": "postcondition_tool_call_succeeded",
                "passed": outcome.ok,
                "detail": str((outcome.error or {}).get("error_code") or "ok"),
            }
        )
        if not outcome.ok:
            return finish(
                ValidationStatus.FAILED,
                "repaired call still fails",
                observed=str((outcome.error or {}).get("error_message") or ""),
            )
        return finish(
            ValidationStatus.PASSED,
            "repaired call is schema-valid in the sandbox",
            observed=f"ok with {sorted(repaired)}",
        )


__all__ = [
    "BFCLSandboxValidator",
    "GENERATOR_NAME",
    "SchemaRenameGenerator",
    "VALIDATOR_NAME",
]
