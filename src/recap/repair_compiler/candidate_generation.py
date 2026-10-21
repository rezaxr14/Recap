"""Repair candidate generation behind a provider protocol.

The default generator is deterministic and rule-based so the whole loop is
testable without an LLM. An LLM provider can be plugged in, but its output is
validated identically: schema, allowlist, evidence references, and size limits.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from recap.config import RepairCompilerSettings
from recap.domain.enums import ActionType, RiskLevel, SideEffectClass
from recap.domain.predicates import Predicate
from recap.domain.repairs import DiagnosisHypothesis, FailureWindow, RepairCandidate
from recap.domain.runs import RunEvent
from recap.domain.strategies import RepairStrategy, StrategyAction
from recap.errors import ValidationError


class CandidateDraft(BaseModel):
    """Strict generation schema. No free-form shell/code fields exist."""

    summary: str = Field(min_length=1, max_length=2000)
    hypothesis: str = Field(min_length=1, max_length=2000)
    evidence_event_ids: list[str] = Field(default_factory=list)
    preconditions: list[Predicate]
    actions: list[StrategyAction]
    postconditions: list[Predicate]
    limitations: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    confidence_rationale: str = ""
    validation_steps: list[str] = Field(default_factory=list)


class CandidateGenerator(Protocol):
    name: str

    def generate(
        self,
        *,
        window: FailureWindow,
        environment: dict[str, Any],
        events: Sequence[RunEvent] | None = None,
    ) -> list[CandidateDraft]: ...


def _family_draft(
    *,
    evidence: list[str],
    tool: str,
    schema_hash: str,
    error_code: str,
    summary: str,
    hypothesis: str,
    action: StrategyAction,
    limitations: list[str] | None = None,
    postcondition_field: str = "records_collected",
) -> CandidateDraft:
    """Build a typed candidate for a schema-versioned failure family."""
    postcondition = (
        Predicate(
            field="tool_call_succeeded",
            operator="eq",
            value=True,
            description="retried call succeeds",
        )
        if postcondition_field == "tool_call_succeeded"
        else Predicate(
            field="records_collected",
            operator="gte",
            value=1,
            description="at least one record retrieved after recovery",
        )
    )
    return CandidateDraft(
        summary=summary,
        hypothesis=hypothesis,
        evidence_event_ids=evidence,
        preconditions=[
            Predicate(
                field="tool_schema_fingerprints." + tool,
                operator="eq",
                value=schema_hash,
                description="tool schema matches the validated version",
            ),
            Predicate(
                field="error_code",
                operator="eq",
                value=error_code,
                description="failure matches this recovery family",
            ),
        ],
        actions=[action],
        postconditions=[postcondition],
        limitations=limitations or ["Only valid for the schema version cited in the evidence."],
        risk_factors=["schema_dependent", "read_only_operation"],
        confidence_rationale=f"Deterministic rule matched {error_code}.",
        validation_steps=["replay_failure_window", "verify_postconditions"],
    )


class DeterministicGenerator:
    """Rule-based generator covering the supported failure categories.

    Returns no draft when the evidence is insufficient (``no_repair_candidate``).
    """

    name = "deterministic-generator-v1"

    def generate(
        self,
        *,
        window: FailureWindow,
        environment: dict[str, Any],
        events: Sequence[RunEvent] | None = None,
    ) -> list[CandidateDraft]:
        if window.incomplete:
            return []
        category = window.failure_category
        tool = window.tool_name or "list_records"
        schema_hash = str(environment.get("tool_schema_hash", ""))
        version = str(environment.get("environment_version", "1"))
        evidence = window.event_ids[:10]

        if category == "invalid_pagination_cursor":
            return [
                CandidateDraft(
                    summary=(
                        "Discard the invalid pagination cursor and restart pagination from the "
                        "first page before requesting subsequent pages."
                    ),
                    hypothesis=(
                        "The request reused an expired or unknown cursor. Fetching a fresh "
                        "cursor from the first page should restore pagination."
                    ),
                    evidence_event_ids=evidence,
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
                            value=["INVALID_CURSOR", "EXPIRED_CURSOR"],
                            description="failure is a cursor validity error",
                        ),
                    ],
                    actions=[
                        StrategyAction(
                            action_type=ActionType.REFRESH_PAGINATION_CURSOR,
                            tool_name=tool,
                            description="Restart pagination and obtain a fresh cursor before continuing.",
                            side_effect_class=SideEffectClass.READ_ONLY,
                        )
                    ],
                    postconditions=[
                        Predicate(
                            field="records_collected",
                            operator="gte",
                            value=1,
                            description="at least one record retrieved after recovery",
                        )
                    ],
                    limitations=["Only valid for cursor/token pagination APIs."],
                    risk_factors=["read_only_operation"],
                    confidence_rationale="Deterministic rule matched an INVALID_CURSOR failure.",
                    validation_steps=["replay_failure_window", "verify_postconditions"],
                )
            ]
        if category in {"transient_rate_limit", "transient_timeout"}:
            return [
                CandidateDraft(
                    summary="Retry the read-only tool call with bounded exponential backoff.",
                    hypothesis="The upstream service rate-limited a transient read operation.",
                    evidence_event_ids=evidence,
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
                            value=["RATE_LIMITED", "TIMEOUT", "UNAVAILABLE"],
                            description="failure is classified as transient",
                        ),
                    ],
                    actions=[
                        StrategyAction(
                            action_type=ActionType.RETRY_WITH_BACKOFF,
                            tool_name=tool,
                            description="Retry the read-only call up to 3 times with backoff.",
                            side_effect_class=SideEffectClass.READ_ONLY,
                            max_attempts=3,
                            backoff_ms=250,
                        )
                    ],
                    postconditions=[
                        Predicate(
                            field="tool_call_succeeded",
                            operator="eq",
                            value=True,
                            description="retried call succeeds",
                        )
                    ],
                    limitations=["Must not be used for write operations without host policy."],
                    risk_factors=["bounded_retry"],
                    confidence_rationale="Deterministic rule matched a transient error code.",
                    validation_steps=["replay_failure_window"],
                )
            ]
        if category == "tool_parameter_mismatch":
            return [
                CandidateDraft(
                    summary="Map the legacy parameter name to the current schema only if verified.",
                    hypothesis="The tool schema renamed a parameter between versions.",
                    evidence_event_ids=evidence,
                    preconditions=[
                        Predicate(
                            field="tool_schema_fingerprints." + tool,
                            operator="eq",
                            value=schema_hash,
                            description="tool schema matches the target version",
                        )
                    ],
                    actions=[
                        StrategyAction(
                            action_type=ActionType.MAP_PARAMETER,
                            tool_name=tool,
                            description="Translate the renamed parameter using the verified schema.",
                            parameters=(
                                {"from": "page_token", "to": "cursor"}
                                if version == "1"
                                else {"from": "cursor", "to": "page_token"}
                            ),
                            side_effect_class=SideEffectClass.READ_ONLY,
                        )
                    ],
                    postconditions=[
                        Predicate(
                            field="tool_call_succeeded",
                            operator="eq",
                            value=True,
                            description="call succeeds with the mapped parameter",
                        )
                    ],
                    limitations=["Only valid when the target schema is confirmed compatible."],
                    risk_factors=["schema_dependent"],
                    confidence_rationale="Deterministic rule matched an unknown-parameter failure.",
                    validation_steps=["check_schema_compatibility"],
                )
            ]
        if category == "authentication_expired":
            return [
                _family_draft(
                    evidence=evidence,
                    tool=tool,
                    schema_hash=schema_hash,
                    error_code=window.error_code or "TOKEN_EXPIRED",
                    summary="Refresh the expired credential, then retry the read-only call.",
                    hypothesis="The request used an expired access token.",
                    action=StrategyAction(
                        action_type=ActionType.REFRESH_AUTH_TOKEN,
                        tool_name=tool,
                        description="Refresh the token and retry with the current credential.",
                        parameters={"token": f"fresh-token-v{version}"},
                        side_effect_class=SideEffectClass.READ_ONLY,
                    ),
                    limitations=["Credential refresh must be authorized by the host."],
                )
            ]
        if category == "missing_idempotency_key":
            key = "idempotency_key" if version == "1" else "request_id"
            return [
                _family_draft(
                    evidence=evidence,
                    tool=tool,
                    schema_hash=schema_hash,
                    error_code=window.error_code or "MISSING_IDEMPOTENCY_KEY",
                    summary=f"Supply the required {key} for this API version.",
                    hypothesis=f"The API version requires a stable {key} on read requests.",
                    action=StrategyAction(
                        action_type=ActionType.SUPPLY_REQUIRED_PARAMETER,
                        tool_name=tool,
                        description=f"Add a bounded {key} value.",
                        parameters={key: "bench-request-key"},
                        side_effect_class=SideEffectClass.READ_ONLY,
                    ),
                    postcondition_field="tool_call_succeeded",
                )
            ]
        if category == "missing_required_parameter":
            parameter = "account" if version == "1" else "account_id"
            return [
                _family_draft(
                    evidence=evidence,
                    tool=tool,
                    schema_hash=schema_hash,
                    error_code=window.error_code or "MISSING_REQUIRED_PARAMETER",
                    summary=f"Supply the required {parameter} parameter.",
                    hypothesis=f"The API version added a required {parameter} parameter.",
                    action=StrategyAction(
                        action_type=ActionType.SUPPLY_REQUIRED_PARAMETER,
                        tool_name=tool,
                        description=f"Add a bounded {parameter} value.",
                        parameters={parameter: "acct-bench"},
                        side_effect_class=SideEffectClass.READ_ONLY,
                    ),
                    postcondition_field="tool_call_succeeded",
                )
            ]
        if category == "enum_value_mismatch":
            value = "created" if version == "1" else "created_at"
            return [
                _family_draft(
                    evidence=evidence,
                    tool=tool,
                    schema_hash=schema_hash,
                    error_code=window.error_code or "INVALID_ENUM_VALUE",
                    summary=f"Map the sort value to the current enum ({value}).",
                    hypothesis="The API renamed an enum value between versions.",
                    action=StrategyAction(
                        action_type=ActionType.MAP_PARAMETER_VALUE,
                        tool_name=tool,
                        description="Use the enum value declared by the verified schema.",
                        parameters={"field": "sort", "to": value},
                        side_effect_class=SideEffectClass.READ_ONLY,
                    ),
                    postcondition_field="tool_call_succeeded",
                )
            ]
        if category == "timestamp_format_mismatch":
            value = "2026/01/01" if version == "1" else "2026-01-01T00:00:00Z"
            return [
                _family_draft(
                    evidence=evidence,
                    tool=tool,
                    schema_hash=schema_hash,
                    error_code=window.error_code or "INVALID_DATE_FORMAT",
                    summary="Normalize the timestamp to the format required by this version.",
                    hypothesis="The API changed its accepted timestamp format.",
                    action=StrategyAction(
                        action_type=ActionType.NORMALIZE_TIMESTAMP,
                        tool_name=tool,
                        description="Use the timestamp format declared by the verified schema.",
                        parameters={"field": "since", "value": value},
                        side_effect_class=SideEffectClass.READ_ONLY,
                    ),
                    postcondition_field="tool_call_succeeded",
                )
            ]
        if category == "response_shape_changed":
            field = "items" if version == "1" else "data"
            return [
                _family_draft(
                    evidence=evidence,
                    tool=tool,
                    schema_hash=schema_hash,
                    error_code=window.error_code or "MISSING_RESPONSE_FIELD",
                    summary=f"Read records from the renamed response field '{field}'.",
                    hypothesis="The response payload moved records to a different field.",
                    action=StrategyAction(
                        action_type=ActionType.MAP_RESPONSE_FIELD,
                        tool_name=tool,
                        description="Read the field declared by the verified response shape.",
                        parameters={"field": field},
                        side_effect_class=SideEffectClass.READ_ONLY,
                    ),
                )
            ]
        return []


def validate_draft(
    draft: CandidateDraft,
    *,
    window: FailureWindow,
    settings: RepairCompilerSettings,
    allowed_event_ids: set[str],
) -> CandidateDraft:
    """Reject invented evidence, disallowed actions, and oversized strategies."""
    unknown_events = [event for event in draft.evidence_event_ids if event not in allowed_event_ids]
    if unknown_events:
        raise ValidationError(
            "candidate references events outside the source run",
            details={"unknown_event_ids": unknown_events[:5]},
        )
    if not draft.evidence_event_ids:
        raise ValidationError("candidate must cite at least one real event")
    if len(draft.actions) > settings.max_strategy_actions:
        raise ValidationError("candidate exceeds maximum strategy action count")
    allowed = {item.value for item in ActionType}
    for action in draft.actions:
        if action.action_type.value not in allowed:
            raise ValidationError(f"unsupported action type: {action.action_type}")
        if action.side_effect_class is SideEffectClass.DESTRUCTIVE:
            raise ValidationError("destructive actions cannot be auto-promoted")
    if not draft.preconditions or not draft.postconditions:
        raise ValidationError("candidate requires preconditions and postconditions")
    return draft


def build_candidate(
    draft: CandidateDraft,
    *,
    window: FailureWindow,
    workspace_id: str,
    environment_id: str,
    environment_fingerprint: str,
    generated_by: str,
    task_family: str = "",
) -> RepairCandidate:
    risk = RiskLevel.LOW
    if any(action.side_effect_class is SideEffectClass.WRITE for action in draft.actions):
        risk = RiskLevel.MEDIUM
    candidate = RepairCandidate(
        source_run_id=window.run_id,
        workspace_id=workspace_id,
        task_family=task_family,
        failure_window=window,
        diagnosis=DiagnosisHypothesis(
            summary=draft.hypothesis,
            evidence_event_ids=draft.evidence_event_ids,
            confidence_rationale=draft.confidence_rationale,
            generated_by=generated_by,
        ),
        preconditions=draft.preconditions,
        strategy=RepairStrategy(
            summary=draft.summary,
            actions=draft.actions,
            limitations=draft.limitations,
            risk_factors=draft.risk_factors,
        ),
        postconditions=draft.postconditions,
        risk_level=risk,
        risk_factors=draft.risk_factors,
        validation_plan=draft.validation_steps,
        environment_id=environment_id,
        environment_fingerprint=environment_fingerprint,
        generated_by=generated_by,
    )
    candidate.content_hash = candidate.compute_hash()
    return candidate


def provider_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON Schema accepted by strict OpenAI-compatible servers.

    Pydantic emits an untyped property for ``Any`` fields (``Predicate.value``):
    only ``default`` and ``title`` are present. LM Studio rejects that as an
    unrecognized schema even though it is valid JSON Schema. Fill in the full
    JSON type set without weakening Pydantic validation afterwards.
    """

    compatible = copy.deepcopy(schema)
    any_json_type = [
        "string",
        "number",
        "integer",
        "boolean",
        "array",
        "object",
        "null",
    ]

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                for value in properties.values():
                    if isinstance(value, dict) and not {
                        "type",
                        "$ref",
                        "anyOf",
                        "oneOf",
                        "allOf",
                        "enum",
                        "const",
                    } & set(value):
                        value["type"] = list(any_json_type)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(compatible)
    return compatible


@dataclass
class GenerationStats:
    """Counters for format acceptance and bounded retries.

    The acceptance script aggregates these values across runs. They are plain
    counters, not a claim that a candidate passed independent validation.
    """

    calls: int = 0
    attempts: int = 0
    format_failures: int = 0
    successes: int = 0
    last_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "attempts": self.attempts,
            "format_failures": self.format_failures,
            "successes": self.successes,
            "last_error": self.last_error,
        }


class LLMCandidateGenerator:
    """Adapter that turns structured LLM output into drafts.

    Model output is validated with the same strict schema and evidence checks as
    deterministic drafts. Malformed output is retried a bounded number of times
    and then raises; it is never silently promoted. The prompt contains only
    redacted, bounded event evidence from the failure window, and the model is
    told that its proposal is a hypothesis.
    """

    name = "llm-generator-v1"
    MAX_EVENT_CHARS = 800

    def __init__(self, provider: Any, *, max_attempts: int = 1) -> None:
        self.provider = provider
        self.max_attempts = max(1, max_attempts)
        self.stats = GenerationStats()

    def generate(
        self,
        *,
        window: FailureWindow,
        environment: dict[str, Any],
        events: Sequence[RunEvent] | None = None,
    ) -> list[CandidateDraft]:
        schema = provider_compatible_schema(CandidateDraft.model_json_schema())
        evidence = _bounded_evidence(window, events)
        base_prompt = (
            "Propose one minimal, typed recovery strategy for the failure below.\n"
            "Return only a JSON object matching the provided schema.\n"
            "Cite only event ids that appear in the evidence. Do not invent observations, "
            "do not claim that validation passed, and do not use destructive actions.\n\n"
            f"Failure window:\n{json.dumps(window.model_dump(mode='json'), default=str, indent=2)}\n\n"
            f"Environment:\n{json.dumps(environment, default=str, indent=2)}\n\n"
            f"Evidence events (already redacted):\n"
            f"{json.dumps(evidence, default=str, indent=2)}\n\n"
            f"JSON schema:\n{json.dumps(schema, indent=2)}"
        )
        self.stats.calls += 1
        last_error = ""
        for _attempt in range(1, self.max_attempts + 1):
            self.stats.attempts += 1
            user_prompt = base_prompt
            if last_error:
                user_prompt += (
                    "\n\nYour previous response did not match the schema. Return only a "
                    "single JSON object that matches it exactly. Validation error summary:\n"
                    f"{last_error[:1000]}"
                )
            raw = self.provider.generate_structured(
                system_prompt=(
                    "You are RepairCompiler. You propose a single recovery hypothesis from "
                    "execution evidence. The proposal is untrusted until it is independently "
                    "validated by running it in an approved environment. Use only allowlisted "
                    "typed actions and cite real evidence event ids. Return only JSON that "
                    "matches the supplied schema."
                ),
                user_prompt=user_prompt,
                schema=schema,
            )
            try:
                draft = CandidateDraft.model_validate(raw)
            except PydanticValidationError as exc:
                last_error = str(exc)
                self.stats.format_failures += 1
                self.stats.last_error = last_error
                continue
            self.stats.successes += 1
            return [draft]
        raise ValidationError(
            f"invalid candidate schema after {self.max_attempts} attempt(s): {last_error}"
        )


def _bounded_evidence(
    window: FailureWindow, events: Sequence[RunEvent] | None
) -> list[dict[str, Any]]:
    """Bounded, redacted evidence for the prompt.

    Payloads are already redacted at persistence. This function only bounds size
    and never adds facts that were absent from the trace.
    """
    if events is None:
        return [{"event_id": event_id} for event_id in window.event_ids[:20]]
    wanted = set(window.event_ids)
    evidence: list[dict[str, Any]] = []
    for event in events:
        if event.id not in wanted:
            continue
        payload = event.payload_redacted or {}
        serialized = json.dumps(payload, default=str, sort_keys=True)
        if len(serialized) > LLMCandidateGenerator.MAX_EVENT_CHARS:
            serialized = serialized[: LLMCandidateGenerator.MAX_EVENT_CHARS] + "...[truncated]"
        evidence.append(
            {
                "event_id": event.id,
                "sequence_number": event.sequence_number,
                "event_type": str(event.event_type),
                "tool_name": event.tool_name or "",
                "payload": serialized,
            }
        )
        if len(evidence) >= 20:
            break
    return evidence
