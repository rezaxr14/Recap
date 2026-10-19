"""Repair candidate generation behind a provider protocol.

The default generator is deterministic and rule-based so the whole loop is
testable without an LLM. An LLM provider can be plugged in, but its output is
validated identically: schema, allowlist, evidence references, and size limits.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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
        if category == "transient_rate_limit":
            return [
                CandidateDraft(
                    summary="Retry the read-only tool call with bounded exponential backoff.",
                    hypothesis="The upstream service rate-limited a transient read operation.",
                    evidence_event_ids=evidence,
                    preconditions=[
                        Predicate(
                            field="error_code",
                            operator="in",
                            value=["RATE_LIMITED", "TIMEOUT", "UNAVAILABLE"],
                            description="failure is classified as transient",
                        )
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
                            parameters={"from": "cursor", "to": "page_token"},
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


class LLMCandidateGenerator:
    """Adapter that turns structured LLM output into drafts.

    Model output is validated with the same strict schema and evidence checks as
    deterministic drafts. Malformed output raises; it is never silently promoted.
    The prompt contains only redacted, bounded event evidence from the failure
    window, and the model is told that its proposal is a hypothesis.
    """

    name = "llm-generator-v1"
    MAX_EVENT_CHARS = 800

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def generate(
        self,
        *,
        window: FailureWindow,
        environment: dict[str, Any],
        events: Sequence[RunEvent] | None = None,
    ) -> list[CandidateDraft]:
        schema = CandidateDraft.model_json_schema()
        evidence = _bounded_evidence(window, events)
        user_prompt = (
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
            raise ValidationError(f"invalid candidate schema: {exc}") from exc
        return [draft]


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
