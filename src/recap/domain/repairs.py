"""Repair candidate, diagnosis, validation, and promotion domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from recap.domain.enums import (
    RepairCandidateStatus,
    RiskLevel,
    ValidationStatus,
)
from recap.domain.predicates import Predicate
from recap.domain.runs import new_id, utc_now
from recap.domain.strategies import RepairStrategy


class DiagnosisHypothesis(BaseModel):
    hypothesis_id: str = Field(default_factory=lambda: new_id("hyp"))
    summary: str = Field(min_length=1, max_length=2000)
    evidence_event_ids: list[str] = Field(default_factory=list)
    alternative_hypotheses: list[str] = Field(default_factory=list)
    falsification_evidence: str = ""
    confidence_rationale: str = ""
    generated_by: str = "deterministic-analyzer-v1"


class FailureWindow(BaseModel):
    run_id: str
    start_sequence: int
    end_sequence: int
    start_event_id: str
    end_event_id: str
    failure_category: str
    normalized_signature: str
    raw_signature: str = ""
    error_code: str = ""
    error_type: str = ""
    tool_name: str = ""
    event_ids: list[str] = Field(default_factory=list)
    retry_count: int = 0
    incomplete: bool = False
    notes: list[str] = Field(default_factory=list)


class RepairCandidate(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cand"))
    source_run_id: str
    workspace_id: str
    task_family: str = ""
    failure_window: FailureWindow
    diagnosis: DiagnosisHypothesis
    preconditions: list[Predicate] = Field(default_factory=list)
    strategy: RepairStrategy
    postconditions: list[Predicate] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    risk_factors: list[str] = Field(default_factory=list)
    validation_plan: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.NOT_STARTED
    status: RepairCandidateStatus = RepairCandidateStatus.DRAFT
    environment_id: str = ""
    environment_fingerprint: str = ""
    content_hash: str = ""
    generated_by: str = "deterministic-analyzer-v1"
    promotion_decision: str = ""
    promotion_reason: str = ""
    promoted_memory_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("preconditions")
    @classmethod
    def _non_empty_preconditions(cls, value: list[Predicate]) -> list[Predicate]:
        if not value:
            raise ValueError("repair candidates require at least one precondition")
        return value

    @field_validator("postconditions")
    @classmethod
    def _non_empty_postconditions(cls, value: list[Predicate]) -> list[Predicate]:
        if not value:
            raise ValueError("repair candidates require at least one postcondition")
        return value

    def compute_hash(self) -> str:
        from recap.domain.environments import canonical_json, prefixed_sha256

        payload = {
            "preconditions": [p.model_dump(mode="json") for p in self.preconditions],
            "strategy": self.strategy.model_dump(mode="json"),
            "postconditions": [p.model_dump(mode="json") for p in self.postconditions],
        }
        return prefixed_sha256(canonical_json(payload))


class ValidationResult(BaseModel):
    id: str = Field(default_factory=lambda: new_id("val"))
    candidate_id: str
    validator_name: str
    validator_version: str = "1"
    status: ValidationStatus
    attempt: int = 1
    environment_id: str = ""
    environment_fingerprint: str = ""
    candidate_hash: str = ""
    observed_outcome: str = ""
    detail: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)
    evidence_artifact: str = ""
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    error: str = ""


class PromotionDecision(BaseModel):
    approved: bool
    reason: str
    checks: list[dict[str, Any]] = Field(default_factory=list)
    memory_id: str | None = None
