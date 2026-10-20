"""Typed retrieval request, candidates, scoring breakdown, and decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from recap.domain.enums import (
    CompatibilityStatus,
    PredicateResult,
    RetrievalMode,
    SelectionStatus,
)
from recap.domain.runs import new_id, utc_now


class RetrievalContext(BaseModel):
    """Current state supplied by the caller. Redacted before persistence."""

    workspace_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    task_family: str = ""
    environment_id: str = ""
    environment_fingerprint: str = ""
    environment_version: str = ""
    environment_type: str = ""
    tool_names: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    tool_schema_fingerprints: dict[str, str] = Field(default_factory=dict)
    error_type: str = ""
    error_code: str = ""
    error_message: str = ""
    failure_signature: str = ""
    state: dict[str, Any] = Field(default_factory=dict)
    allow_unknown_state: bool = False


class RetrievalRequest(BaseModel):
    context: RetrievalContext
    mode: RetrievalMode = RetrievalMode.OUTCOME_AWARE
    candidate_k: int = Field(default=30, ge=1, le=200)
    final_k: int = Field(default=5, ge=1, le=50)
    allow_unvalidated: bool = False
    require_compatible_environment: bool = True
    latency_budget_ms: int = Field(default=800, ge=1, le=30_000)
    policy_version: str = "policy-v1"
    # Ablation selector for chunk modes. Normal operation is event_aware;
    # benchmark runs pin naive_fixed for the chunker ablation.
    chunk_strategy: str = "event_aware"
    run_id: str | None = None
    request_id: str = Field(default_factory=lambda: new_id("req"))

    @field_validator("chunk_strategy")
    @classmethod
    def _known_chunk_strategy(cls, value: str) -> str:
        if value not in {"", "event_aware", "naive_fixed"}:
            raise ValueError("chunk_strategy must be event_aware|naive_fixed|empty")
        return value

    @field_validator("final_k")
    @classmethod
    def _bounded_final(cls, value: int) -> int:
        if value > 50:
            raise ValueError("final_k too large")
        return value


class BranchHit(BaseModel):
    memory_id: str
    memory_version: int
    rank: int
    score: float | None = None
    source: Literal["bm25", "vector", "fused"]


class ScoreComponent(BaseModel):
    name: str
    value: float
    weight: float = 1.0
    explanation: str = ""


class UtilityFeatures(BaseModel):
    relevance: float = 0.0
    exact_signature_match: float = 0.0
    error_code_match: float = 0.0
    tool_match: float = 0.0
    task_family_match: float = 0.0
    environment_match: float = 0.0
    precondition_satisfaction: float = 0.0
    precondition_unknown_ratio: float = 0.0
    reliability_mean: float = 0.0
    reliability_uncertainty: float = 1.0
    evidence_strength: float = 0.0
    negative_evidence: float = 0.0
    validation_recency: float = 0.0
    side_effect_risk: float = 0.0
    compatibility_penalty: float = 0.0
    staleness_penalty: float = 0.0
    uncertainty_penalty: float = 0.0
    total: float = 0.0


class CandidateEvaluation(BaseModel):
    memory_id: str
    memory_version: int
    title: str
    status: str
    branch_ranks: dict[str, int] = Field(default_factory=dict)
    branch_scores: dict[str, float] = Field(default_factory=dict)
    fused_score: float = 0.0
    fused_rank: int = 0
    hard_filter_passed: bool = True
    rejection_reasons: list[str] = Field(default_factory=list)
    applicability_result: PredicateResult = PredicateResult.UNKNOWN
    applicability_reason: str = ""
    compatibility_status: CompatibilityStatus = CompatibilityStatus.UNKNOWN
    compatibility_reason: str = ""
    features: UtilityFeatures = Field(default_factory=UtilityFeatures)
    score_components: list[ScoreComponent] = Field(default_factory=list)
    final_score: float = 0.0
    selected: bool = False
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)
    retrieval_granule: str = "memory_card"


class SelectedMemory(BaseModel):
    memory_id: str
    memory_version: int
    title: str
    summary: str
    strategy_summary: str
    actions: list[dict[str, Any]] = Field(default_factory=list)
    preconditions: list[dict[str, Any]] = Field(default_factory=list)
    postconditions: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_run_ids: list[str] = Field(default_factory=list)
    evidence_trace_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    features: UtilityFeatures
    compatibility_status: CompatibilityStatus
    content_hash: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)
    retrieval_granule: str = "memory_card"


class RejectedCandidate(BaseModel):
    memory_id: str
    memory_version: int
    reason: str
    detail: str = ""


class RetrievalDecision(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dec"))
    request_id: str
    run_id: str | None = None
    memory_snapshot_id: str | None = None
    workspace_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    query: str
    mode: RetrievalMode
    status: SelectionStatus
    reason: str = ""
    policy_version: str = "policy-v1"
    index_generation: int = 0
    embedding_model_id: str = ""
    embedding_fingerprint: str = ""
    retrieval_granule: str = "memory_card"
    candidates: list[CandidateEvaluation] = Field(default_factory=list)
    selected_memories: list[SelectedMemory] = Field(default_factory=list)
    rejected_candidates: list[RejectedCandidate] = Field(default_factory=list)
    retrieval_degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Raw search response before policy selection."""

    decision_id: str
    candidates: list[CandidateEvaluation]
    mode: RetrievalMode
    retrieval_degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
