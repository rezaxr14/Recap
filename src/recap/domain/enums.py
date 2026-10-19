"""Canonical domain enums.

String values are persisted and used in API payloads; never renumber or rename
without a migration.
"""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    INVALID = "invalid"


class OutcomeStatus(StrEnum):
    """Verified application outcome for a memory.

    Only VERIFIED_SUCCESS / VERIFIED_FAILURE update reliability counters.
    """

    VERIFIED_SUCCESS = "verified_success"
    VERIFIED_FAILURE = "verified_failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    TASK_OBSERVATION = "task_observation"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_CALL_SUCCEEDED = "tool_call_succeeded"
    TOOL_CALL_FAILED = "tool_call_failed"
    STATE_SNAPSHOT = "state_snapshot"
    MEMORY_RETRIEVAL_REQUESTED = "memory_retrieval_requested"
    MEMORY_CANDIDATE_RETURNED = "memory_candidate_returned"
    MEMORY_SELECTED = "memory_selected"
    MEMORY_REJECTED = "memory_rejected"
    MEMORY_APPLIED = "memory_applied"
    REPAIR_CANDIDATE_CREATED = "repair_candidate_created"
    REPAIR_VALIDATION_STARTED = "repair_validation_started"
    REPAIR_VALIDATION_FINISHED = "repair_validation_finished"
    POSTCONDITION_CHECKED = "postcondition_checked"
    RUN_FINISHED = "run_finished"


class MemoryStatus(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    ACTIVE = "active"
    SUSPECT = "suspect"
    QUARANTINED = "quarantined"
    RETIRED = "retired"
    REJECTED = "rejected"


class CreatedFrom(StrEnum):
    MANUAL = "manual"
    REPAIR_COMPILER = "repair_compiler"
    IMPORTED = "imported"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ValidationStatus(StrEnum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"
    REQUIRES_REVIEW = "requires_review"


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    UNKNOWN = "unknown"
    SUSPECT = "suspect"
    INCOMPATIBLE = "incompatible"


class RetrievalMode(StrEnum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"
    OUTCOME_AWARE = "outcome_aware"
    # Chunk-level equivalents. Same eligibility constraints as their card
    # counterparts; only the indexed granule changes.
    BM25_CHUNK = "bm25_chunk"
    VECTOR_CHUNK = "vector_chunk"
    HYBRID_RRF_CHUNK = "hybrid_rrf_chunk"
    OUTCOME_AWARE_CHUNK = "outcome_aware_chunk"


class ChunkType(StrEnum):
    MEMORY_CARD = "memory_card"
    FAILURE_WINDOW = "failure_window"
    ACTION_OBSERVATION = "action_observation"
    VALIDATION_EVIDENCE = "validation_evidence"
    # Evaluation baseline only: fixed windows that ignore event boundaries.
    NAIVE_FIXED = "naive_fixed"


class PredicateResult(StrEnum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class SelectionStatus(StrEnum):
    SELECTED = "selected"
    NO_SAFE_MEMORY = "no_safe_memory"
    DEGRADED = "degraded"


class RevalidationReason(StrEnum):
    VERSION_CHANGE = "version_change"
    ENVIRONMENT_CHANGED = "environment_changed"
    RELIABILITY_DROP = "reliability_drop"
    AGE_REVIEW = "age_review"
    CONFLICT = "conflict"
    MANUAL = "manual"


class RevalidationState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ActionType(StrEnum):
    """Allowlisted recovery action templates.

    RECAP never executes these itself; they are guidance for the host agent.
    """

    RETRY_WITH_BACKOFF = "retry_with_backoff"
    REFRESH_PAGINATION_CURSOR = "refresh_pagination_cursor"
    MAP_PARAMETER = "map_parameter"
    REQUEST_MISSING_FIELD = "request_missing_field"
    VERIFY_RESOURCE_EXISTS = "verify_resource_exists"
    HALT = "halt"


class SideEffectClass(StrEnum):
    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class EvaluationSetting(StrEnum):
    FROZEN_MEMORY = "frozen_memory"
    ONLINE_ADAPTATION = "online_adaptation"


class RepairCandidateStatus(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PROMOTED = "promoted"
