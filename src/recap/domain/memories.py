"""Experience memory domain model, versioning, and lifecycle.

A memory is immutable once active except for lifecycle status and separately
recorded statistics. Content edits create a new version.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from recap.domain.enums import CreatedFrom, MemoryStatus, RiskLevel, SideEffectClass
from recap.domain.environments import canonical_json
from recap.domain.predicates import Predicate
from recap.domain.runs import new_id, utc_now
from recap.domain.strategies import RepairStrategy
from recap.errors import TransitionError

# Explicit transition table from the plan (section 8.1).
ALLOWED_TRANSITIONS: dict[MemoryStatus, set[MemoryStatus]] = {
    MemoryStatus.DRAFT: {MemoryStatus.VALIDATING},
    MemoryStatus.VALIDATING: {
        MemoryStatus.ACTIVE,
        MemoryStatus.REJECTED,
        MemoryStatus.DRAFT,
    },
    MemoryStatus.ACTIVE: {MemoryStatus.SUSPECT, MemoryStatus.RETIRED},
    MemoryStatus.SUSPECT: {
        MemoryStatus.ACTIVE,
        MemoryStatus.QUARANTINED,
        MemoryStatus.RETIRED,
    },
    MemoryStatus.QUARANTINED: {MemoryStatus.VALIDATING, MemoryStatus.RETIRED},
    MemoryStatus.RETIRED: set(),
    MemoryStatus.REJECTED: set(),
}

TERMINAL_STATUSES = {MemoryStatus.RETIRED, MemoryStatus.REJECTED}

OPERATIONAL_STATUSES = {MemoryStatus.ACTIVE, MemoryStatus.SUSPECT, MemoryStatus.QUARANTINED}


class MemoryContent(BaseModel):
    """Immutable content payload; hashing covers exactly these fields."""

    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=4000)
    failure_signature: str = ""
    failure_code: str = ""
    task_families: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    tool_schema_fingerprints: dict[str, str] = Field(default_factory=dict)
    # Field-scoped compatibility: hashes of only the schema paths this strategy
    # depends on. Empty means "fall back to the whole tool-schema hash".
    schema_path_hashes: dict[str, str] = Field(default_factory=dict)
    environment_constraints: dict[str, str] = Field(default_factory=dict)
    environment_type: str = ""
    preconditions: list[Predicate] = Field(default_factory=list)
    strategy: RepairStrategy
    postconditions: list[Predicate] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    counterexamples: list[str] = Field(default_factory=list)

    def compute_hash(self) -> str:
        payload = self.model_dump(mode="json")
        return f"sha256:{hashlib.sha256(canonical_json(payload).encode()).hexdigest()}"


class OutcomeStats(BaseModel):
    """Contextual Beta-Binomial evidence.

    ``verified_successes`` and ``verified_failures`` are the only counters used
    for reliability. Unknown/not-applied outcomes are tracked separately.
    """

    verified_successes: int = 0
    verified_failures: int = 0
    partial: int = 0
    unknown: int = 0
    retrieved: int = 0
    selected: int = 0
    applied: int = 0
    rejected: int = 0
    consecutive_failures: int = 0
    last_outcome_at: datetime | None = None

    def posterior(self, alpha0: float = 1.0, beta0: float = 1.0) -> tuple[float, float]:
        """Return (mean, stddev) of the Beta posterior."""
        alpha = alpha0 + self.verified_successes
        beta = beta0 + self.verified_failures
        total = alpha + beta
        mean = alpha / total
        variance = (alpha * beta) / (total * total * (total + 1))
        return mean, variance**0.5


class ExperienceMemory(BaseModel):
    id: str = Field(default_factory=lambda: new_id("mem"))
    workspace_id: str
    schema_version: str = "memory-v1"
    memory_version: int = 1
    supersedes_memory_id: str | None = None
    family_id: str = ""
    content: MemoryContent
    status: MemoryStatus = MemoryStatus.DRAFT
    status_reason: str = ""
    created_from: CreatedFrom = CreatedFrom.MANUAL
    risk_level: RiskLevel = RiskLevel.LOW
    access_scope: str = "workspace"
    evidence_trace_ids: list[str] = Field(default_factory=list)
    evidence_run_ids: list[str] = Field(default_factory=list)
    validation_record_ids: list[str] = Field(default_factory=list)
    environment_fingerprint: str = ""
    environment_id: str = ""
    embedding_model_id: str = ""
    embedding_model_version: str = ""
    embedding_dimension: int = 0
    retrieval_document_id: str = ""
    index_generation: int = 0
    content_hash: str = ""
    stats: OutcomeStats = Field(default_factory=OutcomeStats)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    activated_at: datetime | None = None
    last_validated_at: datetime | None = None

    @model_validator(mode="after")
    def _derive(self) -> ExperienceMemory:
        if not self.content_hash:
            self.content_hash = self.content.compute_hash()
        if not self.family_id:
            self.family_id = self.id
        if not self.retrieval_document_id:
            self.retrieval_document_id = f"{self.id}:v{self.memory_version}"
        return self

    @property
    def is_operationally_eligible(self) -> bool:
        return self.status is MemoryStatus.ACTIVE

    def transition(self, target: MemoryStatus, reason: str) -> ExperienceMemory:
        allowed = ALLOWED_TRANSITIONS[self.status]
        if target not in allowed:
            raise TransitionError(
                f"illegal memory transition {self.status} -> {target}",
                details={
                    "from": str(self.status),
                    "to": str(target),
                    "allowed": sorted(map(str, allowed)),
                },
            )
        if not reason.strip():
            raise TransitionError("memory transitions require a reason")
        copy = self.model_copy(deep=True)
        copy.status = target
        copy.status_reason = reason
        copy.updated_at = utc_now()
        if target is MemoryStatus.ACTIVE:
            copy.activated_at = copy.activated_at or utc_now()
        return copy

    @property
    def side_effect_class(self) -> SideEffectClass:
        return self.content.strategy.side_effect_class


def compute_content_hash(content: MemoryContent) -> str:
    return content.compute_hash()


def build_search_document(memory: ExperienceMemory) -> dict[str, Any]:
    """Canonical Elasticsearch document (see docs/retrieval.md).

    Only allowlisted fields are indexed. Trace payloads and secrets never are:
    every free-text field is passed through the redaction filter again at the
    indexing boundary, so a manually authored memory cannot leak a credential
    into the search index.
    """
    from recap.recorder.redaction import redact_text

    def clean(value: str) -> str:
        return redact_text(value)[0]

    content = memory.content
    preconditions_text = " ".join(
        f"{p.field} {p.operator} {p.value if p.value is not None else ''}"
        for p in content.preconditions
    )
    postconditions_text = " ".join(
        f"{p.field} {p.operator} {p.value if p.value is not None else ''}"
        for p in content.postconditions
    )
    strategy_text = " ".join(
        [content.strategy.summary]
        + [a.description for a in content.strategy.actions if a.description]
    )
    return {
        "memory_id": memory.id,
        "memory_version": memory.memory_version,
        "workspace_id": memory.workspace_id,
        "status": str(memory.status),
        "title": clean(content.title),
        "summary": clean(content.summary),
        "failure_signature": content.failure_signature,
        "failure_code": content.failure_code,
        "task_families": content.task_families,
        "tool_names": content.tool_names,
        "tool_versions": [f"{k}:{v}" for k, v in content.tool_versions.items()],
        "tool_schema_fingerprints": [
            f"{k}:{v}" for k, v in content.tool_schema_fingerprints.items()
        ],
        "environment_id": memory.environment_id,
        "environment_fingerprint": memory.environment_fingerprint,
        "environment_type": content.environment_type,
        "preconditions_text": clean(preconditions_text),
        "strategy_text": clean(strategy_text),
        "postconditions_text": clean(postconditions_text),
        "limitations_text": clean(" ".join(content.known_limitations)),
        "evidence_summary": f"runs={len(memory.evidence_run_ids)} traces={len(memory.evidence_trace_ids)}",
        "created_at": memory.created_at.isoformat(),
        "last_validated_at": memory.last_validated_at.isoformat()
        if memory.last_validated_at
        else None,
        "active_for_retrieval": memory.status is MemoryStatus.ACTIVE,
        "compatibility_status": "compatible" if memory.status is MemoryStatus.ACTIVE else "unknown",
        "embedding_model_id": memory.embedding_model_id,
        "content_hash": memory.content_hash,
        "risk_level": str(memory.risk_level),
        "side_effect_class": str(memory.side_effect_class),
        "verified_successes": memory.stats.verified_successes,
        "verified_failures": memory.stats.verified_failures,
        "index_generation": memory.index_generation,
    }


def build_embedding_text(memory: ExperienceMemory) -> str:
    from recap.recorder.redaction import redact_text

    content = memory.content
    lines = [
        f"Title: {content.title}",
        f"Failure signature: {content.failure_signature}",
        f"Failure code: {content.failure_code}",
        f"When applicable: {content.summary}",
        "Preconditions: "
        + "; ".join(
            p.description or f"{p.field} {p.operator} {p.value}" for p in content.preconditions
        ),
        f"Strategy: {content.strategy.summary}",
        "Expected postconditions: "
        + "; ".join(
            p.description or f"{p.field} {p.operator} {p.value}" for p in content.postconditions
        ),
        "Known limitations: " + "; ".join(content.known_limitations),
        "Tools: " + ", ".join(content.tool_names),
        "Environment: " + content.environment_type,
    ]
    return redact_text("\n".join(line for line in lines if line.strip()))[0]
