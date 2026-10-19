"""Hard eligibility filters.

Hard filters exclude candidates that violate non-negotiable constraints. They
run before expensive reranking/utility scoring and never get overridden by a
high semantic score.
"""

from __future__ import annotations

from dataclasses import dataclass

from recap.config import RecallSettings
from recap.domain.enums import CompatibilityStatus, MemoryStatus, RetrievalMode
from recap.domain.memories import ExperienceMemory
from recap.domain.retrieval import RetrievalContext


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: list[str]
    details: list[str]


def check_eligibility(
    memory: ExperienceMemory,
    context: RetrievalContext,
    *,
    settings: RecallSettings,
    mode: RetrievalMode,
    allow_unvalidated: bool = False,
    compatibility: CompatibilityStatus | None = None,
    apply_compatibility_filters: bool = True,
) -> EligibilityResult:
    reasons: list[str] = []
    details: list[str] = []

    if memory.workspace_id != context.workspace_id:
        reasons.append("wrong_workspace")

    if memory.status is MemoryStatus.ACTIVE:
        pass
    elif memory.status is MemoryStatus.SUSPECT:
        if not settings.allow_suspect_operational_memories:
            reasons.append("suspect_memory_not_allowed")
    elif memory.status is MemoryStatus.QUARANTINED:
        reasons.append("quarantined_memory")
    elif memory.status in {MemoryStatus.RETIRED, MemoryStatus.REJECTED}:
        reasons.append("retired_or_rejected_memory")
    elif memory.status in {MemoryStatus.DRAFT, MemoryStatus.VALIDATING}:
        if not allow_unvalidated or mode not in {
            RetrievalMode.BM25,
            RetrievalMode.VECTOR,
            RetrievalMode.HYBRID_RRF,
        }:
            reasons.append("unvalidated_memory_not_allowed")
        else:
            details.append("unvalidated historical candidate (diagnostic mode)")

    # Known-incompatible environments are never operationally eligible.
    if apply_compatibility_filters:
        if compatibility is CompatibilityStatus.INCOMPATIBLE:
            reasons.append("incompatible_environment")
        elif compatibility is CompatibilityStatus.SUSPECT:
            if not settings.allow_suspect_operational_memories:
                reasons.append("suspect_compatibility")

    # Tool/schema mismatch is a hard filter when the memory declares them.
    if apply_compatibility_filters:
        for tool, expected_hash in memory.content.tool_schema_fingerprints.items():
            actual = context.tool_schema_fingerprints.get(tool)
            if actual and expected_hash and actual != expected_hash:
                reasons.append(f"tool_schema_mismatch:{tool}")
                details.append(f"expected {expected_hash}, got {actual}")
    if settings.require_tool_match and memory.content.tool_names:
        if context.tool_names and not set(memory.content.tool_names) & set(context.tool_names):
            reasons.append("tool_not_relevant")

    # A memory pinned to a different environment fingerprint is not applicable.
    if (
        apply_compatibility_filters
        and memory.environment_fingerprint
        and context.environment_fingerprint
        and memory.environment_fingerprint != context.environment_fingerprint
    ):
        reasons.append("environment_fingerprint_mismatch")

    return EligibilityResult(eligible=not reasons, reasons=reasons, details=details)
