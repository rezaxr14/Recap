"""Applicability evaluation of preconditions and MemoryShift compatibility."""

from __future__ import annotations

from recap.domain.enums import CompatibilityStatus, PredicateResult
from recap.domain.memories import ExperienceMemory
from recap.domain.predicates import ApplicabilityResult, evaluate_all
from recap.domain.retrieval import RetrievalContext


def build_state_snapshot(
    memory: ExperienceMemory, context: RetrievalContext
) -> dict[str, object]:
    """Bounded state used for predicate evaluation.

    Includes caller state plus environment facts the engine knows. No secrets.
    """
    state: dict[str, object] = dict(context.state)
    state.setdefault("environment_id", context.environment_id)
    state.setdefault("environment_fingerprint", context.environment_fingerprint)
    state.setdefault("environment_version", context.environment_version)
    state.setdefault("environment_type", context.environment_type)
    state.setdefault("task_family", context.task_family)
    state.setdefault("error_type", context.error_type)
    state.setdefault("error_code", context.error_code)
    if context.tool_names:
        state.setdefault("tool_names", context.tool_names)
    if context.tool_versions:
        state.setdefault("tool_versions", context.tool_versions)
    if context.tool_schema_fingerprints:
        state.setdefault("tool_schema_fingerprints", context.tool_schema_fingerprints)
    return state


def evaluate_applicability(
    memory: ExperienceMemory, context: RetrievalContext
) -> ApplicabilityResult:
    state = build_state_snapshot(memory, context)
    return evaluate_all(memory.content.preconditions, state)


def is_operationally_applicable(result: ApplicabilityResult) -> bool:
    """Operational selection requires satisfied preconditions.

    Unknown is not satisfied. Low-risk guidance may be shown separately by a
    caller that explicitly opts in, but the default operational path fails closed.
    """
    return result.result is PredicateResult.SATISFIED


def combine_compatibility(
    memory: ExperienceMemory,
    context: RetrievalContext,
    stored: CompatibilityStatus | None,
) -> tuple[CompatibilityStatus, str]:
    """Resolve compatibility for this request from stored state + fingerprints."""
    if stored is CompatibilityStatus.INCOMPATIBLE:
        return CompatibilityStatus.INCOMPATIBLE, "stored compatibility is incompatible"
    if (
        memory.environment_fingerprint
        and context.environment_fingerprint
        and memory.environment_fingerprint != context.environment_fingerprint
    ):
        # Fingerprint changed: at best suspect unless the schema hash also matches.
        if (
            memory.content.tool_schema_fingerprints
            and context.tool_schema_fingerprints
            and all(
                context.tool_schema_fingerprints.get(tool) == digest
                for tool, digest in memory.content.tool_schema_fingerprints.items()
                if tool in context.tool_schema_fingerprints
            )
        ):
            return CompatibilityStatus.SUSPECT, "environment fingerprint changed but tool schemas match"
        return CompatibilityStatus.SUSPECT, "environment fingerprint differs from memory origin"
    if stored is CompatibilityStatus.SUSPECT:
        return CompatibilityStatus.SUSPECT, "stored compatibility is suspect"
    if memory.environment_fingerprint and memory.environment_fingerprint == context.environment_fingerprint:
        return CompatibilityStatus.COMPATIBLE, "environment fingerprint matches"
    if not memory.environment_fingerprint:
        return CompatibilityStatus.UNKNOWN, "memory has no environment fingerprint"
    return stored or CompatibilityStatus.UNKNOWN, "no explicit compatibility evidence"
