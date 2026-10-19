from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from recap.config import RecallSettings, ReliabilitySettings
from recap.domain.enums import CompatibilityStatus, MemoryStatus, PredicateResult
from recap.domain.memories import ExperienceMemory, MemoryContent, OutcomeStats
from recap.domain.retrieval import RetrievalContext
from recap.recall.eligibility import check_eligibility
from recap.recall.utility_policy import (
    compute_features,
    compute_stats_after_outcome,
    should_select,
)
from tests.unit.test_domain import _content


def _memory(**overrides) -> ExperienceMemory:
    data = dict(
        workspace_id="ws-1",
        content=_content(),
        status=MemoryStatus.ACTIVE,
        environment_id="mock-api-v1",
        environment_fingerprint="sha256:env1",
        risk_level="low",
        last_validated_at=datetime.now(UTC) - timedelta(days=1),
    )
    data.update(overrides)
    return ExperienceMemory(**data)


def _context(**overrides) -> RetrievalContext:
    data = dict(
        workspace_id="ws-1",
        query="invalid cursor",
        task_family="api_pagination",
        environment_id="mock-api-v1",
        environment_fingerprint="sha256:env1",
        environment_version="1",
        tool_names=["list_records"],
        error_code="INVALID_CURSOR",
        failure_signature="ValidationError|INVALID_CURSOR|list_records|invalid cursor",
        tool_schema_fingerprints={"list_records": "sha256:abc"},
    )
    data.update(overrides)
    return RetrievalContext(**data)


def test_cross_workspace_memory_is_ineligible() -> None:
    memory = _memory(workspace_id="other-workspace")
    result = check_eligibility(
        memory,
        _context(),
        settings=RecallSettings(),
        mode="outcome_aware",  # type: ignore[arg-type]
        compatibility=CompatibilityStatus.COMPATIBLE,
    )
    assert result.eligible is False
    assert "wrong_workspace" in result.reasons


def test_incompatible_environment_is_hard_filtered() -> None:
    result = check_eligibility(
        _memory(),
        _context(),
        settings=RecallSettings(),
        mode="outcome_aware",  # type: ignore[arg-type]
        compatibility=CompatibilityStatus.INCOMPATIBLE,
    )
    assert result.eligible is False
    assert "incompatible_environment" in result.reasons


def test_schema_mismatch_is_hard_filtered() -> None:
    result = check_eligibility(
        _memory(),
        _context(tool_schema_fingerprints={"list_records": "sha256:different"}),
        settings=RecallSettings(),
        mode="outcome_aware",  # type: ignore[arg-type]
        compatibility=CompatibilityStatus.COMPATIBLE,
    )
    assert result.eligible is False
    assert any(reason.startswith("tool_schema_mismatch") for reason in result.reasons)


def test_retired_memory_is_ineligible() -> None:
    result = check_eligibility(
        _memory(status=MemoryStatus.RETIRED),
        _context(),
        settings=RecallSettings(),
        mode="outcome_aware",  # type: ignore[arg-type]
        compatibility=CompatibilityStatus.COMPATIBLE,
    )
    assert "retired_or_rejected_memory" in result.reasons


def test_utility_rewards_matching_signature_and_penalizes_incompatibility() -> None:
    matching = compute_features(
        memory=_memory(),
        context=_context(),
        relevance=0.9,
        applicability=PredicateResult.SATISFIED,
        satisfaction_ratio=1.0,
        unknown_ratio=0.0,
        compatibility=CompatibilityStatus.COMPATIBLE,
        reliability=ReliabilitySettings(),
    )
    incompatible = compute_features(
        memory=_memory(),
        context=_context(),
        relevance=0.9,
        applicability=PredicateResult.SATISFIED,
        satisfaction_ratio=1.0,
        unknown_ratio=0.0,
        compatibility=CompatibilityStatus.INCOMPATIBLE,
        reliability=ReliabilitySettings(),
    )
    assert matching.exact_signature_match == 1.0
    assert matching.total > incompatible.total


def test_high_similarity_cannot_override_incompatibility() -> None:
    features = compute_features(
        memory=_memory(),
        context=_context(),
        relevance=1.0,
        applicability=PredicateResult.SATISFIED,
        satisfaction_ratio=1.0,
        unknown_ratio=0.0,
        compatibility=CompatibilityStatus.INCOMPATIBLE,
        reliability=ReliabilitySettings(),
    )
    selected, reason = should_select(
        features,
        applicability=PredicateResult.SATISFIED,
        compatibility=CompatibilityStatus.INCOMPATIBLE,
        settings=RecallSettings(),
    )
    assert selected is False
    assert reason == "incompatible_environment"


def test_unknown_preconditions_are_not_selected() -> None:
    features = compute_features(
        memory=_memory(),
        context=_context(),
        relevance=0.9,
        applicability=PredicateResult.UNKNOWN,
        satisfaction_ratio=0.0,
        unknown_ratio=1.0,
        compatibility=CompatibilityStatus.COMPATIBLE,
        reliability=ReliabilitySettings(),
    )
    selected, reason = should_select(
        features,
        applicability=PredicateResult.UNKNOWN,
        compatibility=CompatibilityStatus.COMPATIBLE,
        settings=RecallSettings(),
    )
    assert selected is False
    assert reason == "preconditions_not_satisfied"


def test_low_utility_is_below_threshold() -> None:
    features = compute_features(
        memory=_memory(stats=OutcomeStats(verified_failures=9), last_validated_at=None),
        context=_context(error_code="OTHER", failure_signature="other"),
        relevance=0.05,
        applicability=PredicateResult.SATISFIED,
        satisfaction_ratio=1.0,
        unknown_ratio=0.0,
        compatibility=CompatibilityStatus.UNKNOWN,
        reliability=ReliabilitySettings(),
    )
    selected, reason = should_select(
        features,
        applicability=PredicateResult.SATISFIED,
        compatibility=CompatibilityStatus.UNKNOWN,
        settings=RecallSettings(no_safe_memory_threshold=0.35),
    )
    assert selected is False
    assert reason == "below_utility_threshold"


def test_unknown_outcome_does_not_update_reliability() -> None:
    stats = OutcomeStats()
    updated = compute_stats_after_outcome(stats, outcome="unknown", applied=False)
    assert updated.verified_successes == 0
    assert updated.verified_failures == 0
    assert updated.unknown == 1


def test_beta_posterior_reflects_verified_outcomes() -> None:
    stats = OutcomeStats(verified_successes=8, verified_failures=2)
    mean, stddev = stats.posterior(1.0, 1.0)
    assert mean == pytest.approx(9 / 12)
    assert stddev > 0


def test_reliability_degrades_with_failures() -> None:
    good = compute_features(
        memory=_memory(stats=OutcomeStats(verified_successes=9, verified_failures=1)),
        context=_context(),
        relevance=0.8,
        applicability=PredicateResult.SATISFIED,
        satisfaction_ratio=1.0,
        unknown_ratio=0.0,
        compatibility=CompatibilityStatus.COMPATIBLE,
        reliability=ReliabilitySettings(),
    )
    bad = compute_features(
        memory=_memory(stats=OutcomeStats(verified_successes=1, verified_failures=9)),
        context=_context(),
        relevance=0.8,
        applicability=PredicateResult.SATISFIED,
        satisfaction_ratio=1.0,
        unknown_ratio=0.0,
        compatibility=CompatibilityStatus.COMPATIBLE,
        reliability=ReliabilitySettings(),
    )
    assert good.total > bad.total
