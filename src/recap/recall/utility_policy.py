"""Interpretable outcome-aware utility policy.

Retrieval relevance and expected utility stay separate. Every feature is logged;
the final score is a transparent weighted sum, not an opaque blend. The policy
returns ``no_safe_memory`` when nothing clears the threshold.
"""

from __future__ import annotations

from datetime import UTC, datetime

from recap.config import RecallSettings, ReliabilitySettings
from recap.domain.enums import CompatibilityStatus, PredicateResult
from recap.domain.memories import ExperienceMemory, OutcomeStats
from recap.domain.retrieval import RetrievalContext, ScoreComponent, UtilityFeatures

POLICY_VERSION = "utility-v1"

# Starting defaults, tuned on dev data only. Documented in docs/retrieval.md.
WEIGHTS = {
    "relevance": 0.24,
    "exact_signature_match": 0.18,
    "error_code_match": 0.10,
    "tool_match": 0.05,
    "task_family_match": 0.07,
    "environment_match": 0.06,
    "precondition_satisfaction": 0.12,
    "reliability": 0.10,
    "evidence_strength": 0.03,
    "negative_evidence": -0.10,
    "validation_recency": 0.03,
    "side_effect_risk": -0.06,
    "compatibility_penalty": -0.25,
    "staleness_penalty": -0.05,
    "uncertainty_penalty": -0.06,
}


def _recency_score(last_validated: datetime | None) -> float:
    if last_validated is None:
        return 0.0
    now = datetime.now(UTC)
    if last_validated.tzinfo is None:
        last_validated = last_validated.replace(tzinfo=UTC)
    age_days = max((now - last_validated).total_seconds() / 86_400, 0.0)
    # Half-life of 90 days; age alone never invalidates, it only adjusts priority.
    return 0.5 ** (age_days / 90.0)


def compute_features(
    *,
    memory: ExperienceMemory,
    context: RetrievalContext,
    relevance: float,
    applicability: PredicateResult,
    satisfaction_ratio: float,
    unknown_ratio: float,
    compatibility: CompatibilityStatus,
    reliability: ReliabilitySettings,
) -> UtilityFeatures:
    content = memory.content
    exact_signature = 0.0
    if context.failure_signature and content.failure_signature:
        exact_signature = (
            1.0 if context.failure_signature == content.failure_signature else 0.0
        )
    error_code_match = 0.0
    if context.error_code and content.failure_code:
        error_code_match = 1.0 if context.error_code == content.failure_code else 0.0
    tool_match = 0.0
    if content.tool_names:
        tool_match = 1.0 if set(content.tool_names) & set(context.tool_names) else 0.0
    task_family_match = 0.0
    if content.task_families:
        task_family_match = 1.0 if context.task_family in content.task_families else 0.0
    environment_match = 0.0
    if (
        memory.environment_fingerprint
        and context.environment_fingerprint
        and memory.environment_fingerprint == context.environment_fingerprint
    ):
        environment_match = 1.0

    mean, stddev = memory.stats.posterior(
        reliability.beta_prior_alpha, reliability.beta_prior_beta
    )
    observations = memory.stats.verified_successes + memory.stats.verified_failures
    evidence_strength = min(observations / 5.0, 1.0)
    negative = 0.0
    if observations:
        negative = memory.stats.verified_failures / observations
    validation_recency = _recency_score(memory.last_validated_at)

    risk_value = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(str(memory.risk_level), 0.5)
    compatibility_penalty = {
        CompatibilityStatus.COMPATIBLE: 0.0,
        CompatibilityStatus.UNKNOWN: 0.5,
        CompatibilityStatus.SUSPECT: 0.8,
        CompatibilityStatus.INCOMPATIBLE: 1.0,
    }[compatibility]
    staleness_penalty = 0.0
    if memory.last_validated_at is None:
        staleness_penalty = 0.5
    else:
        staleness_penalty = 1.0 - validation_recency
    uncertainty_penalty = stddev if observations < 5 else stddev * 0.5
    if applicability is PredicateResult.UNKNOWN:
        uncertainty_penalty = min(1.0, uncertainty_penalty + 0.3)

    features = UtilityFeatures(
        relevance=max(0.0, min(relevance, 1.0)),
        exact_signature_match=exact_signature,
        error_code_match=error_code_match,
        tool_match=tool_match,
        task_family_match=task_family_match,
        environment_match=environment_match,
        precondition_satisfaction=satisfaction_ratio,
        precondition_unknown_ratio=unknown_ratio,
        reliability_mean=mean,
        reliability_uncertainty=stddev,
        evidence_strength=evidence_strength,
        negative_evidence=negative,
        validation_recency=validation_recency,
        side_effect_risk=risk_value,
        compatibility_penalty=compatibility_penalty,
        staleness_penalty=staleness_penalty,
        uncertainty_penalty=uncertainty_penalty,
    )
    features.total = score_features(features)
    return features


def score_features(features: UtilityFeatures) -> float:
    raw = (
        WEIGHTS["relevance"] * features.relevance
        + WEIGHTS["exact_signature_match"] * features.exact_signature_match
        + WEIGHTS["error_code_match"] * features.error_code_match
        + WEIGHTS["tool_match"] * features.tool_match
        + WEIGHTS["task_family_match"] * features.task_family_match
        + WEIGHTS["environment_match"] * features.environment_match
        + WEIGHTS["precondition_satisfaction"] * features.precondition_satisfaction
        + WEIGHTS["reliability"] * features.reliability_mean
        + WEIGHTS["evidence_strength"] * features.evidence_strength
        + WEIGHTS["negative_evidence"] * features.negative_evidence
        + WEIGHTS["validation_recency"] * features.validation_recency
        + WEIGHTS["side_effect_risk"] * features.side_effect_risk
        + WEIGHTS["compatibility_penalty"] * features.compatibility_penalty
        + WEIGHTS["staleness_penalty"] * features.staleness_penalty
        + WEIGHTS["uncertainty_penalty"] * features.uncertainty_penalty
    )
    # Normalize into [0, 1] against the maximum achievable positive weight sum.
    positive = sum(value for value in WEIGHTS.values() if value > 0)
    return max(0.0, min(1.0, (raw + 0.3) / (positive + 0.3)))


def build_score_components(features: UtilityFeatures) -> list[ScoreComponent]:
    values = features.model_dump()
    components = []
    for name, weight in WEIGHTS.items():
        key = {
            "reliability": "reliability_mean",
        }.get(name, name)
        if key not in values:
            continue
        components.append(
            ScoreComponent(
                name=name,
                value=float(values[key]),
                weight=weight,
                explanation=_EXPLANATIONS.get(name, ""),
            )
        )
    components.append(
        ScoreComponent(name="total", value=features.total, weight=1.0, explanation="normalized final utility")
    )
    return components


_EXPLANATIONS = {
    "relevance": "fused retrieval relevance in [0,1]",
    "exact_signature_match": "normalized failure signature equality",
    "error_code_match": "exact error code equality",
    "tool_match": "declared tool overlaps current tool set",
    "task_family_match": "current task family declared applicable",
    "environment_match": "environment fingerprint equality",
    "precondition_satisfaction": "fraction of preconditions satisfied",
    "reliability": "Beta-Binomial posterior mean of verified outcomes",
    "evidence_strength": "verified outcome count saturating at 5",
    "negative_evidence": "share of verified outcomes that failed",
    "validation_recency": "90-day half-life recency of validation",
    "side_effect_risk": "risk level mapped to [0,1]",
    "compatibility_penalty": "MemoryShift compatibility severity",
    "staleness_penalty": "inverse validation recency",
    "uncertainty_penalty": "posterior stddev plus unknown-precondition penalty",
}


def should_select(
    features: UtilityFeatures,
    *,
    applicability: PredicateResult,
    compatibility: CompatibilityStatus,
    settings: RecallSettings,
) -> tuple[bool, str]:
    if applicability is not PredicateResult.SATISFIED:
        return False, "preconditions_not_satisfied"
    if compatibility is CompatibilityStatus.INCOMPATIBLE:
        return False, "incompatible_environment"
    if features.total < settings.no_safe_memory_threshold:
        return False, "below_utility_threshold"
    return True, "selected"


def compute_stats_after_outcome(
    stats: OutcomeStats,
    *,
    outcome: str,
    applied: bool,
) -> OutcomeStats:
    """Update counters; only verified outcomes affect reliability."""
    updated = stats.model_copy(deep=True)
    if applied:
        updated.applied += 1
    if outcome == "verified_success":
        updated.verified_successes += 1
        updated.consecutive_failures = 0
    elif outcome == "verified_failure":
        updated.verified_failures += 1
        updated.consecutive_failures += 1
    elif outcome == "partial":
        updated.partial += 1
    else:
        updated.unknown += 1
    updated.last_outcome_at = datetime.now(UTC)
    return updated
