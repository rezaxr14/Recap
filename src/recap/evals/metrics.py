"""Retrieval and outcome metrics computed from recorded data.

No hardcoded numbers. Every metric takes observed ranks/outcomes as input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(relevant & set(retrieved[:k]))
    return hits / len(relevant)


def precision_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(relevant & set(retrieved[:k])) / k


def reciprocal_rank(relevant: set[str], retrieved: Sequence[str]) -> float:
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    def gain(doc_id: str) -> float:
        return 1.0 if doc_id in relevant else 0.0

    dcg = sum(
        gain(doc_id) / math.log2(index + 1) for index, doc_id in enumerate(retrieved[:k], start=1)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile over observed values."""
    if not values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (p / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 7,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if not values:
        return 0.0, 0.0
    import random

    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(samples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(resample) / n)
    estimates.sort()
    lower_index = int((1 - confidence) / 2 * samples)
    upper_index = int((1 + confidence) / 2 * samples) - 1
    return estimates[lower_index], estimates[upper_index]


def paired_difference_ci(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 7,
) -> tuple[float, float, float]:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired comparison requires equal, non-empty samples")
    diffs = [t - b for b, t in zip(baseline, treatment, strict=True)]
    observed = sum(diffs) / len(diffs)
    lower, upper = bootstrap_ci(diffs, samples=samples, seed=seed)
    return observed, lower, upper
