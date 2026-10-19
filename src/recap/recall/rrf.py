"""Application-side Reciprocal Rank Fusion.

RRF(d) = sum_r weight_r / (k + rank_r(d)) with 1-based ranks. Documents missing
from a branch contribute nothing; ranks are never invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankedList:
    source: str
    ids: list[str]
    weight: float = 1.0
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class FusedHit:
    doc_id: str
    score: float
    ranks: dict[str, int]
    branch_scores: dict[str, float]


def reciprocal_rank_fusion(
    branches: list[RankedList],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[FusedHit]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    fused: dict[str, FusedHit] = {}
    for branch in branches:
        seen: set[str] = set()
        rank = 0
        for doc_id in branch.ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            hit = fused.get(doc_id)
            if hit is None:
                hit = FusedHit(doc_id=doc_id, score=0.0, ranks={}, branch_scores={})
                fused[doc_id] = hit
            hit.score += branch.weight / (k + rank)
            hit.ranks[branch.source] = rank
            if doc_id in branch.scores:
                hit.branch_scores[branch.source] = branch.scores[doc_id]
    results = sorted(
        fused.values(),
        key=lambda item: (-item.score, item.doc_id),
    )
    if limit is not None:
        results = results[:limit]
    for index, hit in enumerate(results, start=1):
        hit.ranks["fused"] = index
    return results
