from __future__ import annotations

import pytest

from recap.recall.rrf import RankedList, reciprocal_rank_fusion


def test_rrf_promotes_documents_ranked_in_multiple_branches() -> None:
    lexical = RankedList(source="bm25", ids=["a", "b", "c"])
    vector = RankedList(source="vector", ids=["b", "a", "d"])
    fused = reciprocal_rank_fusion([lexical, vector], k=60)
    assert fused[0].doc_id == "a"
    assert fused[0].ranks["bm25"] == 1
    assert fused[0].ranks["vector"] == 2
    assert {hit.doc_id for hit in fused} == {"a", "b", "c", "d"}


def test_rrf_does_not_invent_ranks_for_absent_documents() -> None:
    lexical = RankedList(source="bm25", ids=["a"])
    vector = RankedList(source="vector", ids=["b"])
    fused = reciprocal_rank_fusion([lexical, vector], k=60)
    by_id = {hit.doc_id: hit for hit in fused}
    assert "vector" not in by_id["a"].ranks
    assert by_id["a"].score == pytest.approx(1 / 61)
    assert by_id["b"].score == pytest.approx(1 / 61)


def test_rrf_strong_single_branch_can_rank_highly() -> None:
    lexical = RankedList(source="bm25", ids=["exact", "x", "y"])
    vector = RankedList(source="vector", ids=["z", "w", "v"])
    fused = reciprocal_rank_fusion([lexical, vector], k=60)
    assert fused[0].doc_id in {"exact", "z"}
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([RankedList(source="a", ids=["x"])], k=0)


def test_rrf_applies_weights() -> None:
    lexical = RankedList(source="bm25", ids=["a"], weight=2.0)
    vector = RankedList(source="vector", ids=["b"], weight=1.0)
    fused = reciprocal_rank_fusion([lexical, vector], k=60)
    by_id = {hit.doc_id: hit for hit in fused}
    assert by_id["a"].score > by_id["b"].score
