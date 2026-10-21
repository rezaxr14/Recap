"""Reranker unit tests.

The reranker only changes candidate order; safety policy still runs after it.
These tests pin the deterministic ordering and the degradation path.
"""

from __future__ import annotations

import pytest

from recap.config import Settings
from recap.errors import ConfigurationError
from recap.recall.reranker import LexicalReranker, build_reranker, tokenize


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    tokens = tokenize("The API rejected cursor INVALID_CURSOR for the tenant")
    assert "the" not in tokens
    assert "for" not in tokens
    assert "invalid_cursor" in tokens
    assert "cursor" in tokens


def test_rerank_prefers_document_containing_distinctive_terms() -> None:
    reranker = LexicalReranker()
    query = "ValidationError INVALID_CURSOR list_records cursor expired"
    documents = [
        "Generic guidance about retrying API calls with exponential backoff.",
        "When list_records returns ValidationError INVALID_CURSOR, refresh the cursor.",
        "Pagination overview and tenant record counts.",
    ]

    ranked = reranker.rerank(query, documents, top_k=3)

    assert [index for index, _ in ranked][0] == 1
    assert ranked[0][1] > ranked[-1][1]


def test_rerank_is_stable_for_ties() -> None:
    reranker = LexicalReranker()
    documents = ["same text", "same text", "same text"]

    ranked = reranker.rerank("same text", documents, top_k=3)

    assert [index for index, _ in ranked] == [0, 1, 2]


def test_rerank_truncates_to_top_k() -> None:
    reranker = LexicalReranker()
    documents = ["cursor", "cursor cursor", "unrelated"]

    ranked = reranker.rerank("cursor", documents, top_k=2)

    assert len(ranked) == 2


def test_rerank_handles_empty_inputs() -> None:
    reranker = LexicalReranker()
    assert reranker.rerank("", ["a", "b"], top_k=2) == [(0, 0.0), (1, 0.0)]
    assert reranker.rerank("query", [], top_k=3) == []
    assert reranker.rerank("query", ["doc"], top_k=0) == []


def test_build_reranker_respects_config() -> None:
    assert build_reranker("none") is None
    assert build_reranker("lexical").name == "lexical-overlap-v1"  # type: ignore[union-attr]
    with pytest.raises(ConfigurationError, match="existing local model directory"):
        build_reranker("cross_encoder", model_path="does-not-exist")
    with pytest.raises(ValueError):
        build_reranker("unknown")


def test_cross_encoder_requires_an_explicit_local_path() -> None:
    settings = Settings()
    settings.recall.reranker = "cross_encoder"
    settings.recall.rerank_model_path = ""
    with pytest.raises(ConfigurationError, match="rerank_model_path"):
        settings.validate_consistency()
