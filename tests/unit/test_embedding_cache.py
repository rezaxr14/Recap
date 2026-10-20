"""Embedding-cache behavior, using SQLite so tests stay offline."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from recap.domain.runs import utc_now
from recap.errors import EmbeddingUnavailable
from recap.infrastructure.db.models import Base, EmbeddingCacheRow
from recap.infrastructure.db.session import Database, build_engine
from recap.infrastructure.embeddings.cache import (
    CachedEmbeddingProvider,
    SqlEmbeddingCache,
    embedding_text_hash,
)
from recap.infrastructure.embeddings.fake import FakeEmbeddingProvider


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self, dimension: int = 8) -> None:
        super().__init__(dimension=dimension)
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return super().embed_query(text)


class BrokenCache(SqlEmbeddingCache):
    def get_many(self, keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], list[float]]:
        raise RuntimeError("cache read failed")

    def put_many(self, entries: Sequence[EmbeddingCacheRow]) -> int:
        raise RuntimeError("cache write failed")


@pytest.fixture()
def cache() -> SqlEmbeddingCache:
    engine = build_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SqlEmbeddingCache(Database(engine))


def _provider(
    cache: SqlEmbeddingCache,
    *,
    fingerprint: str = "sha256:test-model",
    dimension: int = 8,
) -> tuple[CachedEmbeddingProvider, CountingEmbeddingProvider]:
    inner = CountingEmbeddingProvider(dimension=dimension)
    inner.embedding_fingerprint = fingerprint
    return CachedEmbeddingProvider(inner, cache), inner


def test_cache_reuses_vectors_and_dedupes_one_batch(cache: SqlEmbeddingCache) -> None:
    provider, inner = _provider(cache)

    first = provider.embed_documents(["alpha", "beta", "alpha"])

    assert inner.document_calls == [["alpha", "beta"]]
    assert first[0] == first[2]
    assert provider.stats.misses == 2
    assert provider.stats.writes == 2
    assert provider.stats.hits == 0

    second_provider, second_inner = _provider(cache)
    second = second_provider.embed_documents(["alpha", "beta", "alpha"])

    assert second_inner.document_calls == []
    assert second == first
    assert second_provider.stats.hits == 3
    assert second_provider.stats.misses == 0
    assert second_provider.embedding_usage()["hit_rate"] == 1.0


def test_query_and_document_roles_and_fingerprints_are_isolated(
    cache: SqlEmbeddingCache,
) -> None:
    document_provider, document_inner = _provider(cache, fingerprint="sha256:model-a")
    query_provider, query_inner = _provider(cache, fingerprint="sha256:model-a")

    document_provider.embed_documents(["same text"])
    query_provider.embed_query("same text")

    assert document_inner.document_calls == [["same text"]]
    assert query_inner.query_calls == ["same text"]
    assert query_inner.document_calls == []

    other_provider, other_inner = _provider(cache, fingerprint="sha256:model-b")
    other_provider.embed_documents(["same text"])

    assert other_inner.document_calls == [["same text"]]


def test_wrong_dimension_cache_hit_fails_closed(cache: SqlEmbeddingCache) -> None:
    fingerprint = "sha256:test-model"
    key = (fingerprint, embedding_text_hash("alpha", kind="document"))
    now = utc_now()
    cache.put_many(
        [
            EmbeddingCacheRow(
                embedding_fingerprint=key[0],
                text_hash=key[1],
                kind="document",
                dimension=3,
                provider="fake",
                model_id="fake-test-embedding",
                vector_json=[0.0, 0.0, 1.0],
                created_at=now,
                last_accessed_at=now,
                hit_count=0,
            )
        ]
    )
    provider, inner = _provider(cache, fingerprint=fingerprint)

    with pytest.raises(EmbeddingUnavailable) as excinfo:
        provider.embed_documents(["alpha"])

    assert "dimension" in excinfo.value.message
    assert inner.document_calls == []


def test_cache_outage_fails_open(cache: SqlEmbeddingCache) -> None:
    provider, inner = _provider(BrokenCache(cache.database))

    vectors = provider.embed_documents(["alpha"])

    assert len(vectors) == 1
    assert inner.document_calls == [["alpha"]]
    assert provider.stats.errors == 2
    assert provider.stats.embedded_texts == 1


def test_clear_removes_vectors(cache: SqlEmbeddingCache) -> None:
    provider, _inner = _provider(cache)
    provider.embed_documents(["alpha"])
    assert cache.stats()["entries"] == 1

    assert cache.clear() == 1
    assert cache.stats()["entries"] == 0
