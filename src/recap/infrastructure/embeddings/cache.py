"""Postgres-backed embedding cache.

The cache is an optimization, never a source of truth. Postgres
``memory_chunks`` still owns chunk text and the chunk index is still rebuildable;
deleting the cache merely makes the next rebuild re-embed.

Keys are deliberately narrow: ``(embedding_fingerprint, namespaced text hash)``.
The fingerprint includes model id, model version, dimension, normalization, and
task prefixes, while the text hash includes the document/query role. A hit can
therefore never mix vector spaces, and a hit with a different dimension is
treated as a hard error instead of being padded or truncated.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import delete, func, select

from recap.domain.runs import utc_now
from recap.errors import ConfigurationError, EmbeddingUnavailable
from recap.infrastructure.db.models import EmbeddingCacheRow
from recap.infrastructure.db.session import Database
from recap.logging import get_logger

logger = get_logger(__name__)

CACHE_KEY_VERSION = "v1"
DOCUMENT_KIND = "document"
QUERY_KIND = "query"


def embedding_text_hash(text: str, *, kind: str) -> str:
    """Return the namespaced cache hash for one provider input.

    ``kind`` separates document and query inputs even when a model uses no
    task prefixes, so a query vector is never silently treated as a stored
    document vector (and vice versa) for an asymmetric model.
    """
    if kind not in {DOCUMENT_KIND, QUERY_KIND}:
        raise ValueError(f"unknown embedding cache kind: {kind!r}")
    payload = f"{CACHE_KEY_VERSION}\x00{kind}\x00{text}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass
class EmbeddingCacheStats:
    """Per-process cache accounting. Never persisted as a metric of quality."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0
    embedded_texts: int = 0
    embedded_batches: int = 0

    def to_dict(self) -> dict[str, Any]:
        lookups = self.hits + self.misses
        return {
            "enabled": True,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "embedded_texts": self.embedded_texts,
            "embedded_batches": self.embedded_batches,
            "hit_rate": round(self.hits / lookups, 6) if lookups else 0.0,
        }


@dataclass
class _PendingBatch:
    """Unique missing inputs sharing one cache key's text.

    A batch can contain the same text twice; the provider must be called once
    and the vector reused for every occurrence.
    """

    key: tuple[str, str]
    kind: str
    indices: list[int] = field(default_factory=list)


class SqlEmbeddingCache:
    """Thin repository over ``embedding_cache`` using short transactions."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def get_many(self, keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], list[float]]:
        wanted = list(dict.fromkeys(keys))
        if not wanted:
            return {}
        found: dict[tuple[str, str], list[float]] = {}
        with self.database.session() as session:
            for key in wanted:
                row = session.get(EmbeddingCacheRow, key)
                if row is None:
                    continue
                found[key] = [float(value) for value in row.vector_json]
                row.hit_count = row.hit_count + 1
                row.last_accessed_at = utc_now()
        return found

    def put_many(self, entries: Sequence[EmbeddingCacheRow]) -> int:
        if not entries:
            return 0
        written = 0
        with self.database.session() as session:
            for entry in entries:
                key = (entry.embedding_fingerprint, entry.text_hash)
                row = session.get(EmbeddingCacheRow, key)
                if row is None:
                    session.add(entry)
                else:
                    # Preserve the original creation time and hit count; only
                    # refresh the vector identity and access time.
                    row.kind = entry.kind
                    row.dimension = entry.dimension
                    row.provider = entry.provider
                    row.model_id = entry.model_id
                    row.vector_json = entry.vector_json
                    row.last_accessed_at = entry.last_accessed_at
                written += 1
        return written

    def stats(self) -> dict[str, Any]:
        with self.database.session() as session:
            entries = session.scalar(select(func.count()).select_from(EmbeddingCacheRow)) or 0
            hits = session.scalar(select(func.coalesce(func.sum(EmbeddingCacheRow.hit_count), 0)))
            last_accessed = session.scalar(select(func.max(EmbeddingCacheRow.last_accessed_at)))
        return {
            "entries": int(entries),
            "recorded_hits": int(hits or 0),
            "last_accessed_at": last_accessed.isoformat() if last_accessed else None,
        }

    def clear(self) -> int:
        with self.database.session() as session:
            result = session.execute(delete(EmbeddingCacheRow))
            return int(result.rowcount or 0)


class CachedEmbeddingProvider:
    """Cache decorator around any embedding provider.

    Cache failures fail open: the wrapped provider is still called and the error
    is counted, because an unavailable optimization must not make retrieval or
    indexing unavailable. A cache hit with the wrong vector dimension fails
    closed, because silently returning a wrong-dimension vector would corrupt
    the search index.
    """

    def __init__(self, inner: Any, cache: SqlEmbeddingCache) -> None:
        self.inner = inner
        self.cache = cache
        self.stats = EmbeddingCacheStats()
        self.embedding_fingerprint = str(getattr(inner, "embedding_fingerprint", "") or "")
        if not self.embedding_fingerprint:
            raise ConfigurationError("embedding provider is missing an embedding_fingerprint")
        self.dimension = int(getattr(inner, "dimension", 0) or 0)
        if self.dimension < 1:
            raise ConfigurationError("embedding provider is missing a positive dimension")
        self.provider_type = str(getattr(inner, "provider_type", "unknown"))
        self.model_id = str(getattr(inner, "model_id", ""))
        self.model_version = str(getattr(inner, "model_version", ""))
        self.total_latency_ms = 0.0
        self._cache_failure_logged = False

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    # ----------------------------------------------------------- delegation
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        keys = [
            (self.embedding_fingerprint, embedding_text_hash(text, kind=DOCUMENT_KIND))
            for text in items
        ]
        cached = self._lookup(keys)
        vectors: list[list[float] | None] = [None] * len(items)
        pending: dict[tuple[str, str], _PendingBatch] = {}
        for index, key in enumerate(keys):
            vector = cached.get(key)
            if vector is None:
                batch = pending.setdefault(key, _PendingBatch(key=key, kind=DOCUMENT_KIND))
                batch.indices.append(index)
                continue
            self._validate(key, vector)
            vectors[index] = vector
            self.stats.hits += 1

        if pending:
            self.stats.misses += len(pending)
            unique_texts = [items[batch.indices[0]] for batch in pending.values()]
            embedded = self._embed_documents(unique_texts)
            if len(embedded) != len(unique_texts):
                raise EmbeddingUnavailable(
                    "embedding provider returned an unexpected number of vectors",
                    details={"expected": len(unique_texts), "received": len(embedded)},
                )
            entries: list[EmbeddingCacheRow] = []
            for batch, vector in zip(pending.values(), embedded, strict=True):
                self._validate(batch.key, vector)
                for index in batch.indices:
                    vectors[index] = vector
                entries.append(self._entry(batch.key, batch.kind, vector))
            self._store(entries)

        if any(vector is None for vector in vectors):
            raise EmbeddingUnavailable("embedding cache produced an incomplete result")
        return [vector for vector in vectors if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        key = (self.embedding_fingerprint, embedding_text_hash(text, kind=QUERY_KIND))
        cached = self._lookup([key])
        vector = cached.get(key)
        if vector is not None:
            self._validate(key, vector)
            self.stats.hits += 1
            return vector
        self.stats.misses += 1
        started = time.perf_counter()
        query_vector = cast("list[float]", self.inner.embed_query(text))
        self.total_latency_ms += (time.perf_counter() - started) * 1000.0
        self.stats.embedded_texts += 1
        self.stats.embedded_batches += 1
        self._validate(key, query_vector)
        self._store([self._entry(key, QUERY_KIND, query_vector)])
        return query_vector

    # -------------------------------------------------------------- metrics
    def reset_stats(self) -> None:
        self.stats = EmbeddingCacheStats()
        self.total_latency_ms = 0.0

    def embedding_usage(self) -> dict[str, Any]:
        provider_usage: dict[str, Any] = {}
        usage = getattr(self.inner, "embedding_usage", None)
        if callable(usage):
            try:
                provider_usage = dict(usage())
            except Exception as exc:  # provider stats must never break diagnostics
                logger.debug("embedding usage unavailable: %s", type(exc).__name__)
        return {
            **self.stats.to_dict(),
            "provider": self.provider_type,
            "model": self.model_id,
            "dimension": self.dimension,
            "fingerprint": self.embedding_fingerprint,
            "provider_requests": provider_usage.get("requests", self.stats.embedded_batches),
            "provider_latency_ms": round(self.total_latency_ms, 3),
            "prompt_tokens": provider_usage.get("prompt_tokens"),
            "usage_complete": bool(provider_usage.get("usage_complete", True)),
        }

    def health(self) -> dict[str, Any]:
        base: dict[str, Any] = {}
        inner_health = getattr(self.inner, "health", None)
        if callable(inner_health):
            try:
                base = dict(inner_health())
            except Exception as exc:
                base = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        base["cache"] = self.embedding_usage()
        return base

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    # --------------------------------------------------------------- internals
    def _embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        started = time.perf_counter()
        try:
            vectors = self.inner.embed_documents(texts)
        finally:
            self.total_latency_ms += (time.perf_counter() - started) * 1000.0
        self.stats.embedded_texts += len(texts)
        self.stats.embedded_batches += 1
        return list(vectors)

    def _lookup(self, keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], list[float]]:
        try:
            return self.cache.get_many(keys)
        except Exception as exc:  # cache is an optimization
            self._record_cache_error(exc)
            return {}

    def _store(self, entries: Sequence[EmbeddingCacheRow]) -> None:
        if not entries:
            return
        try:
            written = self.cache.put_many(entries)
        except Exception as exc:  # cache is an optimization
            self._record_cache_error(exc)
            return
        self.stats.writes += written

    def _validate(self, key: tuple[str, str], vector: Sequence[float]) -> None:
        if len(vector) != self.dimension:
            raise EmbeddingUnavailable(
                "cached embedding dimension does not match the configured dimension",
                details={
                    "expected": self.dimension,
                    "received": len(vector),
                    "fingerprint": key[0],
                    "text_hash": key[1],
                    "hint": "clear the embedding cache and reindex this generation",
                },
            )

    def _entry(self, key: tuple[str, str], kind: str, vector: Sequence[float]) -> EmbeddingCacheRow:
        now = utc_now()
        return EmbeddingCacheRow(
            embedding_fingerprint=key[0],
            text_hash=key[1],
            kind=kind,
            dimension=len(vector),
            provider=self.provider_type,
            model_id=self.model_id,
            vector_json=[float(value) for value in vector],
            created_at=now,
            last_accessed_at=now,
            hit_count=0,
        )

    def _record_cache_error(self, exc: Exception) -> None:
        self.stats.errors += 1
        if not self._cache_failure_logged:
            self._cache_failure_logged = True
            logger.warning(
                "embedding cache unavailable; continuing without the cache",
                extra={"event": "embedding_cache_error", "error": type(exc).__name__},
            )
