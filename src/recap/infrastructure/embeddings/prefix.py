"""Asymmetric task-prefix decorator for embedding providers.

Some models (for example Nomic) require different prefixes for documents and
queries. Prefixes are configuration, never persisted into chunk text, and are
part of the embedding fingerprint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast


class PrefixedEmbeddingProvider:
    """Delegates to an inner provider and applies document/query prefixes."""

    def __init__(self, inner: Any, *, document_prefix: str, query_prefix: str) -> None:
        self._inner = inner
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [f"{self.document_prefix}{text}" for text in texts]
        return cast("list[list[float]]", self._inner.embed_documents(prefixed))

    def embed_query(self, text: str) -> list[float]:
        return cast("list[float]", self._inner.embed_query(f"{self.query_prefix}{text}"))
