"""Deterministic fake embedding provider for tests.

Produces stable, clearly synthetic vectors keyed by exact text. Named ``fake`` so
it can never be mistaken for a semantic model. Use ``HashingEmbeddingProvider``
for a deterministic embedder with real lexical signal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from recap.providers.base import ProviderCapabilities


class FakeEmbeddingProvider:
    model_id = "fake-test-embedding"
    model_version = "1"
    provider_type = "fake"

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.capabilities = ProviderCapabilities(
            provider_type=self.provider_type,
            model=self.model_id,
            embeddings=True,
            semantic_embeddings=False,
            notes=("test-only synthetic vectors; never evidence of semantic retrieval",),
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "mode": "test_only",
            "semantic": False,
            "model": self.model_id,
        }

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for index in range(self.dimension):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) * 2.0 - 1.0)
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
