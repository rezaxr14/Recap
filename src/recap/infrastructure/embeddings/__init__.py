"""Embedding provider factory.

The default is the deterministic hashing provider so the whole pipeline runs
without network access or paid keys. A real sentence-transformers adapter is
available when the optional dependency is installed.
"""

from __future__ import annotations

from typing import Any

from recap.config import EmbeddingSettings
from recap.errors import ConfigurationError, EmbeddingUnavailable
from recap.infrastructure.embeddings.fake import FakeEmbeddingProvider
from recap.infrastructure.embeddings.hashing import HashingEmbeddingProvider


class SentenceTransformerEmbeddingProvider:
    """Real local embedding adapter (optional dependency).

    The model is loaded lazily so importing RECAP does not require torch.
    """

    model_version = "1"

    def __init__(self, model_id: str, dimension: int) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise EmbeddingUnavailable(
                    "sentence-transformers is not installed; install the "
                    "'sentence-transformers' extra or use provider=hashing"
                ) from exc
            self._model = SentenceTransformer(self.model_id)
            actual = int(self._model.get_sentence_embedding_dimension())
            if actual != self.dimension:
                raise ConfigurationError(
                    "configured embedding dimension does not match model",
                    details={"configured": self.dimension, "model": actual},
                )
        return self._model

    def embed_documents(self, texts: Any) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedding_provider(settings: EmbeddingSettings):
    if settings.provider == "hashing":
        return HashingEmbeddingProvider(dimension=settings.dimension)
    if settings.provider == "fake":
        return FakeEmbeddingProvider(dimension=settings.dimension)
    if settings.provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(settings.model_id, settings.dimension)
    raise ConfigurationError(f"unknown embedding provider: {settings.provider}")
