"""Embedding provider factory.

RECAP is API-first: the production path is ``openai_compatible`` against an
already-running endpoint. The deterministic hashing and fake providers exist for
local development and tests and are reported as non-semantic. The optional
sentence-transformers adapter only loads an explicitly configured local
directory; it never downloads weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from recap.config import EmbeddingSettings
from recap.errors import ConfigurationError, EmbeddingUnavailable
from recap.infrastructure.embeddings.fake import FakeEmbeddingProvider
from recap.infrastructure.embeddings.hashing import HashingEmbeddingProvider
from recap.infrastructure.embeddings.prefix import PrefixedEmbeddingProvider
from recap.providers.base import ProviderCapabilities
from recap.providers.fingerprint import embedding_fingerprint
from recap.providers.openai_compatible import OpenAICompatibleEmbeddingProvider


class SentenceTransformerEmbeddingProvider:
    """Local embedding adapter for an already-downloaded model directory.

    ``local_files_only=True`` is enforced and the path must exist as a directory.
    There is deliberately no model-ID download path.
    """

    provider_type = "sentence_transformers_local"
    model_version = "1"

    def __init__(self, model_path: str, dimension: int) -> None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ConfigurationError(
                "sentence-transformers requires an existing local model directory; "
                "RECAP does not download model weights",
                details={"model_path": str(path)},
            )
        self.model_path = str(path)
        self.model_id = path.name
        self.dimension = dimension
        self._model: Any = None
        self.capabilities = ProviderCapabilities(
            provider_type=self.provider_type,
            model=self.model_id,
            embeddings=True,
            semantic_embeddings=True,
            notes=("local model directory only; remote downloads are disabled",),
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "mode": "local_model",
            "semantic": True,
            "model": self.model_id,
            "model_path": self.model_path,
            "dimension": self.dimension,
        }

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise EmbeddingUnavailable(
                    "sentence-transformers is not installed; install the "
                    "'sentence-transformers' extra or configure an API embedding provider"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_path, local_files_only=True)
            except TypeError:  # pragma: no cover - older library versions
                raise EmbeddingUnavailable(
                    "the installed sentence-transformers version cannot enforce local-only "
                    "loading; upgrade it or configure an API embedding provider"
                ) from None
            actual = int(self._model.get_sentence_embedding_dimension())
            if actual != self.dimension:
                raise ConfigurationError(
                    "configured embedding dimension does not match the local model",
                    details={"configured": self.dimension, "model": actual},
                )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedding_provider(
    settings: EmbeddingSettings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> Any:
    if settings.provider == "hashing":
        provider: Any = HashingEmbeddingProvider(dimension=settings.dimension)
    elif settings.provider == "fake":
        provider = FakeEmbeddingProvider(dimension=settings.dimension)
    elif settings.provider == "openai_compatible":
        provider = OpenAICompatibleEmbeddingProvider(settings, transport=transport)
    elif settings.provider == "sentence-transformers":
        provider = SentenceTransformerEmbeddingProvider(settings.model_path, settings.dimension)
    else:
        raise ConfigurationError(f"unknown embedding provider: {settings.provider}")
    if settings.document_prefix or settings.query_prefix:
        provider = PrefixedEmbeddingProvider(
            provider,
            document_prefix=settings.document_prefix,
            query_prefix=settings.query_prefix,
        )
    provider.embedding_fingerprint = embedding_fingerprint(
        provider=settings.provider,
        model_id=getattr(provider, "model_id", settings.model_id),
        model_version=getattr(provider, "model_version", settings.model_version),
        dimension=settings.dimension,
        normalize=settings.normalize,
        document_prefix=settings.document_prefix,
        query_prefix=settings.query_prefix,
    )
    return provider
