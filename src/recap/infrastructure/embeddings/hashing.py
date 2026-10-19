"""Deterministic hashing embedding provider.

This is a *real* deterministic feature-hashing embedder, not a stub: it tokenizes
text, hashes tokens into a fixed-dimension vector with sublinear term weighting,
and L2-normalizes. It is suitable for local demo/tests and for validating the
vector search path without downloading a model. It is documented as a lexical
hash embedder, not a semantic neural model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

from recap.providers.base import ProviderCapabilities

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "as",
    "at",
    "by",
    "from",
    "after",
    "before",
    "when",
    "then",
    "than",
    "but",
    "not",
    "no",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        tokens.append(raw)
        # split snake_case / hyphenated identifiers into parts as well
        if "_" in raw:
            tokens.extend(part for part in raw.split("_") if part and part not in STOPWORDS)
    return tokens


class HashingEmbeddingProvider:
    model_id = "recap-hashing-embedding-v1"
    model_version = "1"
    provider_type = "hashing"

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 16:
            raise ValueError("embedding dimension must be >= 16")
        self.dimension = dimension
        self.capabilities = ProviderCapabilities(
            provider_type=self.provider_type,
            model=self.model_id,
            embeddings=True,
            semantic_embeddings=False,
            notes=("deterministic lexical feature hashing; not a neural semantic model",),
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "mode": "local_deterministic",
            "semantic": False,
            "model": self.model_id,
            "dimension": self.dimension,
            "detail": "lexical hashing embedder; semantic RAG is disabled",
        }

    def _embed(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * self.dimension
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # deterministic non-zero fallback so cosine similarity is defined
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
