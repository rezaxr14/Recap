"""Ports (interfaces) for external dependencies.

Keeping these small lets core policy logic be tested without network access and
lets Elasticsearch/embedding/provider changes stay at the adapter boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from recap.domain.environments import EnvironmentFingerprint
from recap.domain.memories import ExperienceMemory
from recap.domain.repairs import RepairCandidate, ValidationResult
from recap.domain.retrieval import RetrievalContext


class EmbeddingProvider(Protocol):
    model_id: str
    model_version: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    """Second-pass query-document scorer.

    Implementations only reorder candidates. Safety eligibility is evaluated
    afterwards by the recall policy, never by the reranker.
    """

    name: str

    def rerank(
        self, query: str, documents: Sequence[str], top_k: int
    ) -> list[tuple[int, float]]: ...


# Backwards-compatible name used by existing adapter type hints.
Reranker = RerankerProvider


class LLMProvider(Protocol):
    model_id: str
    model_version: str

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]: ...


class RepairValidator(Protocol):
    name: str
    version: str

    def validate(
        self,
        candidate: RepairCandidate,
        *,
        environment: EnvironmentFingerprint,
        attempt: int = 1,
    ) -> ValidationResult: ...


class MemorySearchRepository(Protocol):
    def ensure_ready(self) -> None: ...

    def index_memory(self, memory: ExperienceMemory, embedding: Sequence[float]) -> None: ...

    def delete_memory(self, memory_id: str, memory_version: int) -> None: ...

    def search(
        self,
        *,
        context: RetrievalContext,
        mode: str,
        candidate_k: int,
        index_generation: int | None = None,
    ) -> dict[str, Any]: ...


class ObjectStorage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...
