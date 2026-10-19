"""Model-provider ports.

Business code depends on these protocols, never on a specific HTTP API. The
capabilities object is part of the contract: a provider must say what it can
actually do instead of letting callers assume that "LLM" implies embeddings or
tool calling.

Nothing in this package downloads model weights. Adapters either talk to an
already-running HTTP endpoint or, for the optional local sentence-transformers
adapter, load an explicitly configured local directory with downloads disabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declared, verifiable capabilities of one configured provider."""

    provider_type: str
    model: str
    chat: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    embeddings: bool = False
    reranking: bool = False
    streaming: bool = False
    semantic_embeddings: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str = ""


class ChatProvider(Protocol):
    """Chat/completions with optional tool calling and structured output."""

    provider_type: str
    model: str
    capabilities: ProviderCapabilities

    def list_models(self) -> list[str]: ...

    def health(self) -> dict[str, Any]: ...

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult: ...

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]: ...


class EmbeddingProvider(Protocol):
    """Document/query embeddings with a fixed, declared dimension."""

    provider_type: str
    model_id: str
    model_version: str
    dimension: int
    capabilities: ProviderCapabilities

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
