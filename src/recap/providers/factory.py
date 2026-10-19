"""Provider construction and redacted capability reporting."""

from __future__ import annotations

from typing import Any

import httpx

from recap.config import LlmSettings, redact_url
from recap.errors import ConfigurationError
from recap.providers.base import ChatProvider, EmbeddingProvider
from recap.providers.openai_compatible import OpenAICompatibleChatProvider


def build_chat_provider(
    role: str,
    settings: LlmSettings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ChatProvider | None:
    """Return a configured chat provider, or ``None`` when the role is disabled.

    A selected provider that cannot be constructed raises. There is no silent
    fallback to a fake model: a misconfigured real provider must fail visibly.
    """
    if settings.provider == "disabled":
        return None
    if settings.provider == "openai_compatible":
        return OpenAICompatibleChatProvider(settings=settings, role=role, transport=transport)
    raise ConfigurationError(f"{role}: unsupported provider type {settings.provider!r}")


def chat_role_report(
    role: str, settings: LlmSettings, provider: ChatProvider | None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": role,
        "configured": provider is not None,
        "provider": settings.provider,
        "base_url": redact_url(settings.base_url) if settings.base_url else "",
        "model": settings.model,
        "api_key_configured": bool(settings.api_key),
        "structured_mode": settings.structured_mode,
    }
    if provider is not None:
        entry["capabilities"] = provider.capabilities.to_dict()
    return entry


def embedding_role_report(settings: Any, provider: EmbeddingProvider | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": "embeddings",
        "configured": provider is not None,
        "provider": settings.provider,
        "base_url": redact_url(settings.base_url) if getattr(settings, "base_url", "") else "",
        "model": settings.model_id,
        "model_version": settings.model_version,
        "dimension": settings.dimension,
        "api_key_configured": bool(getattr(settings, "api_key", "")),
        "degraded": bool(provider is not None and not provider.capabilities.semantic_embeddings),
    }
    if provider is not None:
        entry["capabilities"] = provider.capabilities.to_dict()
    return entry


def provider_report(
    *,
    llm_settings: LlmSettings,
    repair_llm_settings: LlmSettings,
    embedding_settings: Any,
    llm: ChatProvider | None,
    repair_llm: ChatProvider | None,
    embeddings: EmbeddingProvider | None,
) -> dict[str, Any]:
    semantic = bool(embeddings is not None and embeddings.capabilities.semantic_embeddings)
    report: dict[str, Any] = {
        "semantic_retrieval": semantic,
        "degraded_mode": not semantic,
        "roles": {
            "agent_llm": chat_role_report("agent", llm_settings, llm),
            "repair_llm": chat_role_report("repair", repair_llm_settings, repair_llm),
            "embeddings": embedding_role_report(embedding_settings, embeddings),
        },
    }
    if not semantic:
        report["degraded_reason"] = (
            "the configured embedding provider is lexical/test-only; retrieval is not semantic. "
            "Configure EMBEDDING_PROVIDER=openai_compatible with a real embedding model."
        )
    return report
