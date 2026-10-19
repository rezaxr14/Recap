"""Provider adapters: wire format, fail-closed behaviour, and no secrets.

These use ``httpx.MockTransport`` so they test the real request/response code
without network access or credentials. Live provider integration is covered by
an opt-in test file and is never required for CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from recap.config import EmbeddingSettings, LlmSettings, Settings, load_settings
from recap.errors import (
    ConfigurationError,
    EmbeddingUnavailable,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderUnavailable,
)
from recap.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleClient,
    OpenAICompatibleEmbeddingProvider,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _chat_settings(**overrides: object) -> LlmSettings:
    values: dict[str, object] = {
        "provider": "openai_compatible",
        "base_url": "http://provider.test/v1",
        "model": "chat-model",
        "max_retries": 0,
        "retry_backoff_seconds": 0.0,
    }
    values.update(overrides)
    return LlmSettings.model_validate(values)


def _embedding_settings(**overrides: object) -> EmbeddingSettings:
    values: dict[str, object] = {
        "provider": "openai_compatible",
        "base_url": "http://provider.test/v1",
        "model_id": "embed-model",
        "model_version": "1",
        "dimension": 4,
        "max_retries": 0,
        "retry_backoff_seconds": 0.0,
    }
    values.update(overrides)
    return EmbeddingSettings.model_validate(values)


def _provider(handler: Handler, **settings: object) -> OpenAICompatibleChatProvider:
    return OpenAICompatibleChatProvider(
        settings=_chat_settings(**settings), transport=httpx.MockTransport(handler)
    )


# ---------------------------------------------------------------- discovery
def test_list_models_and_health_report_the_configured_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "chat-model"}, {"id": "other"}]})

    provider = _provider(handler)

    assert provider.list_models() == ["chat-model", "other"]
    health = provider.health()
    assert health["status"] == "ok"
    assert health["model_available"] is True


def test_discovery_unsupported_is_reported_not_faked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    provider = _provider(handler)
    health = provider.health()

    assert health["status"] == "ok"
    assert "not supported" in health["detail"]


def test_configured_model_missing_from_discovery_is_degraded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "something-else"}]})

    health = _provider(handler).health()
    assert health["status"] == "degraded"
    assert health["model_available"] is False


# --------------------------------------------------------------------- chat
def test_chat_parses_content_tool_calls_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "chat-model"
        assert payload["messages"][0]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "calling the tool",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "list_records",
                                        "arguments": '{"page_size": 10}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )

    provider = _provider(handler)
    from recap.providers.base import ChatMessage

    result = provider.chat([ChatMessage(role="user", content="go")])

    assert result.content == "calling the tool"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "list_records"
    assert result.tool_calls[0].arguments == {"page_size": 10}
    assert result.usage.total_tokens == 7
    assert result.latency_ms >= 0


def test_invalid_tool_call_arguments_fail_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "list_records", "arguments": "{oops"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    from recap.providers.base import ChatMessage

    with pytest.raises(ProviderResponseError):
        _provider(handler).chat([ChatMessage(role="user", content="go")])


def test_truncated_chat_response_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        )

    from recap.providers.base import ChatMessage

    with pytest.raises(ProviderResponseError, match="truncated"):
        _provider(handler).chat([ChatMessage(role="user", content="go")])


# --------------------------------------------------------- structured output
def test_structured_output_accepts_one_markdown_fence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '```json\n{"summary": "ok"}\n```'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    provider = _provider(handler)
    result = provider.generate_structured(
        system_prompt="system", user_prompt="user", schema={"type": "object"}
    )
    assert result == {"summary": "ok"}


def test_structured_output_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]},
        )

    with pytest.raises(ProviderResponseError, match="not valid JSON"):
        _provider(handler).generate_structured(
            system_prompt="system", user_prompt="user", schema={"type": "object"}
        )


def test_json_object_mode_is_explicit() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]},
        )

    provider = _provider(handler, structured_mode="json_object")
    assert provider.generate_structured(
        system_prompt="s", user_prompt="u", schema={"type": "object"}
    ) == {"ok": True}
    assert seen["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------- embeddings
def test_embeddings_are_sorted_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "embed-model"
        assert payload["input"] == ["a", "b"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 0.0, 0.0, 2.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                ]
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        _embedding_settings(), transport=httpx.MockTransport(handler)
    )
    vectors = provider.embed_documents(["a", "b"])
    assert vectors == [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]


def test_duplicate_embedding_indices_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                    {"index": 0, "embedding": [0.0, 1.0, 0.0, 0.0]},
                ]
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        _embedding_settings(), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderResponseError, match="indices"):
        provider.embed_documents(["a", "b"])


def test_embedding_dimension_mismatch_is_rejected_without_truncation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        )

    provider = OpenAICompatibleEmbeddingProvider(
        _embedding_settings(dimension=4), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EmbeddingUnavailable) as error:
        provider.embed_documents(["a"])
    assert error.value.details["expected"] == 4
    assert error.value.details["received"] == 2


def test_embeddings_are_batched_without_reordering() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["input"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(index), 0.0, 0.0, 0.0]}
                    for index in range(len(payload["input"]))
                ]
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        _embedding_settings(batch_size=2), transport=httpx.MockTransport(handler)
    )
    vectors = provider.embed_documents(["a", "b", "c"])
    assert calls == [["a", "b"], ["c"]]
    assert len(vectors) == 3


# --------------------------------------------------------------------- errors
def test_transient_error_is_retried_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"data": [{"id": "chat-model"}]})

    client = OpenAICompatibleClient(
        base_url="http://provider.test/v1", max_retries=1, retry_backoff_seconds=0.0
    )
    client._client = httpx.Client(
        base_url="http://provider.test/v1/",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.list_models() == ["chat-model"]
    finally:
        client.close()
    assert calls["count"] == 2


def test_authentication_error_is_distinct_and_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    with pytest.raises(ProviderAuthenticationError) as error:
        _provider(handler).list_models()
    assert error.value.details["status"] == 401
    assert "API key" in error.value.message


def test_exhausted_retries_raise_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    client = OpenAICompatibleClient(
        base_url="http://provider.test/v1", max_retries=1, retry_backoff_seconds=0.0
    )
    client._client = httpx.Client(
        base_url="http://provider.test/v1/", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ProviderUnavailable):
            client.list_models()
    finally:
        client.close()


def test_error_details_redact_credentials_in_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    provider = OpenAICompatibleChatProvider(
        settings=_chat_settings(base_url="http://user:pass@provider.test/v1"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderUnavailable) as error:
        provider.list_models()
    serialized = json.dumps(error.value.details)
    assert "user:pass" not in serialized
    assert "***:***@provider.test" in serialized


def test_full_endpoint_url_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="API root"):
        OpenAICompatibleClient(base_url="http://provider.test/v1/chat/completions")


def test_transport_error_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = OpenAICompatibleClient(
        base_url="http://provider.test/v1", max_retries=0, retry_backoff_seconds=0.0
    )
    client._client = httpx.Client(
        base_url="http://provider.test/v1/", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ProviderUnavailable) as error:
            client.list_models()
    finally:
        client.close()
    assert error.value.details["attempts"] == 1


# ------------------------------------------------------------------- config
def test_llm_generator_requires_a_repair_provider() -> None:
    settings = Settings.model_validate({"repair_compiler": {"generator": "llm"}})
    with pytest.raises(ConfigurationError, match="repair_llm"):
        settings.validate_consistency()


def test_openai_embedding_requires_base_url_and_model() -> None:
    with pytest.raises(ConfigurationError, match="base_url"):
        Settings.model_validate(
            {"embeddings": {"provider": "openai_compatible", "model_id": "e"}}
        ).validate_consistency()
    with pytest.raises(ConfigurationError, match="model_id"):
        Settings.model_validate(
            {
                "embeddings": {
                    "provider": "openai_compatible",
                    "base_url": "http://x/v1",
                }
            }
        ).validate_consistency()


def test_production_rejects_non_semantic_embeddings_unless_acknowledged() -> None:
    production = Settings.model_validate(
        {"app": {"environment": "production"}, "embeddings": {"provider": "hashing"}}
    )
    with pytest.raises(ConfigurationError, match="real embedding provider"):
        production.validate_consistency()
    acknowledged = Settings.model_validate(
        {
            "app": {"environment": "production"},
            "embeddings": {"provider": "hashing", "allow_degraded": True},
        }
    )
    acknowledged.validate_consistency()  # explicit degraded mode is allowed


def test_environment_overrides_configure_roles_and_dimension(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-model")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "chat-model")
    monkeypatch.setenv("REPAIR_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("REPAIR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("REPAIR_LLM_MODEL", "repair-model")
    monkeypatch.setenv("REPAIR_GENERATOR", "llm")

    settings = load_settings(config)

    assert settings.embeddings.provider == "openai_compatible"
    assert settings.embeddings.dimension == 768
    assert settings.elasticsearch.vector_dimension == 768
    assert settings.llm.model == "chat-model"
    assert settings.repair_llm.model == "repair-model"
    assert settings.repair_compiler.generator == "llm"


def test_sentence_transformers_requires_an_existing_local_directory(tmp_path) -> None:
    from recap.infrastructure.embeddings import build_embedding_provider

    settings = EmbeddingSettings.model_validate(
        {
            "provider": "sentence-transformers",
            "model_id": "local-model",
            "dimension": 4,
            "model_path": str(tmp_path / "missing-model"),
        }
    )
    with pytest.raises(ConfigurationError, match="local model directory"):
        build_embedding_provider(settings)


def test_non_semantic_providers_report_their_capabilities() -> None:
    from recap.infrastructure.embeddings import build_embedding_provider

    for provider_name in ("hashing", "fake"):
        provider = build_embedding_provider(EmbeddingSettings(provider=provider_name, dimension=32))
        assert provider.capabilities.semantic_embeddings is False
        health = provider.health()
        assert health["semantic"] is False
        if provider_name == "hashing":
            assert "semantic RAG is disabled" in str(health["detail"])


def test_safe_dump_never_contains_provider_keys() -> None:
    settings = Settings.model_validate(
        {
            "embeddings": {
                "provider": "openai_compatible",
                "base_url": "http://user:pass@x/v1",
                "model_id": "e",
                "api_key": "embed-secret",
            },
            "llm": {
                "provider": "openai_compatible",
                "base_url": "http://x/v1",
                "model": "m",
                "api_key": "llm-secret",
            },
            "repair_llm": {
                "provider": "openai_compatible",
                "base_url": "http://x/v1",
                "model": "m",
                "api_key": "repair-secret",
            },
        }
    )
    dumped = json.dumps(settings.safe_dump())
    assert "embed-secret" not in dumped
    assert "llm-secret" not in dumped
    assert "repair-secret" not in dumped
    assert "user:pass" not in dumped
    assert "***:***@x" in dumped
