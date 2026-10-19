"""Opt-in live tests against a real, already-running model endpoint.

These tests are intentionally separate from the deterministic suite:

* They run only when ``RECAP_RUN_PROVIDER_INTEGRATION=1`` is set.
* They skip (never fake) when the corresponding role is not configured.
* They make real chat/embedding calls and assert on real dimensions and real
  Elasticsearch retrieval.

A fake provider must never satisfy these tests. CI does not set the flag and the
tests skip by default.
"""

from __future__ import annotations

import os
import uuid

import pytest
from tests.support import context_for, make_memory

from recap.config import EmbeddingSettings, LlmSettings, load_settings
from recap.container import Container
from recap.domain.enums import RetrievalMode
from recap.domain.retrieval import RetrievalRequest
from recap.infrastructure.embeddings import build_embedding_provider
from recap.providers.factory import build_chat_provider

pytestmark = pytest.mark.integration

if os.environ.get("RECAP_RUN_PROVIDER_INTEGRATION") != "1":
    pytest.skip(
        "live provider tests are opt-in; set RECAP_RUN_PROVIDER_INTEGRATION=1",
        allow_module_level=True,
    )


def _chat_provider() -> tuple[LlmSettings, object]:
    settings = load_settings()
    if settings.llm.provider == "disabled":
        pytest.skip("LLM_PROVIDER is not configured")
    provider = build_chat_provider("llm", settings.llm)
    if provider is None:
        pytest.skip("LLM_PROVIDER is disabled")
    return settings.llm, provider


def _embedding_provider() -> tuple[EmbeddingSettings, object]:
    settings = load_settings()
    if settings.embeddings.provider != "openai_compatible":
        pytest.skip(
            "live embedding assertions require EMBEDDING_PROVIDER=openai_compatible "
            "with a real model"
        )
    return settings.embeddings, build_embedding_provider(settings.embeddings)


def test_configured_chat_endpoint_is_reachable() -> None:
    _, provider = _chat_provider()
    health = provider.health()  # type: ignore[attr-defined]
    assert health["status"] in {"ok", "degraded"}, health
    assert health.get("model_available") is not False, health


def test_small_chat_completion() -> None:
    from recap.providers.base import ChatMessage

    _, provider = _chat_provider()
    result = provider.chat(  # type: ignore[attr-defined]
        [ChatMessage(role="user", content="Reply with exactly the word: OK")],
        max_tokens=16,
    )
    assert result.content.strip()
    assert result.finish_reason


def test_structured_json_output() -> None:
    _, provider = _chat_provider()
    payload = provider.generate_structured(  # type: ignore[attr-defined]
        system_prompt="Return a single JSON object that matches the schema.",
        user_prompt="What is 2 + 2? Put the numeric answer in the 'answer' field.",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    assert isinstance(payload, dict)
    assert payload.get("answer") == 4


def test_real_embedding_matches_configured_dimension() -> None:
    settings, provider = _embedding_provider()
    vector = provider.embed_query("recap live embedding probe")  # type: ignore[attr-defined]
    assert len(vector) == settings.dimension
    assert any(component != 0.0 for component in vector)


def test_real_embeddings_index_and_retrieve_from_elasticsearch(live_settings) -> None:  # type: ignore[no-untyped-def]
    embedding_settings, _ = _embedding_provider()
    token = uuid.uuid4().hex[:10]
    settings = live_settings.model_copy(deep=True)
    settings.elasticsearch.index_prefix = f"recap-live-{token}"
    settings.elasticsearch.index_generation = 1
    settings.elasticsearch.memories_read_alias = f"recap-live-{token}-read"
    settings.elasticsearch.memories_write_alias = f"recap-live-{token}-write"
    settings.elasticsearch.vector_dimension = embedding_settings.dimension
    settings.embeddings = embedding_settings

    container = Container.build(settings)
    try:
        container.migrate()
        container.init_search()
        assert container.provider_status()["semantic_retrieval"] is True

        workspace_id = f"live-provider-{token}"
        memory = make_memory(
            container,
            workspace_id=workspace_id,
            title="Recover from an invalid pagination cursor",
            summary="Discard the expired pagination cursor and start again from page one.",
            failure_signature="ValidationError|INVALID_CURSOR|list_records|cursor is invalid",
            failure_code="INVALID_CURSOR",
            environment_version="1",
        )
        container.es_client.indices.refresh(index=container.index_manager.write_index)

        with container.services() as services:
            decision = services.recall.retrieve(
                RetrievalRequest(
                    context=context_for(workspace_id=workspace_id, environment_version="1"),
                    mode=RetrievalMode.VECTOR,
                    final_k=3,
                )
            )
        assert decision.status.value == "selected", decision.reason
        assert memory.id in [item.memory_id for item in decision.selected_memories]

        indexed = container.es_client.get(
            index=container.index_manager.write_index, id=f"{memory.id}:v1"
        )
        assert len(indexed["_source"]["content_embedding"]) == embedding_settings.dimension
    finally:
        try:
            container.es_client.indices.delete(
                index=container.index_manager.write_index, ignore_unavailable=True
            )
        finally:
            container.es_client.close()
