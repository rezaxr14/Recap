"""OpenAI-compatible chat and embedding adapters.

Works with any endpoint that implements the OpenAI wire format: LM Studio,
llama.cpp servers, vLLM, OpenAI, OpenRouter, and compatible hosted services.
The base URL is the API root (normally ending in ``/v1``); RECAP never appends
provider-specific paths outside ``/models``, ``/chat/completions``, and
``/embeddings``.

This module makes no attempt to discover credentials, scan the network, or
download models. Configuration points at an endpoint that is already running.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Sequence
from typing import Any

import httpx

from recap.config import EmbeddingSettings, LlmSettings, redact_url
from recap.errors import (
    ConfigurationError,
    EmbeddingUnavailable,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderUnavailable,
)
from recap.logging import get_logger
from recap.providers.base import (
    ChatMessage,
    ChatResult,
    ProviderCapabilities,
    ToolCall,
    Usage,
)

logger = get_logger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_ERROR_BODY = 400


def _normalize_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if not url:
        raise ConfigurationError("provider base_url is required")
    if url.endswith("/chat/completions") or url.endswith("/embeddings"):
        raise ConfigurationError(
            "provider base_url must be the API root (for example http://localhost:1234/v1), "
            "not a full endpoint path",
            details={"base_url": redact_url(url)},
        )
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ConfigurationError(
            "provider base_url must start with http:// or https://",
            details={"base_url": redact_url(url)},
        )
    # httpx treats a base URL without a trailing slash as a file, so a relative
    # path would replace the final segment (`/v1`). Add the slash explicitly.
    return url + "/"


def _body_snippet(response: httpx.Response) -> str:
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive
        return ""
    return text[:MAX_ERROR_BODY]


class OpenAICompatibleClient:
    """Minimal, explicit OpenAI wire client shared by all roles."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ConfigurationError("provider max_retries must be >= 0")
        if timeout_seconds <= 0:
            raise ConfigurationError("provider timeout_seconds must be > 0")
        self.base_url = _normalize_base_url(base_url)
        self.max_retries = max_retries
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------- transport
    def _request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.request(method, path, json=json_body)
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning(
                    "provider request timed out",
                    extra={
                        "event": "provider_timeout",
                        "path": path,
                        "attempt": attempt + 1,
                        "base_url": redact_url(self.base_url),
                    },
                )
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning(
                    "provider transport error",
                    extra={
                        "event": "provider_transport_error",
                        "path": path,
                        "attempt": attempt + 1,
                        "error": type(exc).__name__,
                    },
                )
            else:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logger.info(
                    "provider request",
                    extra={
                        "event": "provider_request",
                        "path": path,
                        "status": response.status_code,
                        "latency_ms": round(elapsed_ms, 2),
                        "attempt": attempt + 1,
                    },
                )
                if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    self._sleep_before_retry(attempt, response)
                    continue
                self._raise_for_status(response)
                try:
                    return response.json()
                except ValueError as exc:
                    raise ProviderResponseError(
                        "provider returned a non-JSON response",
                        details={"status": response.status_code, "body": _body_snippet(response)},
                    ) from exc
            if attempt < self.max_retries:
                self._sleep_before_retry(attempt, None)
        raise ProviderUnavailable(
            "provider is unreachable or kept failing",
            details={
                "base_url": redact_url(self.base_url),
                "path": path,
                "attempts": self.max_retries + 1,
                "error": type(last_error).__name__ if last_error else "unknown",
            },
        )

    def _sleep_before_retry(self, attempt: int, response: httpx.Response | None) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                with contextlib.suppress(ValueError):
                    delay = min(float(retry_after), 10.0)
        if delay > 0:
            time.sleep(delay)

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        details = {
            "status": status,
            "base_url": redact_url(self.base_url),
            "body": _body_snippet(response),
        }
        if status in {401, 403}:
            raise ProviderAuthenticationError(
                "provider rejected the configured credentials; check the API key",
                details=details,
            )
        if status == 404:
            raise ProviderResponseError(
                "provider endpoint or model was not found; check base_url and model id",
                details=details,
            )
        if status == 429:
            raise ProviderUnavailable(
                "provider rate-limited the request",
                details=details,
            )
        if status >= 500:
            raise ProviderUnavailable("provider returned a server error", details=details)
        raise ProviderResponseError("provider rejected the request", details=details)

    # ------------------------------------------------------------------ API
    def list_models(self) -> list[str]:
        payload = self._request("GET", "models")
        data = payload.get("data") if isinstance(payload, dict) else None
        if data is None and isinstance(payload, dict):
            data = payload.get("models")
        if not isinstance(data, list):
            raise ProviderResponseError(
                "provider model list has an unexpected shape",
                details={"body": str(payload)[:MAX_ERROR_BODY]},
            )
        models: list[str] = []
        for item in data:
            if isinstance(item, dict):
                identifier = item.get("id") or item.get("name")
                if isinstance(identifier, str):
                    models.append(identifier)
            elif isinstance(item, str):
                models.append(item)
        return models

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        result = self._request("POST", "chat/completions", json_body=payload)
        if not isinstance(result, dict):
            raise ProviderResponseError("provider chat response is not an object")
        return result

    def embed(
        self,
        *,
        model: str,
        texts: Sequence[str],
        dimensions: int | None = None,
    ) -> list[list[float]]:
        payload: dict[str, Any] = {"model": model, "input": list(texts)}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        result = self._request("POST", "embeddings", json_body=payload)
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list):
            raise ProviderResponseError("provider embedding response has no data array")
        ordered: list[tuple[int, list[float]]] = []
        for position, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ProviderResponseError("provider returned a malformed embedding item")
            index = item.get("index", position)
            if not isinstance(index, int):
                index = position
            try:
                vector = [float(value) for value in item["embedding"]]
            except (TypeError, ValueError) as exc:
                raise ProviderResponseError("provider embedding is not numeric") from exc
            ordered.append((index, vector))
        ordered.sort(key=lambda pair: pair[0])
        indices = [index for index, _ in ordered]
        if len(indices) != len(texts) or sorted(indices) != list(range(len(texts))):
            raise ProviderResponseError(
                "provider returned invalid or duplicate embedding indices",
                details={"expected": len(texts), "received": len(ordered)},
            )
        return [vector for _, vector in ordered]


class OpenAICompatibleChatProvider:
    """Chat role adapter. Tool calling and JSON output must be supported by the model."""

    provider_type = "openai_compatible"

    def __init__(
        self,
        *,
        settings: LlmSettings,
        role: str = "llm",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not settings.model.strip():
            raise ConfigurationError(f"{role}: model is required for openai_compatible")
        self.role = role
        self.model = settings.model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.structured_mode = settings.structured_mode
        self._client = OpenAICompatibleClient(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
            max_retries=settings.max_retries,
            retry_backoff_seconds=settings.retry_backoff_seconds,
            transport=transport,
        )
        self.capabilities = ProviderCapabilities(
            provider_type=self.provider_type,
            model=self.model,
            chat=True,
            tool_calling=True,
            structured_output=True,
            streaming=False,
            notes=(
                f"role={role}",
                "capabilities are declared by the adapter; model-specific support must be "
                "verified with `recap doctor --live`",
            ),
        )

    def close(self) -> None:
        self._client.close()

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def health(self) -> dict[str, Any]:
        try:
            models = self.list_models()
        except ProviderResponseError as exc:
            if exc.details.get("status") == 404:
                return {
                    "status": "ok",
                    "detail": "model discovery is not supported by this endpoint",
                    "model": self.model,
                }
            raise
        available = self.model in models
        return {
            "status": "ok" if available or not models else "degraded",
            "model": self.model,
            "model_available": available,
            "model_count": len(models),
            "models": sorted(models)[:50],
        }

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        payload = self._client.chat(
            model=self.model,
            messages=[message.to_wire() for message in messages],
            tools=tools,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        message, usage, finish_reason = self._parse_choice(payload)
        tool_calls = self._parse_tool_calls(message)
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ProviderResponseError("provider message content is not text")
        if finish_reason == "length":
            raise ProviderResponseError(
                "provider truncated the response; increase max_tokens or shorten the prompt"
            )
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if self.structured_mode == "json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "recap_structured_output",
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}
        payload = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )
        message, _usage, finish_reason = self._parse_choice(payload)
        if finish_reason == "length":
            raise ProviderResponseError(
                "provider truncated the structured response; increase max_tokens"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("provider returned an empty structured response")
        parsed = _loads_json_object(content)
        return parsed

    @staticmethod
    def _parse_choice(
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], Usage, str]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError("provider response has no choices")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ProviderResponseError("provider response choice has no message")
        raw_usage_value = payload.get("usage")
        raw_usage: dict[str, Any] = raw_usage_value if isinstance(raw_usage_value, dict) else {}
        try:
            usage = Usage(
                prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
                completion_tokens=int(raw_usage.get("completion_tokens") or 0),
                total_tokens=int(raw_usage.get("total_tokens") or 0),
            )
        except (TypeError, ValueError) as exc:
            raise ProviderResponseError("provider usage fields are not numeric") from exc
        finish_reason = str(choice.get("finish_reason") or "")
        return choice["message"], usage, finish_reason

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            return []
        if not isinstance(raw_calls, list):
            raise ProviderResponseError("provider tool_calls is not a list")
        calls: list[ToolCall] = []
        for item in raw_calls:
            if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
                raise ProviderResponseError("provider returned a malformed tool call")
            function = item["function"]
            name = function.get("name")
            arguments = function.get("arguments") or "{}"
            if not isinstance(name, str) or not name:
                raise ProviderResponseError("provider tool call has no function name")
            if isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ProviderResponseError(
                        "provider tool call arguments are not valid JSON"
                    ) from exc
            elif isinstance(arguments, dict):
                decoded = arguments
            else:
                raise ProviderResponseError("provider tool call arguments have an invalid type")
            if not isinstance(decoded, dict):
                raise ProviderResponseError("provider tool call arguments are not an object")
            calls.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=name,
                    arguments=decoded,
                )
            )
        return calls


class OpenAICompatibleEmbeddingProvider:
    """Embedding role adapter with strict dimension enforcement."""

    provider_type = "openai_compatible"

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not settings.model_id.strip():
            raise ConfigurationError("embeddings.model_id is required for openai_compatible")
        if settings.dimension < 1:
            raise ConfigurationError("embeddings.dimension must be >= 1")
        self.model_id = settings.model_id
        self.model_version = settings.model_version
        self.dimension = settings.dimension
        self.batch_size = max(1, settings.batch_size)
        self.send_dimensions = settings.send_dimensions
        self._client = OpenAICompatibleClient(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
            max_retries=settings.max_retries,
            retry_backoff_seconds=settings.retry_backoff_seconds,
            transport=transport,
        )
        self.capabilities = ProviderCapabilities(
            provider_type=self.provider_type,
            model=self.model_id,
            embeddings=True,
            semantic_embeddings=True,
            notes=(f"dimension={self.dimension}",),
        )

    def close(self) -> None:
        self._client.close()

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def health(self) -> dict[str, Any]:
        try:
            models = self.list_models()
        except ProviderResponseError as exc:
            if exc.details.get("status") == 404:
                return {
                    "status": "ok",
                    "detail": "model discovery is not supported by this endpoint",
                    "model": self.model_id,
                }
            raise
        available = self.model_id in models
        return {
            "status": "ok" if available or not models else "degraded",
            "model": self.model_id,
            "model_available": available,
            "model_count": len(models),
        }

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                batch_vectors = self._client.embed(
                    model=self.model_id,
                    texts=batch,
                    dimensions=self.dimension if self.send_dimensions else None,
                )
            except ProviderUnavailable as exc:
                raise EmbeddingUnavailable(
                    f"embedding provider failed: {exc.message}",
                    details={**exc.details, "model": self.model_id},
                ) from exc
            for vector in batch_vectors:
                if len(vector) != self.dimension:
                    # Never truncate or pad: that would silently mix incompatible
                    # vector spaces in one Elasticsearch field.
                    raise EmbeddingUnavailable(
                        "embedding dimension does not match the configured index dimension",
                        details={
                            "expected": self.dimension,
                            "received": len(vector),
                            "model": self.model_id,
                            "hint": "set EMBEDDING_DIMENSIONS and RECAP_VECTOR_DIMENSION to the "
                            "model's real output size, then reindex a new generation",
                        },
                    )
                vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _loads_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    # Some OpenAI-compatible servers wrap JSON in a Markdown fence even when a
    # JSON response format was requested. Accept exactly one fence; anything else
    # fails closed.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(
            "provider structured output is not valid JSON",
            details={"error": str(exc), "preview": text[:300]},
        ) from exc
    if not isinstance(decoded, dict):
        raise ProviderResponseError("provider structured output is not a JSON object")
    return decoded
