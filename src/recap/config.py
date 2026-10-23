"""Configuration loading.

One validated configuration model with environment overrides. Secrets are never
included in ``safe_dump`` output.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from recap.errors import ConfigurationError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class _SettingsModel(BaseModel):
    """Reject unknown keys so a misspelled setting cannot silently no-op."""

    model_config = {"extra": "forbid"}


class AppSettings(_SettingsModel):
    name: str = "recap"
    environment: str = "local"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    # Empty disables auth, which is the local/replay default. Deployments must
    # set RECAP_API_KEY and keep the CORS origin list narrow.
    api_key: str = ""
    cors_origins: str = "http://localhost:8000,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


class StorageSettings(_SettingsModel):
    postgres_url: str = "postgresql+psycopg://recap:recap@localhost:5433/recap"
    max_event_payload_bytes: int = 65_536


class ElasticsearchSettings(_SettingsModel):
    url: str = "http://localhost:9200"
    api_key: str = ""
    index_prefix: str = "recap"
    memories_read_alias: str = "recap-memories-read"
    memories_write_alias: str = "recap-memories-write"
    chunks_read_alias: str = "recap-chunks-read"
    chunks_write_alias: str = "recap-chunks-write"
    index_generation: int = 1
    chunks_index_generation: int = 2
    request_timeout_seconds: float = 5.0
    vector_dimension: int = 256
    hybrid_strategy: str = "app_rrf"
    rrf_k: int = 60
    # Identifier-aware subfields (``INVALID_CURSOR`` <-> "invalid cursor").
    # Opt-in because an existing index must be rebuilt into a new generation
    # before the subfields exist; enabling it against an old mapping is a
    # configuration error, not a silent query failure.
    identifier_analysis: bool = False
    knn_num_candidates: int = 200
    dense_vector_quantization: str = "none"

    @field_validator("hybrid_strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in {"app_rrf", "es_rrf"}:
            raise ValueError("hybrid_strategy must be 'app_rrf' or 'es_rrf'")
        return value

    @field_validator("dense_vector_quantization")
    @classmethod
    def _known_quantization(cls, value: str) -> str:
        if value not in {"none", "int8_hnsw"}:
            raise ValueError("dense_vector_quantization must be 'none' or 'int8_hnsw'")
        return value

    @field_validator("knn_num_candidates")
    @classmethod
    def _known_num_candidates(cls, value: int) -> int:
        if value < 1:
            raise ValueError("knn_num_candidates must be >= 1")
        return value

    @property
    def dense_vector_index_type(self) -> str:
        return "hnsw" if self.dense_vector_quantization == "none" else "int8_hnsw"

    @property
    def write_index(self) -> str:
        return f"{self.index_prefix}-memories-v{self.index_generation}"

    @property
    def chunks_write_index(self) -> str:
        return f"{self.index_prefix}-chunks-v{self.chunks_index_generation}"


class EmbeddingSettings(_SettingsModel):
    """Embedding role. Never downloads weights; ``openai_compatible`` calls an API."""

    provider: str = "hashing"
    model_id: str = "recap-hashing-embedding-v1"
    model_version: str = "1"
    dimension: int = 256
    batch_size: int = 32
    base_url: str = ""
    api_key: str = ""
    # ``timeout_seconds`` is the legacy single-knob fallback. The per-phase
    # timeouts below make a slow local model distinguishable from a dead socket:
    # connecting, uploading a prompt, and waiting for the first/last token have
    # very different healthy durations.
    timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 0.0  # 0 -> use timeout_seconds
    write_timeout_seconds: float = 30.0
    pool_timeout_seconds: float = 30.0
    overall_timeout_seconds: float = 300.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    send_dimensions: bool = False
    model_path: str = ""
    document_prefix: str = ""
    query_prefix: str = ""
    normalize: bool = True
    # The Postgres embedding cache is an optimization only. It is safe to leave
    # on: keys include the exact fingerprint and provider-input hash, so a hit
    # can never cross models, dimensions, prefixes, or document/query roles.
    cache_enabled: bool = True
    # A lexical hashing/fake embedder is allowed in local development and
    # tests. Production and any deployment that claims semantic RAG must set a
    # real provider, or explicitly acknowledge the degraded mode here.
    allow_degraded: bool = False

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        if value not in {
            "hashing",
            "fake",
            "openai_compatible",
            "sentence-transformers",
        }:
            raise ValueError(
                "provider must be hashing|fake|openai_compatible|sentence-transformers"
            )
        return value

    @property
    def effective_read_timeout(self) -> float:
        return self.read_timeout_seconds or self.timeout_seconds

    @model_validator(mode="after")
    def _timeouts(self) -> EmbeddingSettings:
        values = {
            "timeout_seconds": self.timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "write_timeout_seconds": self.write_timeout_seconds,
            "pool_timeout_seconds": self.pool_timeout_seconds,
            "overall_timeout_seconds": self.overall_timeout_seconds,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"embeddings.{name} must be > 0")
        if self.read_timeout_seconds < 0:
            raise ValueError("embeddings.read_timeout_seconds must be >= 0")
        return self


class LlmSettings(_SettingsModel):
    """One chat/completions role (agent or repair compiler)."""

    provider: str = "disabled"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # Legacy fallback used only when ``read_timeout_seconds`` is left at zero.
    timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 0.0
    write_timeout_seconds: float = 30.0
    pool_timeout_seconds: float = 30.0
    # Hard wall-clock cap for one model call, independent of per-read progress.
    overall_timeout_seconds: float = 900.0
    # A stream that produces no bytes for this long is treated as stalled.
    no_progress_timeout_seconds: float = 180.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    temperature: float = 0.0
    # Bounded output is required: an unbounded local model can otherwise consume
    # the entire context window or hang while generating a runaway response.
    max_tokens: int = 512
    structured_mode: str = "json_schema"
    stream: bool = True
    # Local runtimes usually serve one generation at a time. Serializing calls
    # avoids queueing timeouts that look like model failures.
    max_concurrency: int = 1

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        if value not in {"disabled", "openai_compatible"}:
            raise ValueError("provider must be disabled|openai_compatible")
        return value

    @field_validator("structured_mode")
    @classmethod
    def _known_structured_mode(cls, value: str) -> str:
        if value not in {"json_schema", "json_object"}:
            raise ValueError("structured_mode must be json_schema|json_object")
        return value

    @property
    def effective_read_timeout(self) -> float:
        return self.read_timeout_seconds or self.timeout_seconds

    @model_validator(mode="after")
    def _inference_limits(self) -> LlmSettings:
        values = {
            "timeout_seconds": self.timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "write_timeout_seconds": self.write_timeout_seconds,
            "pool_timeout_seconds": self.pool_timeout_seconds,
            "overall_timeout_seconds": self.overall_timeout_seconds,
            "no_progress_timeout_seconds": self.no_progress_timeout_seconds,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"llm.{name} must be > 0")
        if self.read_timeout_seconds < 0:
            raise ValueError("llm.read_timeout_seconds must be >= 0")
        if self.max_tokens < 1:
            raise ValueError("llm.max_tokens must be >= 1 (bounded output is required)")
        if not 1 <= self.max_concurrency <= 32:
            raise ValueError("llm.max_concurrency must be within [1, 32]")
        return self


class RecallSettings(_SettingsModel):
    default_mode: str = "outcome_aware"
    lexical_candidate_k: int = 30
    vector_candidate_k: int = 30
    final_k: int = 5
    max_candidates: int = 100
    allow_unvalidated_operational_memories: bool = False
    allow_suspect_operational_memories: bool = False
    require_tool_match: bool = False
    no_safe_memory_threshold: float = 0.35
    max_per_strategy: int = 2
    reranker: str = "lexical"
    rerank_model_path: str = ""
    rerank_max_length: int = 512
    rerank_batch_size: int = 16

    @field_validator("reranker")
    @classmethod
    def _known_reranker(cls, value: str) -> str:
        if value not in {"none", "lexical", "cross_encoder"}:
            raise ValueError("reranker must be 'none', 'lexical', or 'cross_encoder'")
        return value

    @field_validator("rerank_max_length")
    @classmethod
    def _known_rerank_max_length(cls, value: int) -> int:
        if value < 16 or value > 8192:
            raise ValueError("rerank_max_length must be within [16, 8192]")
        return value

    @field_validator("rerank_batch_size")
    @classmethod
    def _known_rerank_batch_size(cls, value: int) -> int:
        if value < 1 or value > 256:
            raise ValueError("rerank_batch_size must be within [1, 256]")
        return value


CHUNK_TYPES = (
    "memory_card",
    "failure_window",
    "action_observation",
    "validation_evidence",
)


class ChunkingSettings(_SettingsModel):
    """Deterministic, event-aware chunking. No tokenizer or model downloads."""

    strategy: str = "event_aware"
    version: str = "2"
    target_tokens: int = 256
    max_tokens: int = 512
    overlap_units: int = 1
    token_counter: str = "approx_regex"
    token_multiplier: float = 1.3
    max_chunks_per_memory: int = 32
    failure_window_events: int = 12
    include_types: list[str] = Field(default_factory=lambda: list(CHUNK_TYPES))

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in {"event_aware", "card_only", "naive_fixed"}:
            raise ValueError("strategy must be event_aware|card_only|naive_fixed")
        return value

    @field_validator("token_counter")
    @classmethod
    def _known_counter(cls, value: str) -> str:
        if value != "approx_regex":
            raise ValueError("token_counter must be approx_regex")
        return value

    @field_validator("include_types")
    @classmethod
    def _known_types(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(CHUNK_TYPES))
        if unknown:
            raise ValueError(f"unknown chunk types: {unknown}")
        if not value:
            raise ValueError("include_types must not be empty")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def _bounds(self) -> ChunkingSettings:
        if self.target_tokens < 16:
            raise ValueError("chunking.target_tokens must be >= 16")
        if self.max_tokens < self.target_tokens:
            raise ValueError("chunking.max_tokens must be >= target_tokens")
        if self.max_tokens > 8192:
            raise ValueError("chunking.max_tokens must be <= 8192")
        if not 0 <= self.overlap_units <= 4:
            raise ValueError("chunking.overlap_units must be within [0, 4]")
        if self.max_chunks_per_memory < 1:
            raise ValueError("chunking.max_chunks_per_memory must be >= 1")
        if self.failure_window_events < 1:
            raise ValueError("chunking.failure_window_events must be >= 1")
        if not 1.0 <= self.token_multiplier <= 3.0:
            raise ValueError("chunking.token_multiplier must be within [1.0, 3.0]")
        return self


class ReliabilitySettings(_SettingsModel):
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0
    minimum_verified_outcomes_for_alert: int = 5
    suspect_posterior_mean_threshold: float = 0.55
    quarantine_posterior_mean_threshold: float = 0.35
    consecutive_failures_for_suspect: int = 3


class RepairCompilerSettings(_SettingsModel):
    enabled: bool = True
    generator: str = "deterministic"
    max_failure_window_events: int = 20
    max_candidate_generation_attempts: int = 2
    require_validation_for_promotion: bool = True
    max_validation_trials: int = 10
    max_strategy_actions: int = 8

    @field_validator("generator")
    @classmethod
    def _known_generator(cls, value: str) -> str:
        if value not in {"deterministic", "llm"}:
            raise ValueError("generator must be deterministic|llm")
        return value


class MemoryShiftSettings(_SettingsModel):
    enabled: bool = True
    schema_mismatch_status: str = "incompatible"
    version_mismatch_status: str = "suspect"
    revalidation_enabled: bool = True


class EvaluationSettings(_SettingsModel):
    default_split: str = "dev"
    frozen_memory_default: bool = True
    store_per_task_results: bool = True
    bootstrap_samples: int = 2000
    random_seed: int = 7


class Settings(_SettingsModel):
    app: AppSettings = Field(default_factory=AppSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    repair_llm: LlmSettings = Field(default_factory=LlmSettings)
    recall: RecallSettings = Field(default_factory=RecallSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    repair_compiler: RepairCompilerSettings = Field(default_factory=RepairCompilerSettings)
    memory_shift: MemoryShiftSettings = Field(default_factory=MemoryShiftSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    def safe_dump(self) -> dict[str, Any]:
        data = self.model_dump()
        data["storage"]["postgres_url"] = redact_url(data["storage"]["postgres_url"])
        data["elasticsearch"]["url"] = redact_url(data["elasticsearch"]["url"])
        data["elasticsearch"]["api_key"] = "***" if self.elasticsearch.api_key else ""
        for section in ("embeddings", "llm", "repair_llm"):
            data[section]["api_key"] = "***" if data[section].get("api_key") else ""
            data[section]["base_url"] = redact_url(data[section].get("base_url") or "")
        return data

    def validate_consistency(self) -> None:
        if self.elasticsearch.vector_dimension != self.embeddings.dimension:
            raise ConfigurationError(
                "elasticsearch.vector_dimension must match embeddings.dimension",
                details={
                    "vector_dimension": self.elasticsearch.vector_dimension,
                    "embedding_dimension": self.embeddings.dimension,
                },
            )
        if not 0.0 <= self.recall.no_safe_memory_threshold <= 1.0:
            raise ConfigurationError("recall.no_safe_memory_threshold must be within [0, 1]")
        if self.recall.reranker == "cross_encoder" and not self.recall.rerank_model_path.strip():
            raise ConfigurationError(
                "recall.reranker=cross_encoder requires recall.rerank_model_path pointing at "
                "an already-downloaded local CrossEncoder directory; RECAP never downloads weights"
            )
        if self.chunking.max_tokens > 8191:
            raise ConfigurationError("chunking.max_tokens must be below the embedding input limit")
        self._validate_embedding_role()
        self._validate_llm_role("llm", self.llm)
        self._validate_llm_role("repair_llm", self.repair_llm)
        if self.repair_compiler.generator == "llm" and self.repair_llm.provider == "disabled":
            raise ConfigurationError(
                "repair_compiler.generator=llm requires a configured repair_llm provider; "
                "set REPAIR_LLM_PROVIDER/REPAIR_LLM_BASE_URL/REPAIR_LLM_MODEL or switch the "
                "generator back to deterministic"
            )
        if (
            self.app.environment == "production"
            and self.embeddings.provider in {"hashing", "fake"}
            and not self.embeddings.allow_degraded
        ):
            raise ConfigurationError(
                "production requires a real embedding provider (openai_compatible) unless "
                "embeddings.allow_degraded is explicitly true; RECAP must not silently claim "
                "semantic retrieval while using a lexical hash embedder"
            )

    def _validate_embedding_role(self) -> None:
        embeddings = self.embeddings
        if embeddings.dimension < 1:
            raise ConfigurationError("embeddings.dimension must be >= 1")
        if embeddings.provider == "openai_compatible":
            if not embeddings.base_url.strip():
                raise ConfigurationError(
                    "embeddings.base_url is required for provider=openai_compatible"
                )
            if not embeddings.model_id.strip() or embeddings.model_id == (
                "recap-hashing-embedding-v1"
            ):
                raise ConfigurationError(
                    "embeddings.model_id is required for provider=openai_compatible; the "
                    "hashing default is not a real embedding model"
                )
        if embeddings.provider == "sentence-transformers" and not embeddings.model_path:
            raise ConfigurationError(
                "provider=sentence-transformers requires embeddings.model_path pointing at an "
                "already-downloaded local model directory; RECAP never downloads weights"
            )

    @staticmethod
    def _validate_llm_role(role: str, settings: LlmSettings) -> None:
        if settings.provider == "disabled":
            return
        if not settings.base_url.strip():
            raise ConfigurationError(f"{role}.base_url is required for provider=openai_compatible")
        if not settings.model.strip():
            raise ConfigurationError(f"{role}.model is required for provider=openai_compatible")


def redact_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***:***@{host}"


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply supported environment variables, later entries winning.

    Unprefixed names follow the README/provider conventions; ``RECAP_``-prefixed
    aliases are kept for existing deployments and take precedence.
    """

    dimension_from_embedding_env = False
    vector_dimension_explicit = bool(
        os.environ.get("RECAP_VECTOR_DIMENSION") or os.environ.get("ELASTICSEARCH_VECTOR_DIMENSION")
    )

    mapping: list[tuple[str, str, str, str]] = [
        ("DATABASE_URL", "storage", "postgres_url", "str"),
        ("ELASTICSEARCH_URL", "elasticsearch", "url", "str"),
        ("ELASTICSEARCH_API_KEY", "elasticsearch", "api_key", "str"),
        ("RECAP_LOG_LEVEL", "app", "log_level", "str"),
        ("RECAP_ENVIRONMENT", "app", "environment", "str"),
        ("RECAP_API_KEY", "app", "api_key", "str"),
        ("RECAP_CORS_ORIGINS", "app", "cors_origins", "str"),
        # Embedding role.
        ("EMBEDDING_PROVIDER", "embeddings", "provider", "str"),
        ("EMBEDDING_BASE_URL", "embeddings", "base_url", "str"),
        ("EMBEDDING_API_KEY", "embeddings", "api_key", "str"),
        ("EMBEDDING_MODEL", "embeddings", "model_id", "str"),
        ("EMBEDDING_MODEL_VERSION", "embeddings", "model_version", "str"),
        ("EMBEDDING_DIMENSIONS", "embeddings", "dimension", "int"),
        ("EMBEDDING_BATCH_SIZE", "embeddings", "batch_size", "int"),
        ("EMBEDDING_TIMEOUT_SECONDS", "embeddings", "timeout_seconds", "float"),
        ("EMBEDDING_CONNECT_TIMEOUT_SECONDS", "embeddings", "connect_timeout_seconds", "float"),
        ("EMBEDDING_READ_TIMEOUT_SECONDS", "embeddings", "read_timeout_seconds", "float"),
        ("EMBEDDING_WRITE_TIMEOUT_SECONDS", "embeddings", "write_timeout_seconds", "float"),
        ("EMBEDDING_POOL_TIMEOUT_SECONDS", "embeddings", "pool_timeout_seconds", "float"),
        ("EMBEDDING_OVERALL_TIMEOUT_SECONDS", "embeddings", "overall_timeout_seconds", "float"),
        ("EMBEDDING_MAX_RETRIES", "embeddings", "max_retries", "int"),
        ("EMBEDDING_SEND_DIMENSIONS", "embeddings", "send_dimensions", "bool"),
        ("EMBEDDING_MODEL_PATH", "embeddings", "model_path", "str"),
        ("EMBEDDING_DOCUMENT_PREFIX", "embeddings", "document_prefix", "str"),
        ("EMBEDDING_QUERY_PREFIX", "embeddings", "query_prefix", "str"),
        ("EMBEDDING_NORMALIZE", "embeddings", "normalize", "bool"),
        ("EMBEDDING_CACHE_ENABLED", "embeddings", "cache_enabled", "bool"),
        ("EMBEDDING_ALLOW_DEGRADED", "embeddings", "allow_degraded", "bool"),
        ("RECAP_EMBEDDING_PROVIDER", "embeddings", "provider", "str"),
        ("RECAP_EMBEDDING_DIMENSION", "embeddings", "dimension", "int"),
        ("RECAP_EMBEDDING_CACHE_ENABLED", "embeddings", "cache_enabled", "bool"),
        ("RECAP_EMBEDDING_ALLOW_DEGRADED", "embeddings", "allow_degraded", "bool"),
        # Agent LLM role.
        ("LLM_PROVIDER", "llm", "provider", "str"),
        ("LLM_BASE_URL", "llm", "base_url", "str"),
        ("LLM_API_KEY", "llm", "api_key", "str"),
        ("LLM_MODEL", "llm", "model", "str"),
        ("LLM_TIMEOUT_SECONDS", "llm", "timeout_seconds", "float"),
        ("LLM_CONNECT_TIMEOUT_SECONDS", "llm", "connect_timeout_seconds", "float"),
        ("LLM_READ_TIMEOUT_SECONDS", "llm", "read_timeout_seconds", "float"),
        ("LLM_WRITE_TIMEOUT_SECONDS", "llm", "write_timeout_seconds", "float"),
        ("LLM_POOL_TIMEOUT_SECONDS", "llm", "pool_timeout_seconds", "float"),
        ("LLM_OVERALL_TIMEOUT_SECONDS", "llm", "overall_timeout_seconds", "float"),
        ("LLM_NO_PROGRESS_TIMEOUT_SECONDS", "llm", "no_progress_timeout_seconds", "float"),
        ("LLM_STREAM", "llm", "stream", "bool"),
        ("LLM_MAX_CONCURRENCY", "llm", "max_concurrency", "int"),
        ("LLM_MAX_RETRIES", "llm", "max_retries", "int"),
        ("LLM_TEMPERATURE", "llm", "temperature", "float"),
        ("LLM_MAX_TOKENS", "llm", "max_tokens", "int"),
        ("LLM_STRUCTURED_MODE", "llm", "structured_mode", "str"),
        ("RECAP_LLM_PROVIDER", "llm", "provider", "str"),
        # RepairCompiler LLM role.
        ("REPAIR_LLM_PROVIDER", "repair_llm", "provider", "str"),
        ("REPAIR_LLM_BASE_URL", "repair_llm", "base_url", "str"),
        ("REPAIR_LLM_API_KEY", "repair_llm", "api_key", "str"),
        ("REPAIR_LLM_MODEL", "repair_llm", "model", "str"),
        ("REPAIR_LLM_TIMEOUT_SECONDS", "repair_llm", "timeout_seconds", "float"),
        (
            "REPAIR_LLM_CONNECT_TIMEOUT_SECONDS",
            "repair_llm",
            "connect_timeout_seconds",
            "float",
        ),
        ("REPAIR_LLM_READ_TIMEOUT_SECONDS", "repair_llm", "read_timeout_seconds", "float"),
        ("REPAIR_LLM_WRITE_TIMEOUT_SECONDS", "repair_llm", "write_timeout_seconds", "float"),
        ("REPAIR_LLM_POOL_TIMEOUT_SECONDS", "repair_llm", "pool_timeout_seconds", "float"),
        (
            "REPAIR_LLM_OVERALL_TIMEOUT_SECONDS",
            "repair_llm",
            "overall_timeout_seconds",
            "float",
        ),
        (
            "REPAIR_LLM_NO_PROGRESS_TIMEOUT_SECONDS",
            "repair_llm",
            "no_progress_timeout_seconds",
            "float",
        ),
        ("REPAIR_LLM_STREAM", "repair_llm", "stream", "bool"),
        ("REPAIR_LLM_MAX_CONCURRENCY", "repair_llm", "max_concurrency", "int"),
        ("REPAIR_LLM_MAX_RETRIES", "repair_llm", "max_retries", "int"),
        ("REPAIR_LLM_TEMPERATURE", "repair_llm", "temperature", "float"),
        ("REPAIR_LLM_MAX_TOKENS", "repair_llm", "max_tokens", "int"),
        ("REPAIR_LLM_STRUCTURED_MODE", "repair_llm", "structured_mode", "str"),
        ("RECAP_REPAIR_LLM_PROVIDER", "repair_llm", "provider", "str"),
        # Compiler and retrieval.
        ("REPAIR_GENERATOR", "repair_compiler", "generator", "str"),
        ("RECAP_REPAIR_GENERATOR", "repair_compiler", "generator", "str"),
        ("RECAP_RERANKER", "recall", "reranker", "str"),
        ("RERANK_PROVIDER", "recall", "reranker", "str"),
        ("RERANK_MODEL_PATH", "recall", "rerank_model_path", "str"),
        ("RERANK_MAX_LENGTH", "recall", "rerank_max_length", "int"),
        ("RERANK_BATCH_SIZE", "recall", "rerank_batch_size", "int"),
        ("RECAP_VECTOR_DIMENSION", "elasticsearch", "vector_dimension", "int"),
        ("ELASTICSEARCH_VECTOR_DIMENSION", "elasticsearch", "vector_dimension", "int"),
        ("RECAP_INDEX_GENERATION", "elasticsearch", "index_generation", "int"),
        ("ELASTICSEARCH_INDEX_PREFIX", "elasticsearch", "index_prefix", "str"),
        ("RECAP_INDEX_PREFIX", "elasticsearch", "index_prefix", "str"),
        ("ELASTICSEARCH_IDENTIFIER_ANALYSIS", "elasticsearch", "identifier_analysis", "bool"),
        ("ELASTICSEARCH_KNN_NUM_CANDIDATES", "elasticsearch", "knn_num_candidates", "int"),
        (
            "ELASTICSEARCH_DENSE_VECTOR_QUANTIZATION",
            "elasticsearch",
            "dense_vector_quantization",
            "str",
        ),
        ("ELASTICSEARCH_MEMORIES_READ_ALIAS", "elasticsearch", "memories_read_alias", "str"),
        ("ELASTICSEARCH_MEMORIES_WRITE_ALIAS", "elasticsearch", "memories_write_alias", "str"),
        ("ELASTICSEARCH_CHUNKS_READ_ALIAS", "elasticsearch", "chunks_read_alias", "str"),
        ("ELASTICSEARCH_CHUNKS_WRITE_ALIAS", "elasticsearch", "chunks_write_alias", "str"),
        ("RECAP_CHUNKS_INDEX_GENERATION", "elasticsearch", "chunks_index_generation", "int"),
        ("RECAP_CHUNKING_STRATEGY", "chunking", "strategy", "str"),
        ("RECAP_CHUNKING_VERSION", "chunking", "version", "str"),
        ("RECAP_CHUNKING_TARGET_TOKENS", "chunking", "target_tokens", "int"),
        ("RECAP_CHUNKING_MAX_TOKENS", "chunking", "max_tokens", "int"),
    ]
    for env_name, section, key, kind in mapping:
        value = _read_env(env_name, kind)
        if value is None:
            continue
        raw.setdefault(section, {})
        raw[section][key] = value
        if section == "embeddings" and key == "dimension":
            dimension_from_embedding_env = True

    reranker_enabled = _read_env("RERANKER_ENABLED", "bool")
    if reranker_enabled is False:
        raw.setdefault("recall", {})["reranker"] = "none"

    # A single embedding dimension should not require setting the ES mapping
    # dimension twice. An explicit vector dimension still wins.
    if dimension_from_embedding_env and not vector_dimension_explicit:
        raw.setdefault("elasticsearch", {})["vector_dimension"] = raw["embeddings"]["dimension"]
    return raw


def _read_env(name: str, kind: str) -> Any:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        if kind == "int":
            return int(value)
        if kind == "float":
            return float(value)
        if kind == "bool":
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{name} must be a boolean")
        return value
    except ValueError as exc:
        raise ConfigurationError(f"invalid value for {name}: {value}") from exc


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else Path(os.environ.get("RECAP_CONFIG", DEFAULT_CONFIG_PATH))
    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            raise ConfigurationError(f"invalid YAML config: {exc}") from exc
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigurationError("configuration root must be a mapping")
            raw = loaded
    raw = _apply_env_overrides(raw)
    try:
        settings = Settings.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigurationError(f"invalid configuration: {exc}") from exc
    settings.validate_consistency()
    return settings


@lru_cache(maxsize=4)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
