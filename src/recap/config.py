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
from pydantic import BaseModel, Field, field_validator

from recap.errors import ConfigurationError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class AppSettings(BaseModel):
    name: str = "recap"
    environment: str = "local"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"


class StorageSettings(BaseModel):
    postgres_url: str = "postgresql+psycopg://recap:recap@localhost:5433/recap"
    max_event_payload_bytes: int = 65_536


class ElasticsearchSettings(BaseModel):
    url: str = "http://localhost:9200"
    api_key: str = ""
    index_prefix: str = "recap"
    memories_read_alias: str = "recap-memories-read"
    memories_write_alias: str = "recap-memories-write"
    index_generation: int = 1
    request_timeout_seconds: float = 5.0
    vector_dimension: int = 256
    hybrid_strategy: str = "app_rrf"
    rrf_k: int = 60

    @field_validator("hybrid_strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in {"app_rrf", "es_rrf"}:
            raise ValueError("hybrid_strategy must be 'app_rrf' or 'es_rrf'")
        return value

    @property
    def write_index(self) -> str:
        return f"{self.index_prefix}-memories-v{self.index_generation}"


class EmbeddingSettings(BaseModel):
    provider: str = "hashing"
    model_id: str = "recap-hashing-embedding-v1"
    model_version: str = "1"
    dimension: int = 256
    batch_size: int = 32

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        if value not in {"hashing", "fake", "sentence-transformers"}:
            raise ValueError("provider must be hashing|fake|sentence-transformers")
        return value


class RecallSettings(BaseModel):
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

    @field_validator("reranker")
    @classmethod
    def _known_reranker(cls, value: str) -> str:
        if value not in {"none", "lexical"}:
            raise ValueError("reranker must be 'none' or 'lexical'")
        return value


class ReliabilitySettings(BaseModel):
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0
    minimum_verified_outcomes_for_alert: int = 5
    suspect_posterior_mean_threshold: float = 0.55
    quarantine_posterior_mean_threshold: float = 0.35
    consecutive_failures_for_suspect: int = 3


class RepairCompilerSettings(BaseModel):
    enabled: bool = True
    max_failure_window_events: int = 20
    max_candidate_generation_attempts: int = 2
    require_validation_for_promotion: bool = True
    max_validation_trials: int = 10
    max_strategy_actions: int = 8


class MemoryShiftSettings(BaseModel):
    enabled: bool = True
    schema_mismatch_status: str = "incompatible"
    version_mismatch_status: str = "suspect"
    revalidation_enabled: bool = True


class EvaluationSettings(BaseModel):
    default_split: str = "dev"
    frozen_memory_default: bool = True
    store_per_task_results: bool = True
    bootstrap_samples: int = 2000
    random_seed: int = 7


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    recall: RecallSettings = Field(default_factory=RecallSettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    repair_compiler: RepairCompilerSettings = Field(default_factory=RepairCompilerSettings)
    memory_shift: MemoryShiftSettings = Field(default_factory=MemoryShiftSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    def safe_dump(self) -> dict[str, Any]:
        data = self.model_dump()
        data["storage"]["postgres_url"] = _redact_url(data["storage"]["postgres_url"])
        data["elasticsearch"]["api_key"] = "***" if self.elasticsearch.api_key else ""
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


def _redact_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***:***@{host}"


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "DATABASE_URL": ("storage", "postgres_url"),
        "ELASTICSEARCH_URL": ("elasticsearch", "url"),
        "ELASTICSEARCH_API_KEY": ("elasticsearch", "api_key"),
        "RECAP_LOG_LEVEL": ("app", "log_level"),
        "RECAP_ENVIRONMENT": ("app", "environment"),
        "RECAP_EMBEDDING_PROVIDER": ("embeddings", "provider"),
        "RECAP_EMBEDDING_DIMENSION": ("embeddings", "dimension"),
        "RECAP_VECTOR_DIMENSION": ("elasticsearch", "vector_dimension"),
        "RECAP_INDEX_GENERATION": ("elasticsearch", "index_generation"),
        "RECAP_RERANKER": ("recall", "reranker"),
    }
    for env_name, (section, key) in mapping.items():
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        raw.setdefault(section, {})
        if key in {"dimension", "vector_dimension", "index_generation"}:
            raw[section][key] = int(value)
        else:
            raw[section][key] = value
    return raw


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
