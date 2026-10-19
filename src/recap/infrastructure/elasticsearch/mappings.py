"""Explicit Elasticsearch mappings.

Versioned index with:
- BM25 ``text`` fields for lexical retrieval
- ``dense_vector`` (cosine, HNSW) for semantic retrieval
- keyword fields for hard filters
- no dynamic mapping for free-form payloads

Keep all DSL here so version-specific syntax does not leak into business logic.
"""

from __future__ import annotations

from typing import Any

MEMORY_PROPERTIES: dict[str, Any] = {
    "memory_id": {"type": "keyword"},
    "memory_version": {"type": "integer"},
    "workspace_id": {"type": "keyword"},
    "status": {"type": "keyword"},
    "title": {
        "type": "text",
        "analyzer": "standard",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
    },
    "summary": {"type": "text", "analyzer": "standard"},
    "failure_signature": {
        "type": "text",
        "analyzer": "standard",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}},
    },
    "failure_code": {"type": "keyword"},
    "task_families": {"type": "keyword"},
    "tool_names": {"type": "keyword"},
    "tool_versions": {"type": "keyword"},
    "tool_schema_fingerprints": {"type": "keyword"},
    "environment_id": {"type": "keyword"},
    "environment_fingerprint": {"type": "keyword"},
    "environment_type": {"type": "keyword"},
    "preconditions_text": {"type": "text", "analyzer": "standard"},
    "strategy_text": {"type": "text", "analyzer": "standard"},
    "postconditions_text": {"type": "text", "analyzer": "standard"},
    "limitations_text": {"type": "text", "analyzer": "standard"},
    "evidence_summary": {"type": "text", "analyzer": "standard"},
    "created_at": {"type": "date"},
    "last_validated_at": {"type": "date"},
    "active_for_retrieval": {"type": "boolean"},
    "compatibility_status": {"type": "keyword"},
    "embedding_model_id": {"type": "keyword"},
    "content_hash": {"type": "keyword"},
    "risk_level": {"type": "keyword"},
    "side_effect_class": {"type": "keyword"},
    "verified_successes": {"type": "integer"},
    "verified_failures": {"type": "integer"},
    "index_generation": {"type": "integer"},
}

# Text fields searched by the lexical branch, with field boosts.
LEXICAL_FIELDS: dict[str, float] = {
    "failure_signature": 3.0,
    "failure_code": 2.5,
    "title": 2.0,
    "summary": 1.5,
    "strategy_text": 1.2,
    "preconditions_text": 1.0,
    "postconditions_text": 0.8,
    "limitations_text": 0.5,
}


def memory_index_body(*, dimension: int, number_of_shards: int = 1) -> dict[str, Any]:
    properties = dict(MEMORY_PROPERTIES)
    properties["content_embedding"] = {
        "type": "dense_vector",
        "dims": dimension,
        "index": True,
        "similarity": "cosine",
        "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
    }
    return {
        "settings": {
            "number_of_shards": number_of_shards,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
        },
        "mappings": {
            "dynamic": "strict",
            "properties": properties,
        },
    }
