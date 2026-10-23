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

# Identifier-aware analysis. The default analyzer keeps ``INVALID_CURSOR`` as a
# single token, so a prose query such as "invalid cursor" cannot match the code.
# This analyzer also emits the delimited parts (``invalid``, ``cursor``) and
# keeps the original token, which lets identifiers match natural language
# without changing the stored value.
IDENTIFIER_ANALYZER = "recap_identifier"
IDENTIFIER_FILTER = "recap_identifier_delimiters"
IDENTIFIER_SUBFIELD = "identifier"
IDENTIFIER_TEXT_FIELDS = ("title", "summary", "failure_signature", "failure_code")

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

# Ablation fields with every boost removed. Used to measure the analyzer and
# fusion layers without the metadata boosts that make v2 retrieval look easy.
PURE_LEXICAL_FIELDS: dict[str, float] = {
    "title": 1.0,
    "summary": 1.0,
    "strategy_text": 1.0,
    "preconditions_text": 1.0,
    "postconditions_text": 1.0,
    "limitations_text": 1.0,
}

IDENTIFIER_LEXICAL_FIELDS: dict[str, float] = {
    f"{field}.{IDENTIFIER_SUBFIELD}": 1.0 for field in IDENTIFIER_TEXT_FIELDS
}

DENSE_VECTOR_INDEX_TYPES = ("hnsw", "int8_hnsw")


def _identifier_analysis() -> dict[str, Any]:
    return {
        "analysis": {
            "filter": {
                IDENTIFIER_FILTER: {
                    "type": "word_delimiter_graph",
                    "generate_word_parts": True,
                    "generate_number_parts": True,
                    "split_on_case_change": True,
                    "split_on_numerics": True,
                    "preserve_original": True,
                    "stem_english_possessive": False,
                }
            },
            "analyzer": {
                IDENTIFIER_ANALYZER: {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [IDENTIFIER_FILTER, "lowercase"],
                }
            },
        }
    }


def _identifier_properties() -> dict[str, Any]:
    properties = dict(MEMORY_PROPERTIES)
    for field in IDENTIFIER_TEXT_FIELDS:
        spec = dict(properties[field])
        fields = dict(spec.get("fields") or {})
        fields[IDENTIFIER_SUBFIELD] = {
            "type": "text",
            "analyzer": IDENTIFIER_ANALYZER,
        }
        spec["fields"] = fields
        properties[field] = spec
    return properties


def memory_index_body(
    *,
    dimension: int,
    number_of_shards: int = 1,
    meta: dict[str, Any] | None = None,
    identifier_analysis: bool = False,
    dense_vector_type: str = "hnsw",
) -> dict[str, Any]:
    if dense_vector_type not in DENSE_VECTOR_INDEX_TYPES:
        raise ValueError(
            f"dense_vector_type must be one of {DENSE_VECTOR_INDEX_TYPES}, "
            f"got {dense_vector_type!r}"
        )
    properties = _identifier_properties() if identifier_analysis else dict(MEMORY_PROPERTIES)
    properties["content_embedding"] = {
        "type": "dense_vector",
        "dims": dimension,
        "index": True,
        "similarity": "cosine",
        "index_options": {
            "type": dense_vector_type,
            "m": 16,
            "ef_construction": 100,
        },
    }
    mappings: dict[str, Any] = {
        "dynamic": "strict",
        "properties": properties,
        # The vector is an index structure, not document content. Keeping a copy
        # in _source roughly doubles index size and hides any quantization win,
        # and no read path needs it (search returns metadata, not embeddings).
        "_source": {"excludes": ["content_embedding"]},
    }
    if meta:
        mappings["_meta"] = dict(meta)
    body: dict[str, Any] = {
        "settings": {
            "number_of_shards": number_of_shards,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
        },
        "mappings": mappings,
    }
    if identifier_analysis:
        body["settings"].update(_identifier_analysis())
    return body


# ------------------------------------------------------------------- chunks
CHUNK_PROPERTIES: dict[str, Any] = {
    "chunk_id": {"type": "keyword"},
    "text_hash": {"type": "keyword"},
    "memory_id": {"type": "keyword"},
    "memory_version": {"type": "integer"},
    "workspace_id": {"type": "keyword"},
    "run_id": {"type": "keyword"},
    "chunk_type": {"type": "keyword"},
    "chunk_text": {"type": "text", "analyzer": "standard"},
    "seq_start": {"type": "integer"},
    "seq_end": {"type": "integer"},
    "anchor_seq": {"type": "integer"},
    "token_estimate": {"type": "integer"},
    "event_ids": {"type": "keyword"},
    "failure_code": {"type": "keyword"},
    "tool_names": {"type": "keyword"},
    "tool_versions": {"type": "keyword"},
    "environment_id": {"type": "keyword"},
    "environment_fingerprint": {"type": "keyword"},
    "task_families": {"type": "keyword"},
    "chunker_version": {"type": "keyword"},
    "chunk_strategy": {"type": "keyword"},
    "embedding_model_id": {"type": "keyword"},
    "embedding_fingerprint": {"type": "keyword"},
    "redaction_rule_version": {"type": "keyword"},
    "exact_ids": {"type": "keyword"},
    "truncated": {"type": "boolean"},
    "active_for_retrieval": {"type": "boolean"},
    "created_at": {"type": "date"},
}

CHUNK_LEXICAL_FIELDS: dict[str, float] = {
    "chunk_text": 2.0,
    "failure_code": 2.5,
    "tool_names": 1.5,
    "task_families": 1.0,
}


def chunk_index_body(
    *,
    dimension: int,
    number_of_shards: int = 1,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    properties = dict(CHUNK_PROPERTIES)
    properties["content_embedding"] = {
        "type": "dense_vector",
        "dims": dimension,
        "index": True,
        "similarity": "cosine",
        "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
    }
    mappings: dict[str, Any] = {
        "dynamic": "strict",
        "properties": properties,
    }
    if meta:
        # The fingerprint/dimension identity of the vectors in this field.
        # ``IndexManager`` refuses to start when this disagrees with settings.
        mappings["_meta"] = dict(meta)
    return {
        "settings": {
            "number_of_shards": number_of_shards,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
        },
        "mappings": mappings,
    }
