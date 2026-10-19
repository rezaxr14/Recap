"""Embedding fingerprinting.

Vectors from two different models, dimensions, or task prefixes must never share
an Elasticsearch dense_vector field. The fingerprint is the canonical identity
of the embedding configuration and is stored on every chunk document and
retrieval decision.
"""

from __future__ import annotations

import hashlib

from recap.domain.environments import canonical_json


def embedding_fingerprint(
    *,
    provider: str,
    model_id: str,
    model_version: str,
    dimension: int,
    normalize: bool,
    document_prefix: str,
    query_prefix: str,
) -> str:
    payload = {
        "provider": provider,
        "model_id": model_id,
        "model_version": model_version,
        "dimension": dimension,
        "normalize": normalize,
        "document_prefix": document_prefix,
        "query_prefix": query_prefix,
    }
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
