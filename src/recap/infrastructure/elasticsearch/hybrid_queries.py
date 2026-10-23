"""Elasticsearch query DSL for lexical, vector, and hybrid retrieval.

All DSL is confined here. The application-side RRF is the default so branch
ranks are always reproducible and inspectable; native ES RRF is available behind
a configuration flag and reports which path was used.
"""

from __future__ import annotations

from typing import Any

from recap.infrastructure.elasticsearch.mappings import (
    IDENTIFIER_LEXICAL_FIELDS,
    LEXICAL_FIELDS,
    PURE_LEXICAL_FIELDS,
)

ALLOWED_STATUSES_OPERATIONAL = ["active"]
ALLOWED_STATUSES_WITH_SUSPECT = ["active", "suspect"]


def _base_filters(
    *,
    workspace_id: str,
    allowed_statuses: list[str],
    include_incompatible: bool = False,
) -> list[dict[str, Any]]:
    """Non-negotiable ES-level filters only.

    Task family, tool names, environment fingerprint, and failure code are
    *not* filtered here: they are evaluated in the application so that a
    mismatched candidate can still be retrieved and explained as rejected.
    """
    filters: list[dict[str, Any]] = [
        {"term": {"workspace_id": workspace_id}},
        {"terms": {"status": allowed_statuses}},
    ]
    if not include_incompatible:
        filters.append({"bool": {"must_not": [{"term": {"compatibility_status": "incompatible"}}]}})
    return filters


def build_lexical_query(
    *,
    query: str,
    workspace_id: str,
    allowed_statuses: list[str],
    failure_signature: str = "",
    environment_fingerprint: str | None = None,
    environment_id: str | None = None,
    task_family: str | None = None,
    tool_names: list[str] | None = None,
    failure_code: str | None = None,
    size: int = 30,
    pure: bool = False,
    identifier_analysis: bool = False,
) -> dict[str, Any]:
    fields = dict(PURE_LEXICAL_FIELDS if pure else LEXICAL_FIELDS)
    if identifier_analysis:
        fields.update(IDENTIFIER_LEXICAL_FIELDS)
    should: list[dict[str, Any]] = []
    if query.strip():
        should.append(
            {
                "multi_match": {
                    "query": query,
                    "fields": [f"{field}^{boost}" for field, boost in fields.items()],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        )
    if failure_signature.strip() and not pure:
        should.append(
            {
                "match_phrase": {
                    "failure_signature": {
                        "query": failure_signature,
                        "boost": 4.0,
                        "slop": 2,
                    }
                }
            }
        )
    if not should:
        should.append({"match_all": {"boost": 0.1}})
    # Soft relevance boosts; these never exclude candidates. A pure retrieval
    # ablation receives only the natural-language query, so metadata signals are
    # not allowed to leak into its ranking.
    if failure_code and not pure:
        should.append({"term": {"failure_code": {"value": failure_code, "boost": 2.0}}})
    if task_family and not pure:
        should.append({"term": {"task_families": {"value": task_family, "boost": 1.2}}})
    if tool_names and not pure:
        should.append({"terms": {"tool_names": tool_names, "boost": 1.1}})
    return {
        "size": size,
        "track_total_hits": False,
        "_source": [
            "memory_id",
            "memory_version",
            "status",
            "title",
            "workspace_id",
            "environment_fingerprint",
            "environment_id",
            "compatibility_status",
            "failure_signature",
            "failure_code",
            "task_families",
            "tool_names",
            "content_hash",
            "risk_level",
            "side_effect_class",
            "verified_successes",
            "verified_failures",
            "last_validated_at",
        ],
        "query": {
            "bool": {
                "should": should,
                "minimum_should_match": 1,
                "filter": _base_filters(
                    workspace_id=workspace_id,
                    allowed_statuses=allowed_statuses,
                ),
            }
        },
        "sort": [{"_score": {"order": "desc"}}],
    }


def build_vector_query(
    *,
    embedding: list[float],
    workspace_id: str,
    allowed_statuses: list[str],
    environment_fingerprint: str | None = None,
    environment_id: str | None = None,
    task_family: str | None = None,
    tool_names: list[str] | None = None,
    failure_code: str | None = None,
    size: int = 30,
    num_candidates: int = 200,
) -> dict[str, Any]:
    return {
        "size": size,
        "track_total_hits": False,
        "_source": [
            "memory_id",
            "memory_version",
            "status",
            "title",
            "workspace_id",
            "environment_fingerprint",
            "environment_id",
            "compatibility_status",
            "failure_signature",
            "failure_code",
            "task_families",
            "tool_names",
            "content_hash",
            "risk_level",
            "side_effect_class",
            "verified_successes",
            "verified_failures",
            "last_validated_at",
        ],
        "knn": {
            "field": "content_embedding",
            "query_vector": embedding,
            "k": size,
            "num_candidates": max(num_candidates, size),
            "filter": {
                "bool": {
                    "filter": _base_filters(
                        workspace_id=workspace_id,
                        allowed_statuses=allowed_statuses,
                    )
                }
            },
        },
    }


def build_es_rrf_query(
    *,
    query: str,
    embedding: list[float],
    workspace_id: str,
    allowed_statuses: list[str],
    failure_signature: str = "",
    environment_fingerprint: str | None = None,
    environment_id: str | None = None,
    task_family: str | None = None,
    tool_names: list[str] | None = None,
    failure_code: str | None = None,
    size: int = 30,
    rank_constant: int = 60,
    num_candidates: int = 200,
    pure: bool = False,
    identifier_analysis: bool = False,
) -> dict[str, Any]:
    """Native Elasticsearch RRF retriever (8.14+ syntax).

    ``knn`` is a retriever in its own right; wrapping it in ``standard`` (as an
    earlier draft did) sends a query clause where the query DSL expects one and
    Elasticsearch rejects the request. The native path is license-gated in some
    distributions, so callers must surface the failure instead of silently
    falling back to application-side fusion.
    """
    lexical = build_lexical_query(
        query=query,
        workspace_id=workspace_id,
        allowed_statuses=allowed_statuses,
        failure_signature=failure_signature,
        environment_fingerprint=environment_fingerprint,
        environment_id=environment_id,
        task_family=task_family,
        tool_names=tool_names,
        failure_code=failure_code,
        size=size,
        pure=pure,
        identifier_analysis=identifier_analysis,
    )
    vector = build_vector_query(
        embedding=embedding,
        workspace_id=workspace_id,
        allowed_statuses=allowed_statuses,
        size=size,
        num_candidates=num_candidates,
    )
    return {
        "size": size,
        "track_total_hits": False,
        "_source": lexical["_source"],
        "retriever": {
            "rrf": {
                "retrievers": [
                    {"standard": {"query": lexical["query"]}},
                    {"knn": vector["knn"]},
                ],
                "rank_constant": rank_constant,
            }
        },
    }
