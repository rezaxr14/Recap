"""Elasticsearch client construction and readiness checks.

Connection failures produce explicit ``SearchUnavailable`` errors; callers must
not silently claim a search happened.
"""

from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch, exceptions as es_exceptions

from recap.config import ElasticsearchSettings
from recap.errors import SearchUnavailable
from recap.logging import get_logger

logger = get_logger(__name__)


def build_client(settings: ElasticsearchSettings) -> Elasticsearch:
    kwargs: dict[str, Any] = {
        "hosts": [settings.url],
        "request_timeout": settings.request_timeout_seconds,
        "retry_on_timeout": True,
        "max_retries": 2,
    }
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    return Elasticsearch(**kwargs)


def check_cluster(client: Elasticsearch) -> dict[str, Any]:
    try:
        info = client.info()
        health = client.cluster.health(wait_for_status="yellow", timeout="5s")
    except es_exceptions.ConnectionError as exc:
        raise SearchUnavailable("cannot connect to Elasticsearch", details={"error": str(exc)}) from exc
    except es_exceptions.ApiError as exc:  # pragma: no cover - cluster-side
        raise SearchUnavailable("Elasticsearch returned an error", details={"error": str(exc)}) from exc
    return {
        "version": info.get("version", {}).get("number", "unknown"),
        "cluster_name": info.get("cluster_name", ""),
        "status": health.get("status", "unknown"),
        "number_of_nodes": health.get("number_of_nodes", 0),
    }
