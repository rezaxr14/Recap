"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from recap.errors import SearchUnavailable

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def ready(request: Request) -> dict[str, object]:
    container = request.app.state.get_container()
    health = container.health()
    db_ok = health.get("database") == "ok"
    es = health.get("elasticsearch")
    es_ok = isinstance(es, dict) and es.get("status") in {"green", "yellow"}
    return {
        "status": "ready" if db_ok and es_ok else "degraded",
        "database": health.get("database"),
        "elasticsearch": es,
    }


@router.get("/api/v1/diagnostics")
def diagnostics(request: Request) -> dict[str, object]:
    container = request.app.state.get_container()
    settings = container.settings
    result: dict[str, object] = {
        "version": "0.1.0",
        "config": settings.safe_dump(),
        "embedding_model_id": getattr(container.embeddings, "model_id", ""),
    }
    try:
        result["search"] = container.index_manager.health()
    except SearchUnavailable as exc:
        result["search"] = {"status": "unavailable", "detail": exc.message}
    return result
