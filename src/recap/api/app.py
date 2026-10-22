"""FastAPI application factory and error handling."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from recap import __version__
from recap.api.routes import (
    environments,
    health,
    memories,
    repairs,
    retrieval,
    runs,
)
from recap.config import get_settings
from recap.container import Container
from recap.errors import RecapError
from recap.logging import configure_logging, correlation_id


def create_app(container: Container | None = None) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.app.log_level)
    app = FastAPI(
        title="RECAP",
        version=__version__,
        description="Retrieval-Enhanced Correction and Adaptive Persistence",
    )
    app.state.container = container

    def get_container() -> Container:
        container: Container | None = app.state.container
        if container is None:
            container = Container.build(settings)
            app.state.container = container
        return container

    app.state.get_container = get_container

    if settings.app.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.app.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type", "x-api-key", "x-correlation-id"],
            max_age=600,
        )

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next: Callable[[Request], Any]) -> Any:
        # Local development and the static replay app need no auth. A deployed
        # API must set RECAP_API_KEY; the comparison is constant-time.
        if settings.app.api_key and request.url.path not in {"/health/live", "/health/ready"}:
            supplied = request.headers.get("x-api-key", "")
            if not secrets.compare_digest(supplied, settings.app.api_key):
                return JSONResponse(
                    status_code=401,
                    content={
                        "code": "unauthorized",
                        "message": "missing or invalid X-API-Key",
                        "details": {},
                    },
                )
        return await call_next(request)

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next: Callable[[Request], Any]) -> Any:
        header = request.headers.get("x-correlation-id")
        token = correlation_id.set(header or None)
        try:
            response = await call_next(request)
        finally:
            correlation_id.reset(token)
        if header:
            response.headers["x-correlation-id"] = header
        return response

    @app.exception_handler(RecapError)
    async def recap_error_handler(_: Request, exc: RecapError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        # Do not leak stack traces or internal details to clients.
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "internal server error", "details": {}},
        )

    prefix = settings.app.api_prefix
    app.include_router(health.router)
    app.include_router(runs.router, prefix=prefix)
    app.include_router(memories.router, prefix=prefix)
    app.include_router(retrieval.router, prefix=prefix)
    app.include_router(repairs.router, prefix=prefix)
    app.include_router(environments.router, prefix=prefix)
    return app


app = create_app()
