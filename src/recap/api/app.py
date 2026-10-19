"""FastAPI application factory and error handling."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from recap import __version__
from recap.config import get_settings
from recap.container import Container
from recap.errors import RecapError
from recap.logging import configure_logging, correlation_id
from recap.api.routes import (
    environments,
    health,
    memories,
    repairs,
    retrieval,
    runs,
)


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
        if app.state.container is None:
            app.state.container = Container.build(settings)
        return app.state.container

    app.state.get_container = get_container

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
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
