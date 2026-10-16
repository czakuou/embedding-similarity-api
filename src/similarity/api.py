"""FastAPI app factory: slice routers, /health, /ready, /metrics, Problem Details."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from limits.aio.strategies import FixedWindowRateLimiter
from prometheus_fastapi_instrumentator import Instrumentator

from similarity.config import ApiSettings
from similarity.observability import configure_logging
from similarity.submit import make_redis_storage, rate_limit_item
from similarity.submit import router as submit_router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = app.state.settings
    app.state.redis = redis.from_url(settings.redis_url, decode_responses=True)
    storage = make_redis_storage(settings.redis_url)
    app.state.rate_limiter = FixedWindowRateLimiter(storage)
    app.state.rate_limit_item = rate_limit_item(settings.rate_limit)
    try:
        yield
    finally:
        await app.state.redis.aclose()


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    configure_logging(settings.log_level)

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings

    app.include_router(submit_router)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        try:
            await app.state.redis.ping()
        except redis.RedisError:
            return JSONResponse(status_code=503, content={"status": "not ready"})
        return JSONResponse(status_code=200, content={"status": "ready"})

    Instrumentator().instrument(app).expose(app)

    return app
