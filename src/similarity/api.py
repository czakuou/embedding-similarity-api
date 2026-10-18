"""FastAPI app factory: slice routers, /health, /ready, /metrics, Problem Details."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable
from contextlib import asynccontextmanager
from typing import cast

import redis.asyncio as redis
import structlog
from arq import create_pool
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from limits.aio.strategies import FixedWindowRateLimiter
from prometheus_fastapi_instrumentator import Instrumentator
from starlette import status

from similarity.analysis import arq_redis_settings, deserialize_job, serialize_job
from similarity.config import ApiSettings
from similarity.dependencies import RedisDep
from similarity.observability import configure_logging
from similarity.results import router as results_router
from similarity.submit import make_redis_storage, rate_limit_item
from similarity.submit import router as submit_router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = app.state.settings
    app.state.redis = redis.from_url(settings.redis_url, decode_responses=True)
    app.state.arq_pool = await create_pool(
        arq_redis_settings(settings.redis_url),
        default_queue_name=settings.queue_name,
        job_serializer=serialize_job,
        job_deserializer=deserialize_job,
    )
    storage = make_redis_storage(settings.redis_url)
    app.state.rate_limiter = FixedWindowRateLimiter(storage)
    app.state.rate_limit_item = rate_limit_item(settings.rate_limit)
    try:
        yield
    finally:
        await app.state.arq_pool.aclose()
        await app.state.redis.aclose()


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    configure_logging(settings.log_level)

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings

    app.include_router(submit_router)
    app.include_router(results_router)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok"})

    @app.get("/ready")
    async def ready(redis_client: RedisDep) -> JSONResponse:
        try:
            # redis-py's stub shares one signature between the sync and async clients.
            await cast("Awaitable[bool]", redis_client.ping())
        except redis.RedisError:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not ready"}
            )
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})

    Instrumentator().instrument(app).expose(app)

    return app
