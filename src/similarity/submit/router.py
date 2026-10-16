"""FastAPI route for POST /v1/similarity-analyses."""

from __future__ import annotations

import random

import redis.asyncio as redis
import structlog
from fastapi import APIRouter, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse
from limits import RateLimitItem
from limits.aio.strategies import FixedWindowRateLimiter

from similarity.config import ApiSettings
from similarity.submit.enqueue import decide_and_enqueue
from similarity.submit.outcomes import (
    Accepted,
    Invalid,
    QueueFull,
    RateLimited,
    outcome_to_response,
)

logger = structlog.get_logger()

router = APIRouter()


@router.post("/v1/similarity-analyses")
async def submit_analysis(
    request: Request,
    file: UploadFile,
    query: str = Form(default=""),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
) -> JSONResponse:
    settings: ApiSettings = request.app.state.settings
    redis_client: redis.Redis = request.app.state.redis
    limiter: FixedWindowRateLimiter = request.app.state.rate_limiter
    rate_limit: RateLimitItem = request.app.state.rate_limit_item

    raw_content = await file.read(settings.max_file_size_bytes + 1)
    content_too_large = len(raw_content) > settings.max_file_size_bytes

    outcome = await decide_and_enqueue(
        redis_client=redis_client,
        limiter=limiter,
        rate_limit=rate_limit,
        settings=settings,
        client_id=x_client_id,
        filename=file.filename,
        raw_content=raw_content,
        content_too_large=content_too_large,
        query=query,
        rng=random.Random(),  # noqa: S311 -- jitter for Retry-After, not cryptographic
    )

    match outcome:
        case Accepted():
            logger.info("analysis.accepted", analysis_id=outcome.analysis_id)
        case Invalid(reason=reason):
            logger.info("analysis.rejected", reason=reason)
        case RateLimited():
            logger.info("analysis.rejected", reason="rate_limited")
        case QueueFull():
            logger.info("analysis.rejected", reason="queue_full")

    return outcome_to_response(outcome)
