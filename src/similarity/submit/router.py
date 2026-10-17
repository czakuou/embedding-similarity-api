"""FastAPI route for POST /v1/similarity-analyses."""

from __future__ import annotations

import random

import structlog
from fastapi import APIRouter, Form, Header, UploadFile
from fastapi.responses import JSONResponse

from similarity.api_paths import SIMILARITY_ANALYSES_PATH
from similarity.dependencies import RateLimiterDep, RateLimitItemDep, RedisDep, SettingsDep
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


@router.post(SIMILARITY_ANALYSES_PATH)
async def submit_analysis(
    file: UploadFile,
    settings: SettingsDep,
    redis_client: RedisDep,
    limiter: RateLimiterDep,
    rate_limit: RateLimitItemDep,
    query: str = Form(default=""),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
) -> JSONResponse:
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
