"""GET + DELETE slice: ownership check, 404 / 410 / 409 decisions, reads and deletes."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import UUID

import redis.asyncio as redis
from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from starlette import status

from similarity.analysis import (
    AnalysisRecord,
    Status,
    analysis_key,
    hash_to_record,
    missing_record_outcome,
    validate_client_id,
)
from similarity.api_paths import SIMILARITY_ANALYSIS_ITEM_ROUTE, analysis_url
from similarity.dependencies import ClockDep, RedisDep, SettingsDep


@dataclass(frozen=True, slots=True)
class Found:
    record: AnalysisRecord


@dataclass(frozen=True, slots=True)
class NotFound:
    """Never existed, belongs to another client, or was deleted by the client."""


@dataclass(frozen=True, slots=True)
class Gone:
    """Existed but its TTL has elapsed; no tombstone is kept."""


GetOutcome = Found | NotFound | Gone


def decide_get_outcome(
    record: AnalysisRecord | None,
    *,
    analysis_id: UUID,
    client_id: str,
    result_ttl_s: int,
    now: datetime,
) -> GetOutcome:
    """Pure decision: ownership mismatch is indistinguishable from not-found, so
    existence is never leaked to a client that doesn't own the record. A missing
    record falls back to the UUIDv7 timestamp to decide 404 vs 410, since no
    tombstone is stored.
    """
    if record is None:
        outcome = missing_record_outcome(analysis_id, result_ttl_s, now)
        return Gone() if outcome == "gone" else NotFound()
    if record.client_id != client_id:
        return NotFound()
    return Found(record=record)


@dataclass(frozen=True, slots=True)
class DeletableNow:
    """Owned by this client and in a terminal state: safe to delete."""

    record: AnalysisRecord


@dataclass(frozen=True, slots=True)
class StillInProgress:
    pass


DeleteOutcome = DeletableNow | NotFound | Gone | StillInProgress


def decide_delete_outcome(
    record: AnalysisRecord | None,
    *,
    analysis_id: UUID,
    client_id: str,
    result_ttl_s: int,
    now: datetime,
) -> DeleteOutcome:
    """Pure decision for DELETE."""
    if record is None:
        outcome = missing_record_outcome(analysis_id, result_ttl_s, now)
        return Gone() if outcome == "gone" else NotFound()
    if record.client_id != client_id:
        return NotFound()
    if record.status in (Status.QUEUED, Status.PROCESSING):
        return StillInProgress()
    return DeletableNow(record=record)


async def fetch_record(redis_client: redis.Redis, analysis_id: UUID) -> AnalysisRecord | None:
    # redis-py's stub shares one signature between the sync and async clients, so it
    # types HGETALL as `Awaitable[dict] | dict`; on redis.asyncio.Redis it is always
    # awaitable, and always str:str since the app sets decode_responses=True.
    raw = cast("Awaitable[dict[str, str]]", redis_client.hgetall(analysis_key(str(analysis_id))))
    fields = await raw
    if not fields:
        return None
    return hash_to_record(fields)


def invalid_client_id_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": "missing or invalid X-Client-Id",
            "status": status.HTTP_422_UNPROCESSABLE_CONTENT,
            "code": "VALIDATION_ERROR",
        },
    )


def record_to_response_body(record: AnalysisRecord) -> dict[str, object]:
    location = analysis_url(record.id)
    body: dict[str, object] = {
        "id": record.id,
        "status": record.status.value,
        "attempts": record.attempts,
        "created_at": record.created_at.isoformat(),
        "expires_at": record.expires_at.isoformat(),
        "links": {"self": location},
    }
    if record.updated_at is not None:
        body["updated_at"] = record.updated_at.isoformat()
    if record.status is Status.DONE:
        body["result"] = record.result
        body["document_truncated"] = record.document_truncated
    if record.status is Status.FAILED and record.error is not None:
        body["error"] = {
            "code": record.error.code,
            "message": record.error.message,
            "attempts": record.error.attempts,
        }
    return body


router = APIRouter()


@router.get(SIMILARITY_ANALYSIS_ITEM_ROUTE)
async def get_analysis(
    analysis_id: UUID,
    settings: SettingsDep,
    redis_client: RedisDep,
    clock: ClockDep,
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
) -> JSONResponse:
    validated_client_id = validate_client_id(x_client_id)
    if validated_client_id is None:
        return invalid_client_id_response()

    record = await fetch_record(redis_client, analysis_id)
    outcome = decide_get_outcome(
        record,
        analysis_id=analysis_id,
        client_id=validated_client_id,
        result_ttl_s=settings.result_ttl_s,
        now=clock(),
    )

    match outcome:
        case Found(record=found_record):
            headers = {}
            if found_record.status in (Status.QUEUED, Status.PROCESSING):
                headers["Retry-After"] = str(settings.poll_interval_hint_s)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                headers=headers,
                content=record_to_response_body(found_record),
            )
        case Gone():
            return JSONResponse(status_code=status.HTTP_410_GONE, content=None)
        case NotFound():
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=None)


@router.delete(SIMILARITY_ANALYSIS_ITEM_ROUTE)
async def delete_analysis(
    analysis_id: UUID,
    settings: SettingsDep,
    redis_client: RedisDep,
    clock: ClockDep,
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
) -> JSONResponse:
    validated_client_id = validate_client_id(x_client_id)
    if validated_client_id is None:
        return invalid_client_id_response()

    record = await fetch_record(redis_client, analysis_id)
    outcome = decide_delete_outcome(
        record,
        analysis_id=analysis_id,
        client_id=validated_client_id,
        result_ttl_s=settings.result_ttl_s,
        now=clock(),
    )

    match outcome:
        case DeletableNow():
            await redis_client.delete(analysis_key(str(analysis_id)))
            return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
        case StillInProgress():
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                media_type="application/problem+json",
                content={
                    "type": "about:blank",
                    "title": "analysis is still queued or processing",
                    "status": status.HTTP_409_CONFLICT,
                    "code": "CONFLICT",
                },
            )
        case Gone() | NotFound():
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=None)
