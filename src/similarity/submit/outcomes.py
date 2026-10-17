"""Sum type for the POST decision and its mapping to an HTTP response."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from fastapi.responses import JSONResponse
from starlette import status

from similarity.api_paths import analysis_url

InvalidReason = Literal[
    "unsupported_media_type",
    "payload_too_large",
    "validation_error",
]


@dataclass(frozen=True, slots=True)
class Accepted:
    analysis_id: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class RateLimited:
    retry_after_s: int


@dataclass(frozen=True, slots=True)
class QueueFull:
    retry_after_s: int


@dataclass(frozen=True, slots=True)
class Invalid:
    reason: InvalidReason
    detail: str


Outcome = Accepted | RateLimited | QueueFull | Invalid

_INVALID_STATUS_CODES: dict[InvalidReason, int] = {
    "unsupported_media_type": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    "payload_too_large": status.HTTP_413_CONTENT_TOO_LARGE,
    "validation_error": status.HTTP_422_UNPROCESSABLE_CONTENT,
}
_INVALID_CODES: dict[InvalidReason, str] = {
    "unsupported_media_type": "UNSUPPORTED_MEDIA_TYPE",
    "payload_too_large": "PAYLOAD_TOO_LARGE",
    "validation_error": "VALIDATION_ERROR",
}


def outcome_to_response(outcome: Outcome) -> JSONResponse:
    match outcome:
        case Accepted(analysis_id=analysis_id, created_at=created_at, expires_at=expiry):
            location = analysis_url(analysis_id)
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                headers={"Location": location, "Retry-After": "1"},
                content={
                    "id": analysis_id,
                    "status": "queued",
                    "created_at": created_at.isoformat(),
                    "expires_at": expiry.isoformat(),
                    "links": {"self": location},
                },
            )
        case RateLimited(retry_after_s=retry_after_s):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(retry_after_s)},
                media_type="application/problem+json",
                content={
                    "type": "about:blank",
                    "title": "Rate limited",
                    "status": status.HTTP_429_TOO_MANY_REQUESTS,
                    "code": "RATE_LIMITED",
                },
            )
        case QueueFull(retry_after_s=retry_after_s):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": str(retry_after_s)},
                media_type="application/problem+json",
                content={
                    "type": "about:blank",
                    "title": "Service overloaded",
                    "status": status.HTTP_503_SERVICE_UNAVAILABLE,
                    "code": "OVERLOADED",
                },
            )
        case Invalid(reason=reason, detail=detail):
            return JSONResponse(
                status_code=_INVALID_STATUS_CODES[reason],
                media_type="application/problem+json",
                content={
                    "type": "about:blank",
                    "title": detail,
                    "status": _INVALID_STATUS_CODES[reason],
                    "code": _INVALID_CODES[reason],
                },
            )
