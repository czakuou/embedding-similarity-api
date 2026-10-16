"""Sum type for the POST decision and its mapping to an HTTP response."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from fastapi.responses import JSONResponse

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
    "unsupported_media_type": 415,
    "payload_too_large": 413,
    "validation_error": 422,
}
_INVALID_CODES: dict[InvalidReason, str] = {
    "unsupported_media_type": "UNSUPPORTED_MEDIA_TYPE",
    "payload_too_large": "PAYLOAD_TOO_LARGE",
    "validation_error": "VALIDATION_ERROR",
}


def outcome_to_response(outcome: Outcome) -> JSONResponse:
    match outcome:
        case Accepted(analysis_id=analysis_id, created_at=created_at, expires_at=expiry):
            location = f"/v1/similarity-analyses/{analysis_id}"
            return JSONResponse(
                status_code=202,
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
                status_code=429,
                headers={"Retry-After": str(retry_after_s)},
                media_type="application/problem+json",
                content={
                    "type": "about:blank",
                    "title": "Rate limited",
                    "status": 429,
                    "code": "RATE_LIMITED",
                },
            )
        case QueueFull(retry_after_s=retry_after_s):
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": str(retry_after_s)},
                media_type="application/problem+json",
                content={
                    "type": "about:blank",
                    "title": "Service overloaded",
                    "status": 503,
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
