"""Shared kernel: the API <-> worker contract.

Record schema, Redis key names, status transitions and UUIDv7/expiry math live
here because both processes depend on them. Nothing else is shared between
slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal
from uuid import UUID


class Status(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class WorkerPicked:
    """A worker read the message and is about to compute (attempts already incremented)."""


@dataclass(frozen=True, slots=True)
class ComputeSucceeded:
    result: list[float]
    document_truncated: bool


@dataclass(frozen=True, slots=True)
class ComputeFailed:
    """A single item's computation raised; attempts has already been incremented."""


@dataclass(frozen=True, slots=True)
class AttemptsExhausted:
    """Message was reclaimed and attempts > MAX_ATTEMPTS: give up without computing."""


Event = WorkerPicked | ComputeSucceeded | ComputeFailed | AttemptsExhausted


@dataclass(frozen=True, slots=True)
class IllegalTransition:
    status: Status
    event: Event


def transition(status: Status, event: Event) -> Status | IllegalTransition:
    """Pure state machine from SPEC §5. `failed` and `done` are terminal."""
    match (status, event):
        case (Status.QUEUED, WorkerPicked()):
            return Status.PROCESSING
        case (Status.PROCESSING, ComputeSucceeded()):
            return Status.DONE
        case (Status.PROCESSING, ComputeFailed()):
            # Stays `processing`: the message is unacked and will be reclaimed later.
            return Status.PROCESSING
        case (Status.PROCESSING, AttemptsExhausted()):
            return Status.FAILED
        case _:
            return IllegalTransition(status=status, event=event)


# --- Redis key names (the cross-process naming contract) -------------------


def analysis_key(analysis_id: str) -> str:
    return f"analysis:{analysis_id}"


# --- Record schema -----------------------------------------------------------

ErrorCode = Literal["PROCESSING_FAILED"]


@dataclass(frozen=True, slots=True)
class AnalysisError:
    code: ErrorCode
    message: str
    attempts: int


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    id: str
    client_id: str
    status: Status
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    updated_at: datetime | None = None
    result: list[float] | None = None
    document_truncated: bool | None = None
    error: AnalysisError | None = None


# --- UUIDv7 / expiry math ----------------------------------------------------

_UUID7_TIMESTAMP_SHIFT = 80
_UUID7_TIMESTAMP_MASK = 0xFFFFFFFFFFFF


def uuid7_timestamp(analysis_id: UUID) -> datetime:
    """Extract the millisecond creation timestamp embedded in a UUIDv7's top 48 bits."""
    ms_since_epoch = (analysis_id.int >> _UUID7_TIMESTAMP_SHIFT) & _UUID7_TIMESTAMP_MASK
    return datetime.fromtimestamp(ms_since_epoch / 1000, tz=UTC)


def expires_at(created_at: datetime, result_ttl_s: int) -> datetime:
    return created_at + timedelta(seconds=result_ttl_s)


GoneOrNotFound = Literal["gone", "not_found"]


def missing_record_outcome(analysis_id: UUID, result_ttl_s: int, now: datetime) -> GoneOrNotFound:
    """Decide 410 vs 404 for a record that no longer exists in Redis (SPEC §4.2).

    No tombstones are stored, so the only evidence left is the UUIDv7 timestamp
    embedded in the id itself.
    """
    created_at = uuid7_timestamp(analysis_id)
    if created_at + timedelta(seconds=result_ttl_s) < now:
        return "gone"
    return "not_found"
