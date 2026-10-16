"""Shared kernel: the API <-> worker contract.

Record schema, Redis key names, status transitions and UUIDv7/expiry math live
here because both processes depend on them. Nothing else is shared between
slices.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal
from uuid import UUID

CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_client_id(client_id: str | None) -> str | None:
    """Returns the client id if valid, None otherwise. Shared by submit.py and
    results.py: both need the same `X-Client-Id` format contract (SPEC §4.1).
    """
    if client_id is None:
        return None
    if not CLIENT_ID_PATTERN.match(client_id):
        return None
    return client_id


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


def record_to_hash(record: AnalysisRecord) -> dict[str, str]:
    """Redis hashes only store strings; this is the single place that encodes an
    AnalysisRecord into one, so the API and worker never drift on field encoding.
    """
    fields: dict[str, str] = {
        "id": record.id,
        "client_id": record.client_id,
        "status": record.status.value,
        "created_at": record.created_at.isoformat(),
        "expires_at": record.expires_at.isoformat(),
        "attempts": str(record.attempts),
    }
    if record.updated_at is not None:
        fields["updated_at"] = record.updated_at.isoformat()
    if record.result is not None:
        fields["result"] = json.dumps(record.result)
    if record.document_truncated is not None:
        fields["document_truncated"] = str(int(record.document_truncated))
    if record.error is not None:
        fields["error"] = json.dumps(
            {
                "code": record.error.code,
                "message": record.error.message,
                "attempts": record.error.attempts,
            }
        )
    return fields


def hash_to_record(fields: dict[str, str]) -> AnalysisRecord:
    """Inverse of record_to_hash; raises KeyError/ValueError on a malformed hash,
    which should never happen since both writers go through record_to_hash.
    """
    error: AnalysisError | None = None
    if "error" in fields:
        raw_error = json.loads(fields["error"])
        error = AnalysisError(**raw_error)

    return AnalysisRecord(
        id=fields["id"],
        client_id=fields["client_id"],
        status=Status(fields["status"]),
        created_at=datetime.fromisoformat(fields["created_at"]),
        expires_at=datetime.fromisoformat(fields["expires_at"]),
        attempts=int(fields["attempts"]),
        updated_at=datetime.fromisoformat(fields["updated_at"]) if "updated_at" in fields else None,
        result=json.loads(fields["result"]) if "result" in fields else None,
        document_truncated=bool(int(fields["document_truncated"]))
        if "document_truncated" in fields
        else None,
        error=error,
    )


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
