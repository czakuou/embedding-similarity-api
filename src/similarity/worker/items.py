"""Pure parsing and decisions for one queue item (a stream message)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueueItem:
    msg_id: str
    analysis_id: str
    client_id: str
    document: str
    query: str
    document_truncated: bool


def parse_queue_item(msg_id: str, fields: dict[str, str]) -> QueueItem:
    return QueueItem(
        msg_id=msg_id,
        analysis_id=fields["analysis_id"],
        client_id=fields["client_id"],
        document=fields["document"],
        query=fields["query"],
        document_truncated=bool(int(fields["document_truncated"])),
    )


def parse_xreadgroup_response(
    response: list[tuple[str, list[tuple[str, dict[str, str]]]]],
) -> list[QueueItem]:
    """XREADGROUP returns [[stream_name, [(msg_id, fields), ...]]] (verified against a
    live Redis during implementation, not assumed from docs)."""
    if not response:
        return []
    _stream_name, messages = response[0]
    return [parse_queue_item(msg_id, fields) for msg_id, fields in messages]


def parse_xautoclaim_response(
    response: tuple[str, list[tuple[str, dict[str, str]]], list[str]],
) -> list[QueueItem]:
    """XAUTOCLAIM returns (next_cursor, [(msg_id, fields), ...], deleted_ids)."""
    _next_cursor, messages, _deleted_ids = response
    return [parse_queue_item(msg_id, fields) for msg_id, fields in messages]


def should_give_up(attempts: int, max_attempts: int) -> bool:
    """Defensive check before even attempting computation: attempts has already
    exceeded MAX_ATTEMPTS, so this item must have been redelivered after it
    should already have failed. Normally is_last_attempt_failure catches this
    earlier (SPEC §5 poison circuit breaker); this is the fallback.
    """
    return attempts > max_attempts


def is_last_attempt_failure(attempts: int, max_attempts: int) -> bool:
    """BEHAVIOR.md scenario 5: a document failing on every one of MAX_ATTEMPTS
    attempts ends up `failed` with `attempts == MAX_ATTEMPTS` (not + 1). Called
    only after a computation attempt has actually failed, so the item still
    gets every one of its allotted attempts before giving up.
    """
    return attempts >= max_attempts
