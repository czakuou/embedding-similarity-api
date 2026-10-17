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
    """SPEC §5: a message reclaimed with attempts > MAX_ATTEMPTS fails without computing
    (poison circuit breaker). Attempts is incremented before this check runs.
    """
    return attempts > max_attempts
