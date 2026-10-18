from __future__ import annotations

from similarity.worker.items import (
    QueueItem,
    is_last_attempt_failure,
    parse_xautoclaim_response,
    parse_xreadgroup_response,
    should_give_up,
)


def test_given_live_xreadgroup_response_shape_when_parsing_then_items_extracted() -> None:
    # Shape verified against a live Redis during implementation: [[stream, [(id, fields)]]]
    response = [
        (
            "analyses:jobs",
            [
                (
                    "123-0",
                    {
                        "analysis_id": "a1",
                        "client_id": "client-a",
                        "document": "doc",
                        "query": "q",
                        "document_truncated": "0",
                    },
                )
            ],
        )
    ]
    assert parse_xreadgroup_response(response) == [
        QueueItem(
            msg_id="123-0",
            analysis_id="a1",
            client_id="client-a",
            document="doc",
            query="q",
            document_truncated=False,
        )
    ]


def test_given_empty_xreadgroup_response_when_parsing_then_no_items() -> None:
    assert parse_xreadgroup_response([]) == []


def test_given_live_xautoclaim_response_shape_when_parsing_then_items_extracted() -> None:
    # Shape verified against a live Redis: (next_cursor, [(id, fields)], deleted_ids)
    response: tuple[str, list[tuple[str, dict[str, str]]], list[str]] = (
        "0-0",
        [
            (
                "456-0",
                {
                    "analysis_id": "a2",
                    "client_id": "client-b",
                    "document": "doc2",
                    "query": "q2",
                    "document_truncated": "1",
                },
            )
        ],
        [],
    )
    assert parse_xautoclaim_response(response) == [
        QueueItem(
            msg_id="456-0",
            analysis_id="a2",
            client_id="client-b",
            document="doc2",
            query="q2",
            document_truncated=True,
        )
    ]


def test_given_attempts_within_limit_when_checking_give_up_then_false() -> None:
    assert should_give_up(attempts=1, max_attempts=3) is False
    assert should_give_up(attempts=3, max_attempts=3) is False


def test_given_attempts_beyond_limit_when_checking_give_up_then_true() -> None:
    assert should_give_up(attempts=4, max_attempts=3) is True


def test_given_attempts_below_max_when_checking_last_attempt_failure_then_false() -> None:
    assert is_last_attempt_failure(attempts=1, max_attempts=3) is False
    assert is_last_attempt_failure(attempts=2, max_attempts=3) is False


def test_given_attempts_at_max_when_checking_last_attempt_failure_then_true() -> None:
    # The 3rd failed attempt (not a 4th) is the one that gives up, so a failed
    # analysis ends with attempts == MAX_ATTEMPTS.
    assert is_last_attempt_failure(attempts=3, max_attempts=3) is True
