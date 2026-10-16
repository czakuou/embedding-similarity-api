from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from similarity.analysis import AnalysisRecord, Status, expires_at
from similarity.results import (
    DeletableNow,
    Found,
    Gone,
    NotFound,
    StillInProgress,
    decide_delete_outcome,
    decide_get_outcome,
)

FIXED_UUID7 = UUID("019b76da-a800-7000-8000-000000000000")
CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _record(status: Status, client_id: str = "client-a") -> AnalysisRecord:
    return AnalysisRecord(
        id=str(FIXED_UUID7),
        client_id=client_id,
        status=status,
        created_at=CREATED_AT,
        expires_at=expires_at(CREATED_AT, 86_400),
    )


def test_given_owned_record_when_getting_then_found() -> None:
    record = _record(Status.DONE)
    outcome = decide_get_outcome(
        record, analysis_id=FIXED_UUID7, client_id="client-a", result_ttl_s=86_400, now=CREATED_AT
    )
    assert outcome == Found(record=record)


def test_given_record_owned_by_another_client_when_getting_then_not_found() -> None:
    record = _record(Status.DONE, client_id="client-b")
    outcome = decide_get_outcome(
        record, analysis_id=FIXED_UUID7, client_id="client-a", result_ttl_s=86_400, now=CREATED_AT
    )
    assert outcome == NotFound()


def test_given_missing_record_within_ttl_when_getting_then_not_found() -> None:
    outcome = decide_get_outcome(
        None,
        analysis_id=FIXED_UUID7,
        client_id="client-a",
        result_ttl_s=86_400,
        now=CREATED_AT + timedelta(hours=1),
    )
    assert outcome == NotFound()


def test_given_missing_record_past_ttl_when_getting_then_gone() -> None:
    outcome = decide_get_outcome(
        None,
        analysis_id=FIXED_UUID7,
        client_id="client-a",
        result_ttl_s=86_400,
        now=CREATED_AT + timedelta(hours=24, seconds=1),
    )
    assert outcome == Gone()


def test_given_done_record_owned_by_client_when_deleting_then_deletable() -> None:
    record = _record(Status.DONE)
    outcome = decide_delete_outcome(
        record, analysis_id=FIXED_UUID7, client_id="client-a", result_ttl_s=86_400, now=CREATED_AT
    )
    assert outcome == DeletableNow(record=record)


def test_given_failed_record_owned_by_client_when_deleting_then_deletable() -> None:
    record = _record(Status.FAILED)
    outcome = decide_delete_outcome(
        record, analysis_id=FIXED_UUID7, client_id="client-a", result_ttl_s=86_400, now=CREATED_AT
    )
    assert outcome == DeletableNow(record=record)


def test_given_queued_or_processing_record_when_deleting_then_still_in_progress() -> None:
    for status in (Status.QUEUED, Status.PROCESSING):
        record = _record(status)
        outcome = decide_delete_outcome(
            record,
            analysis_id=FIXED_UUID7,
            client_id="client-a",
            result_ttl_s=86_400,
            now=CREATED_AT,
        )
        assert outcome == StillInProgress()


def test_given_record_owned_by_another_client_when_deleting_then_not_found() -> None:
    record = _record(Status.DONE, client_id="client-b")
    outcome = decide_delete_outcome(
        record, analysis_id=FIXED_UUID7, client_id="client-a", result_ttl_s=86_400, now=CREATED_AT
    )
    assert outcome == NotFound()
