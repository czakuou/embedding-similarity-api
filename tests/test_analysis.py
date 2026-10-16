from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from similarity.analysis import (
    AnalysisError,
    AnalysisRecord,
    Status,
    analysis_key,
    expires_at,
    hash_to_record,
    missing_record_outcome,
    record_to_hash,
    uuid7_timestamp,
    validate_client_id,
)

# A UUIDv7 whose top 48 bits encode 2026-01-01T00:00:00.000Z, built and verified
# by round-tripping through uuid7_timestamp() during implementation (not hand-guessed hex).
FIXED_UUID7 = UUID("019b76da-a800-7000-8000-000000000000")
FIXED_UUID7_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_analysis_key_uses_colon_separated_namespace() -> None:
    assert analysis_key("abc-123") == "analysis:abc-123"


def test_given_valid_client_id_when_validating_then_returned_unchanged() -> None:
    assert validate_client_id("client-a_1") == "client-a_1"


def test_given_missing_or_malformed_client_id_when_validating_then_rejected() -> None:
    assert validate_client_id(None) is None
    assert validate_client_id("") is None
    assert validate_client_id("has a space") is None
    assert validate_client_id("a" * 65) is None


def test_uuid7_timestamp_extracts_embedded_creation_time() -> None:
    extracted = uuid7_timestamp(FIXED_UUID7)
    assert extracted == FIXED_UUID7_CREATED_AT


def test_expires_at_adds_ttl_seconds_to_created_at() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assert expires_at(created_at, result_ttl_s=86_400) == datetime(2026, 1, 2, tzinfo=UTC)


def test_given_record_missing_and_ttl_elapsed_when_checking_outcome_then_gone() -> None:
    now = FIXED_UUID7_CREATED_AT + timedelta(hours=24, seconds=1)
    assert missing_record_outcome(FIXED_UUID7, result_ttl_s=86_400, now=now) == "gone"


def test_given_record_missing_and_ttl_not_elapsed_when_checking_outcome_then_not_found() -> None:
    now = FIXED_UUID7_CREATED_AT + timedelta(hours=1)
    assert missing_record_outcome(FIXED_UUID7, result_ttl_s=86_400, now=now) == "not_found"


def test_given_record_missing_at_exact_ttl_boundary_when_checking_outcome_then_not_found() -> None:
    # SPEC: "id_time + RESULT_TTL_S < now" -> strictly less than, so equality is still "not_found".
    now = FIXED_UUID7_CREATED_AT + timedelta(hours=24)
    assert missing_record_outcome(FIXED_UUID7, result_ttl_s=86_400, now=now) == "not_found"


def test_given_minimal_queued_record_when_round_tripped_through_hash_then_equal() -> None:
    record = AnalysisRecord(
        id="abc-123",
        client_id="client-a",
        status=Status.QUEUED,
        created_at=FIXED_UUID7_CREATED_AT,
        expires_at=expires_at(FIXED_UUID7_CREATED_AT, 86_400),
    )
    assert hash_to_record(record_to_hash(record)) == record


def test_given_done_record_with_result_when_round_tripped_through_hash_then_equal() -> None:
    record = AnalysisRecord(
        id="abc-123",
        client_id="client-a",
        status=Status.DONE,
        created_at=FIXED_UUID7_CREATED_AT,
        expires_at=expires_at(FIXED_UUID7_CREATED_AT, 86_400),
        attempts=1,
        updated_at=FIXED_UUID7_CREATED_AT + timedelta(seconds=1),
        result=[0.87],
        document_truncated=False,
    )
    assert hash_to_record(record_to_hash(record)) == record


def test_given_failed_record_with_error_when_round_tripped_through_hash_then_equal() -> None:
    record = AnalysisRecord(
        id="abc-123",
        client_id="client-a",
        status=Status.FAILED,
        created_at=FIXED_UUID7_CREATED_AT,
        expires_at=expires_at(FIXED_UUID7_CREATED_AT, 86_400),
        attempts=3,
        error=AnalysisError(code="PROCESSING_FAILED", message="boom", attempts=3),
    )
    assert hash_to_record(record_to_hash(record)) == record
