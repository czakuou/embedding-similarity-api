from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from similarity.analysis import (
    analysis_key,
    expires_at,
    missing_record_outcome,
    uuid7_timestamp,
)

# A UUIDv7 whose top 48 bits encode 2026-01-01T00:00:00.000Z, built and verified
# by round-tripping through uuid7_timestamp() during implementation (not hand-guessed hex).
FIXED_UUID7 = UUID("019b76da-a800-7000-8000-000000000000")
FIXED_UUID7_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_analysis_key_uses_colon_separated_namespace() -> None:
    assert analysis_key("abc-123") == "analysis:abc-123"


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
