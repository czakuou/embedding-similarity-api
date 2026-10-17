from __future__ import annotations

import pytest

from similarity.config import ApiSettings, WorkerSettings


def test_given_doc_chars_above_file_size_when_constructing_api_settings_then_raises() -> None:
    with pytest.raises(ValueError, match="MAX_DOC_CHARS"):
        ApiSettings(max_doc_chars=200_000, max_file_size_bytes=102_400)


def test_given_heartbeat_too_short_when_constructing_worker_settings_then_raises() -> None:
    with pytest.raises(ValueError, match="HEARTBEAT_MAX_AGE_S"):
        WorkerSettings(heartbeat_max_age_s=1, read_block_ms=5_000)


def test_given_zero_max_attempts_when_constructing_worker_settings_then_raises() -> None:
    with pytest.raises(ValueError, match="MAX_ATTEMPTS"):
        WorkerSettings(max_attempts=0)
