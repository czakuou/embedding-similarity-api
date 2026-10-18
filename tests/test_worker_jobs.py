from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
import redis.asyncio as redis
from arq import Retry

from similarity.analysis import AnalysisRecord, Status, analysis_key, expires_at
from similarity.config import WorkerSettings
from similarity.worker.batcher import Batcher
from similarity.worker.jobs import analyze, fetch_record, write_record

from .fake_encoder import POISON_MARKER, FakeEncoder


class CountingEncoder(FakeEncoder):
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encode_calls += 1
        return super().encode(texts)


@pytest.fixture
async def ctx(redis_client: redis.Redis) -> AsyncIterator[dict[str, Any]]:
    encoder = CountingEncoder()
    batcher = Batcher(encoder, max_size=8, linger_s=0)
    task = asyncio.create_task(batcher.run())
    yield {
        "settings": WorkerSettings(max_attempts=3, visibility_timeout_s=1),
        "records": redis_client,
        "batcher": batcher,
        "encoder": encoder,
        "job_try": 1,
    }
    task.cancel()


async def _queued_record(records: redis.Redis, analysis_id: str = "a1") -> AnalysisRecord:
    created_at = datetime.now(UTC)
    record = AnalysisRecord(
        id=analysis_id,
        client_id="client-a",
        status=Status.QUEUED,
        created_at=created_at,
        expires_at=expires_at(created_at, 86_400),
    )
    await write_record(records, record)
    return record


async def test_given_queued_analysis_when_job_runs_then_record_is_done_with_one_score(
    ctx: dict[str, Any],
) -> None:
    await _queued_record(ctx["records"])

    await analyze(ctx, "a1", "the quick brown fox", "fox")

    record = await fetch_record(ctx["records"], "a1")
    assert record is not None
    assert record.status is Status.DONE
    assert record.result is not None and len(record.result) == 1
    assert record.attempts == 1


async def test_given_record_deleted_while_queued_when_job_runs_then_nothing_is_written_back(
    ctx: dict[str, Any],
) -> None:
    await analyze(ctx, "never-existed", "the quick brown fox", "fox")

    assert not await ctx["records"].exists(analysis_key("never-existed"))
    assert ctx["encoder"].encode_calls == 0


async def test_given_attempts_already_exhausted_when_job_runs_then_failed_without_computing(
    ctx: dict[str, Any],
) -> None:
    await _queued_record(ctx["records"])
    ctx["job_try"] = 4  # one past MAX_ATTEMPTS: redelivered after crashes

    await analyze(ctx, "a1", "the quick brown fox", "fox")

    record = await fetch_record(ctx["records"], "a1")
    assert record is not None
    assert record.status is Status.FAILED
    assert record.error is not None and record.error.attempts == 4
    assert ctx["encoder"].encode_calls == 0


async def test_given_failing_encoder_when_job_runs_then_retried_until_attempts_run_out(
    ctx: dict[str, Any],
) -> None:
    await _queued_record(ctx["records"])
    poisoned = f"{POISON_MARKER} the encoder rejects this"

    ctx["job_try"] = 1
    with pytest.raises(Retry):
        await analyze(ctx, "a1", poisoned, "fox")
    midway = await fetch_record(ctx["records"], "a1")
    assert midway is not None and midway.status is Status.PROCESSING

    ctx["job_try"] = 3  # the last allowed attempt
    await analyze(ctx, "a1", poisoned, "fox")
    record = await fetch_record(ctx["records"], "a1")
    assert record is not None
    assert record.status is Status.FAILED
    assert record.error is not None
    assert "Traceback" not in record.error.message
