from __future__ import annotations

import asyncio

import pytest
import redis.asyncio as redis
from fastapi.testclient import TestClient
from starlette import status

from similarity.config import WorkerSettings
from similarity.worker.batch import increment_attempts
from similarity.worker.loop import ensure_consumer_group, read_batch

from .conftest import get_record, process_one_batch, submit_valid_document

pytestmark = pytest.mark.behavior


async def test_given_crashed_worker_when_message_idle_then_reclaimed_and_completed(
    app_client: TestClient,
    worker_redis: redis.Redis,
    worker_settings: WorkerSettings,
) -> None:
    # worker_settings fixture sets visibility_timeout_s=1 (BEHAVIOR.md).
    response = submit_valid_document(app_client, "client-a")
    assert response.status_code == status.HTTP_202_ACCEPTED, response.text
    analysis_id = response.json()["id"]

    # w1 reads the message (creating a pending-entry in the consumer group)
    # and increments attempts (SPEC §5: HINCRBY before computing) but crashes
    # before ever acking it -- simulated by stopping right there.
    await ensure_consumer_group(
        worker_redis, worker_settings.stream_key, worker_settings.consumer_group
    )
    crashed_items = await read_batch(
        worker_redis,
        stream_key=worker_settings.stream_key,
        consumer_group=worker_settings.consumer_group,
        consumer_name="w1",
        batch_max_size=worker_settings.batch_max_size,
        read_block_ms=worker_settings.read_block_ms,
    )
    await increment_attempts(worker_redis, crashed_items[0].analysis_id)

    await asyncio.sleep(1.1)  # real VISIBILITY_TIMEOUT_S=1 must actually elapse for XAUTOCLAIM

    # w2 reclaims the idle message and completes it.
    await process_one_batch(worker_redis, worker_settings, consumer_name="w2", force_reclaim=True)

    record = await get_record(worker_redis, analysis_id)
    assert record.status.value == "done"
    assert record.attempts == 2
