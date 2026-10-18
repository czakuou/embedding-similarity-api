from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

import pytest
import redis.asyncio as redis
from arq.constants import in_progress_key_prefix, retry_key_prefix
from fastapi.testclient import TestClient
from starlette import status

from similarity.config import WorkerSettings
from similarity.encoder import SentenceTransformerEncoder

from .conftest import get_record, process_queue, submit_valid_document

pytestmark = pytest.mark.behavior


async def test_given_crashed_worker_when_job_marker_expires_then_picked_up_and_completed(
    app_client: TestClient,
    worker_redis: redis.Redis,
    worker_settings: WorkerSettings,
    real_encoder: SentenceTransformerEncoder,
) -> None:
    response = submit_valid_document(app_client, "client-a")
    assert response.status_code == status.HTTP_202_ACCEPTED, response.text
    analysis_id = response.json()["id"]

    # What a worker that started the job and was then killed leaves behind: arq's
    # in-progress marker (it expires on its own, here after 1 s instead of timeout + 10 s),
    # the try counter at 1, and a record that says `processing`. The job stays in the queue.
    await worker_redis.set(in_progress_key_prefix + analysis_id, b"1", px=1000)
    await worker_redis.set(retry_key_prefix + analysis_id, 1)
    # redis-py's stub types HSET as `Awaitable[int] | int` (see tests/behavior/conftest.py).
    record_update = worker_redis.hset(
        f"analysis:{analysis_id}", mapping={"status": "processing", "attempts": 1}
    )
    await cast("Awaitable[int]", record_update)

    await asyncio.sleep(1.1)  # the marker has expired: the work is considered abandoned
    await process_queue(worker_settings, real_encoder)

    record = await get_record(worker_redis, analysis_id)
    assert record.status.value == "done"
    assert record.attempts == 2
