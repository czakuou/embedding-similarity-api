from __future__ import annotations

import asyncio

import pytest
import redis.asyncio as redis
from fastapi.testclient import TestClient
from starlette import status

from similarity.config import ApiSettings, WorkerSettings

from .conftest import ApiSettingsFactory, get_record, process_one_batch

pytestmark = pytest.mark.behavior


@pytest.fixture
def api_settings(api_settings_factory: ApiSettingsFactory) -> ApiSettings:
    # This scenario queues 4 analyses; the default max_queue_depth=3 (shared
    # across scenarios) would reject the 4th before the worker ever sees it.
    return api_settings_factory(max_queue_depth=10)


async def test_given_poisoned_document_in_batch_when_processed_then_only_it_fails_with_reason(
    app_client: TestClient,
    worker_redis: redis.Redis,
    worker_settings: WorkerSettings,
) -> None:
    # worker_settings fixture sets visibility_timeout_s=1 and max_attempts=3.
    healthy_ids = []
    for i in range(3):
        response = app_client.post(
            "/v1/similarity-analyses",
            headers={"X-Client-Id": f"client-{i}"},
            files={"file": ("fox.txt", b"a fox document", "text/plain")},
            data={"query": "fox over dog"},
        )
        assert response.status_code == status.HTTP_202_ACCEPTED, response.text
        healthy_ids.append(response.json()["id"])

    poison_response = app_client.post(
        "/v1/similarity-analyses",
        headers={"X-Client-Id": "client-poison"},
        files={"file": ("poison.txt", b"contains __POISON__ marker", "text/plain")},
        data={"query": "fox over dog"},
    )
    assert poison_response.status_code == status.HTTP_202_ACCEPTED, poison_response.text
    poison_id = poison_response.json()["id"]

    await process_one_batch(worker_redis, worker_settings)

    for i, analysis_id in enumerate(healthy_ids):
        poll = app_client.get(
            f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": f"client-{i}"}
        )
        body = poll.json()
        assert body["status"] == "done"
        assert body["result"] is not None

    poisoned = await get_record(worker_redis, poison_id)
    assert poisoned.status.value == "processing"
    assert poisoned.attempts == 1

    # Reclaim and retry until attempts are exhausted (max_attempts=3): each
    # retry needs the real visibility_timeout_s (1s) to actually elapse, since
    # XAUTOCLAIM's min_idle_time is measured by Redis's own clock.
    for _ in range(2):
        await asyncio.sleep(1.1)
        await process_one_batch(worker_redis, worker_settings, force_reclaim=True)

    failed = await get_record(worker_redis, poison_id)
    assert failed.status.value == "failed"
    assert failed.error is not None
    assert failed.error.code == "PROCESSING_FAILED"
    assert failed.error.attempts == 3
    assert "Traceback" not in failed.error.message

    # "the message is no longer in the queue"
    assert await worker_redis.xlen(worker_settings.stream_key) == 0
