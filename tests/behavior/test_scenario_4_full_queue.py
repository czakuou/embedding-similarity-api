from __future__ import annotations

import pytest
import redis.asyncio as redis
from fastapi.testclient import TestClient
from starlette import status

from similarity.config import WorkerSettings

from .conftest import process_one_batch, submit_valid_document

pytestmark = pytest.mark.behavior


async def test_given_full_queue_when_submitting_then_503_and_accepted_work_completes(
    app_client: TestClient,
    worker_redis: redis.Redis,
    worker_settings: WorkerSettings,
) -> None:
    # api_settings fixture sets max_queue_depth=3; different client ids avoid
    # tripping the (separately tested) per-client rate limit.
    accepted_ids = []
    for i in range(3):
        response = submit_valid_document(app_client, f"filler-{i}")
        assert response.status_code == status.HTTP_202_ACCEPTED, response.text
        accepted_ids.append(response.json()["id"])

    rejected = submit_valid_document(app_client, "client-overflow")
    assert rejected.status_code == status.HTTP_503_SERVICE_UNAVAILABLE, rejected.text
    retry_after = int(rejected.headers["Retry-After"])
    assert 5 <= retry_after <= 10
    assert rejected.json()["code"] == "OVERLOADED"

    await process_one_batch(worker_redis, worker_settings)

    for i, analysis_id in enumerate(accepted_ids):
        poll = app_client.get(
            f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": f"filler-{i}"}
        )
        assert poll.json()["status"] == "done"

    new_submission = submit_valid_document(app_client, "client-after-drain")
    assert new_submission.status_code == status.HTTP_202_ACCEPTED, new_submission.text
