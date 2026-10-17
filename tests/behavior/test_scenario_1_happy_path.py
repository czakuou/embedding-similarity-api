from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import redis.asyncio as redis
from fastapi.testclient import TestClient
from starlette import status

from similarity.config import WorkerSettings

from .conftest import process_one_batch

pytestmark = pytest.mark.behavior


async def test_given_valid_document_when_submitted_and_processed_then_result_is_single_float(
    app_client: TestClient,
    worker_redis: redis.Redis,
    worker_settings: WorkerSettings,
) -> None:
    response = app_client.post(
        "/v1/similarity-analyses",
        headers={"X-Client-Id": "client-a"},
        files={"file": ("fox.txt", b"The quick brown fox jumps over the lazy dog.", "text/plain")},
        data={"query": "fox over dog"},
    )

    assert response.status_code == status.HTTP_202_ACCEPTED, response.text
    body = response.json()
    assert response.headers["Location"] == f"/v1/similarity-analyses/{body['id']}"
    assert "Retry-After" in response.headers
    assert body["status"] == "queued"
    created_at = datetime.fromisoformat(body["created_at"])
    expires_at = datetime.fromisoformat(body["expires_at"])
    assert expires_at - created_at == timedelta(hours=24)

    await process_one_batch(worker_redis, worker_settings)

    poll = app_client.get(body["links"]["self"], headers={"X-Client-Id": "client-a"})
    assert poll.status_code == status.HTTP_200_OK, poll.text
    result_body = poll.json()
    assert result_body["status"] == "done"
    assert len(result_body["result"]) == 1
    assert -1.0 <= result_body["result"][0] <= 1.0
    assert result_body["document_truncated"] is False
