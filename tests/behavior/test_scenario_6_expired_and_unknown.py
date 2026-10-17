from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette import status
from uuid6 import uuid7

from similarity.config import WorkerSettings
from similarity.dependencies import get_clock

from .conftest import process_one_batch, submit_valid_document

pytestmark = pytest.mark.behavior


async def test_given_expired_analysis_when_polled_then_410(
    app: FastAPI,
    app_client: TestClient,
    worker_redis: redis.Redis,
    worker_settings: WorkerSettings,
) -> None:
    response = submit_valid_document(app_client, "client-a")
    assert response.status_code == status.HTTP_202_ACCEPTED, response.text
    analysis_id = response.json()["id"]

    await process_one_batch(worker_redis, worker_settings)
    done = app_client.get(
        f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": "client-a"}
    )
    assert done.json()["status"] == "done"

    # The record's own TTL (RESULT_TTL_S) has elapsed in Redis -- the API's
    # 404-vs-410 decision then falls back to the UUIDv7 timestamp embedded in
    # analysis_id, compared against the injected clock (SPEC §3.5).
    await worker_redis.delete(f"analysis:{analysis_id}")
    real_now = get_clock()()

    def future_clock() -> datetime:
        return real_now + timedelta(hours=24, seconds=1)

    app.dependency_overrides[get_clock] = lambda: future_clock

    expired = app_client.get(
        f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": "client-a"}
    )
    assert expired.status_code == status.HTTP_410_GONE, expired.text

    app.dependency_overrides.clear()


def test_given_unknown_analysis_id_when_polled_then_404(app_client: TestClient) -> None:
    # A freshly minted (never submitted) UUIDv7: its embedded timestamp is
    # "now", well within RESULT_TTL_S, so missing_record_outcome must say
    # "never existed" (404), not "existed but expired" (410).
    unknown_id = uuid7()
    response = app_client.get(
        f"/v1/similarity-analyses/{unknown_id}", headers={"X-Client-Id": "client-a"}
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text


def test_given_foreign_analysis_when_polled_by_other_client_then_404(
    app_client: TestClient,
) -> None:
    response = submit_valid_document(app_client, "client-b")
    assert response.status_code == status.HTTP_202_ACCEPTED, response.text
    analysis_id = response.json()["id"]

    poll = app_client.get(
        f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": "client-a"}
    )
    assert poll.status_code == status.HTTP_404_NOT_FOUND, poll.text
