from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, cast

import httpx
import pytest
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from testcontainers.community.redis import RedisContainer

from similarity.analysis import AnalysisRecord, analysis_key, hash_to_record
from similarity.api import create_app
from similarity.config import ApiSettings, WorkerSettings
from similarity.worker.items import QueueItem
from similarity.worker.loop import ensure_consumer_group, run_one_iteration

from ..fake_encoder import FakeEncoder

ApiSettingsFactory = Callable[..., ApiSettings]


def submit_valid_document(client: TestClient, client_id: str) -> httpx.Response:
    response: httpx.Response = client.post(
        "/v1/similarity-analyses",
        headers={"X-Client-Id": client_id},
        files={"file": ("fox.txt", b"a fox document", "text/plain")},
        data={"query": "fox over dog"},
    )
    return response


async def get_record(redis_client: redis.Redis, analysis_id: str) -> AnalysisRecord:
    # redis-py's stub shares one signature between the sync and async clients
    # (see similarity/worker/batch.py), so HGETALL needs a cast before awaiting.
    fields = await cast(
        "Awaitable[dict[str, str]]", redis_client.hgetall(analysis_key(analysis_id))
    )
    return hash_to_record(fields)


@pytest.fixture(autouse=True)
async def _flush_redis_between_tests(redis_container: RedisContainer) -> None:
    """redis_container is session-scoped (one container for the whole run), so
    every scenario must start from a clean slate (BEHAVIOR.md "FLUSHDB between
    tests") -- otherwise a previous scenario's queue depth or rate-limit
    counters leak into the next one.
    """
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(redis_container.port)
    client = redis.from_url(f"redis://{host}:{port}/0", decode_responses=True)
    try:
        await client.flushdb()
    finally:
        await client.aclose()


@pytest.fixture
def api_settings_factory(redis_container: RedisContainer) -> ApiSettingsFactory:
    """Builds ApiSettings pointed at the shared test Redis, with small overrides
    so rate-limit/queue-depth scenarios run in milliseconds (BEHAVIOR.md "How
    these scenarios run"), not by racing real time windows. Any kwarg can be
    overridden per-test (e.g. max_queue_depth, max_file_size_bytes).
    """
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(redis_container.port)

    def build(**overrides: Any) -> ApiSettings:
        defaults: dict[str, Any] = {
            "redis_url": f"redis://{host}:{port}/0",
            "rate_limit": "2/minute",
            "max_queue_depth": 3,
            "retry_after_base_s": 5,
        }
        return ApiSettings(**(defaults | overrides))

    return build


@pytest.fixture
def api_settings(api_settings_factory: ApiSettingsFactory) -> ApiSettings:
    return api_settings_factory()


@pytest.fixture
def worker_settings(api_settings: ApiSettings) -> WorkerSettings:
    return WorkerSettings(
        redis_url=api_settings.redis_url,
        stream_key=api_settings.stream_key,
        consumer_group=api_settings.consumer_group,
        result_ttl_s=api_settings.result_ttl_s,
        max_attempts=3,
        visibility_timeout_s=1,
        batch_max_size=32,
    )


@pytest.fixture
def app(api_settings: ApiSettings) -> FastAPI:
    return create_app(api_settings)


@pytest.fixture
def app_client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def worker_redis(worker_settings: WorkerSettings) -> AsyncIterator[redis.Redis]:
    client = redis.from_url(worker_settings.redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


async def process_one_batch(
    worker_redis: redis.Redis,
    worker_settings: WorkerSettings,
    *,
    consumer_name: str = "worker-test",
    force_reclaim: bool = False,
) -> list[QueueItem]:
    """Drives the worker exactly one iteration (BEHAVIOR.md: no background
    threads or sleeps) using the real consumer-group mechanics (XREADGROUP /
    XAUTOCLAIM) against the same Redis the API wrote to, with a fake encoder.
    """
    await ensure_consumer_group(
        worker_redis, worker_settings.stream_key, worker_settings.consumer_group
    )
    return await run_one_iteration(
        worker_redis,
        FakeEncoder(),
        worker_settings,
        consumer_name=consumer_name,
        force_reclaim=force_reclaim,
    )
