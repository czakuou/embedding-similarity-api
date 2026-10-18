from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, cast

import httpx
import pytest
import redis.asyncio as redis
from arq.worker import Worker
from fastapi import FastAPI
from fastapi.testclient import TestClient
from testcontainers.community.redis import RedisContainer

from similarity.analysis import AnalysisRecord, analysis_key, hash_to_record
from similarity.api import create_app
from similarity.config import ApiSettings, WorkerSettings
from similarity.encoder import Encoder, SentenceTransformerEncoder
from similarity.worker.app import worker_kwargs

ApiSettingsFactory = Callable[..., ApiSettings]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@pytest.fixture(scope="session")
def real_encoder() -> SentenceTransformerEncoder:
    """The actual model, loaded once for the whole test session: these
    scenarios run against the real model, not a fake, and loading it per
    test would cost several seconds each.
    """
    encoder = SentenceTransformerEncoder(MODEL_NAME, MODEL_REVISION)
    encoder.warm_up()
    return encoder


class FailingEncoderWrapper:
    """Wraps the real encoder but simulates an infrastructure-level failure
    (e.g. a request to the model that never completes) for exactly one test
    analysis. The Encoder Protocol only ever sees text, not analysis_id, so
    the test identifies "the analysis that will fail" by giving it a unique
    document text known only to the test -- not because that text is
    special to the model (it isn't: the real model has no notion of a
    "poisoned" input). This is a business scenario about failure handling,
    not about any particular document content.
    """

    def __init__(self, inner: Encoder, failing_document_text: str) -> None:
        self._inner = inner
        self._failing_document_text = failing_document_text

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._failing_document_text in texts:
            raise RuntimeError("simulated infrastructure failure")
        return self._inner.encode(texts)

    def similarity_pairwise(
        self, embeddings1: list[list[float]], embeddings2: list[list[float]]
    ) -> list[float]:
        return self._inner.similarity_pairwise(embeddings1, embeddings2)


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
    every scenario must start from a clean slate via FLUSHDB -- otherwise a
    previous scenario's queue depth or rate-limit counters leak into the
    next one.
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
    so rate-limit/queue-depth scenarios run in milliseconds instead of by
    racing real time windows. Any kwarg can be overridden per-test (e.g.
    max_queue_depth, max_file_size_bytes).
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
        queue_name=api_settings.queue_name,
        result_ttl_s=api_settings.result_ttl_s,
        max_attempts=3,
        visibility_timeout_s=1,
        batch_max_size=32,
        batch_linger_ms=50,
        poll_delay_s=0.05,
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


async def process_queue(
    worker_settings: WorkerSettings, encoder: Encoder, *, max_jobs: int = -1
) -> None:
    """Runs a real arq worker in burst mode: it exits once the queue is empty, including jobs
    deferred for a retry. `max_jobs` stops it after that many job starts instead, which lets
    a scenario look at the state between attempts. Real Redis, real arq polling; only the
    process boundary and signal handling are left out.
    """
    worker = Worker(
        **worker_kwargs(worker_settings, lambda: encoder, serve_metrics=False),
        burst=True,
        handle_signals=False,
        max_burst_jobs=max_jobs,
    )
    try:
        await worker.async_run()
    finally:
        await worker.close()
