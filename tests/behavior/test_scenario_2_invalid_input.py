from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import redis.asyncio as redis
from fastapi.testclient import TestClient
from starlette import status

from similarity.api import create_app
from similarity.config import ApiSettings

from .conftest import ApiSettingsFactory

pytestmark = pytest.mark.behavior


@pytest.fixture
def small_file_settings(api_settings_factory: ApiSettingsFactory) -> ApiSettings:
    """A small MAX_FILE_SIZE_BYTES so the "file larger than the limit" example
    doesn't require constructing a 100KB+ upload.
    """
    return api_settings_factory(max_file_size_bytes=16, max_doc_chars=16)


@pytest.fixture
def small_file_client(small_file_settings: ApiSettings) -> Iterator[TestClient]:
    app = create_app(small_file_settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def small_file_redis(small_file_settings: ApiSettings) -> AsyncIterator[redis.Redis]:
    client = redis.from_url(small_file_settings.redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("filename", "content", "query", "expected_status", "expected_code"),
    [
        pytest.param(
            "data.csv",
            b"fox over dog",
            "fox over dog",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "UNSUPPORTED_MEDIA_TYPE",
            id="wrong-extension",
        ),
        pytest.param(
            "doc.txt",
            b"x" * 100,
            "fox over dog",
            status.HTTP_413_CONTENT_TOO_LARGE,
            "PAYLOAD_TOO_LARGE",
            id="too-large",
        ),
        pytest.param(
            "doc.txt",
            b"\xff\xfe invalid utf8",
            "fox over dog",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "VALIDATION_ERROR",
            id="bad-utf8",
        ),
        pytest.param(
            "doc.txt",
            b"",
            "fox over dog",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "VALIDATION_ERROR",
            id="empty-file",
        ),
        pytest.param(
            "doc.txt",
            b"a fox document",
            "",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "VALIDATION_ERROR",
            id="empty-query",
        ),
        pytest.param(
            "doc.txt",
            b"a fox document",
            "x" * 1_001,
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "VALIDATION_ERROR",
            id="query-too-long",
        ),
    ],
)
async def test_given_invalid_input_when_submitted_then_rejected_and_not_queued(
    small_file_client: TestClient,
    small_file_settings: ApiSettings,
    small_file_redis: redis.Redis,
    filename: str,
    content: bytes,
    query: str,
    expected_status: int,
    expected_code: str,
) -> None:
    depth_before = await small_file_redis.zcard(small_file_settings.queue_name)

    response = small_file_client.post(
        "/v1/similarity-analyses",
        headers={"X-Client-Id": "client-a"},
        files={"file": (filename, content, "text/plain")},
        data={"query": query},
    )

    assert response.status_code == expected_status, response.text
    assert response.json()["code"] == expected_code
    assert await small_file_redis.zcard(small_file_settings.queue_name) == depth_before
