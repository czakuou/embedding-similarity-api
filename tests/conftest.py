from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import redis.asyncio as redis
from testcontainers.community.redis import RedisContainer


@pytest.fixture(scope="session")
def redis_container() -> Iterator[RedisContainer]:
    with RedisContainer("redis:7.4-alpine") as container:
        yield container


@pytest.fixture
async def redis_client(redis_container: RedisContainer) -> AsyncIterator[redis.Redis]:
    url = (
        f"redis://{redis_container.get_container_host_ip()}:"
        f"{redis_container.get_exposed_port(redis_container.port)}/0"
    )
    client = redis.from_url(url, decode_responses=True)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.aclose()
