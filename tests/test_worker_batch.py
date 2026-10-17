from __future__ import annotations

from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import cast

import redis.asyncio as redis

from similarity.analysis import (
    AnalysisRecord,
    Status,
    analysis_key,
    expires_at,
    hash_to_record,
    record_to_hash,
)
from similarity.analysis import AnalysisRecord as Record
from similarity.worker.batch import process_batch
from similarity.worker.items import QueueItem

from .fake_encoder import FakeEncoder

STREAM_KEY = "test-stream"
CONSUMER_GROUP = "test-group"


# Same redis-py stub quirk as similarity/worker/batch.py: several methods share one
# signature between the sync and async clients, so their return type as seen by mypy
# doesn't narrow to "always awaitable" the way it behaves on redis.asyncio.Redis.
async def _get_record(redis_client: redis.Redis, analysis_id: str) -> AnalysisRecord:
    fields = await cast(
        "Awaitable[dict[str, str]]", redis_client.hgetall(analysis_key(analysis_id))
    )
    return hash_to_record(fields)


async def _put_record(redis_client: redis.Redis, record: AnalysisRecord) -> None:
    await cast(
        "Awaitable[int]", redis_client.hset(analysis_key(record.id), mapping=record_to_hash(record))
    )


async def _seed_queued_analysis(
    redis_client: redis.Redis, *, analysis_id: str, document: str, query: str
) -> str:
    created_at = datetime.now(UTC)
    record = Record(
        id=analysis_id,
        client_id="client-a",
        status=Status.QUEUED,
        created_at=created_at,
        expires_at=expires_at(created_at, 86_400),
    )
    await _put_record(redis_client, record)
    msg_id: str = await redis_client.xadd(
        STREAM_KEY,
        {
            "analysis_id": analysis_id,
            "client_id": "client-a",
            "document": document,
            "query": query,
            "document_truncated": "0",
        },
    )
    await redis_client.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0")
    return msg_id


async def test_given_healthy_item_when_batch_processed_then_record_is_done(
    redis_client: redis.Redis,
) -> None:
    msg_id = await _seed_queued_analysis(
        redis_client, analysis_id="a1", document="a fox document", query="fox"
    )
    item = QueueItem(
        msg_id=msg_id,
        analysis_id="a1",
        client_id="client-a",
        document="a fox document",
        query="fox",
        document_truncated=False,
    )

    await process_batch(
        redis_client,
        FakeEncoder(),
        [item],
        stream_key=STREAM_KEY,
        consumer_group=CONSUMER_GROUP,
        max_attempts=3,
        now=datetime.now(UTC),
    )

    record = await _get_record(redis_client, "a1")
    assert record.status is Status.DONE
    assert record.result is not None
    assert len(record.result) == 1
    assert -1.0 <= record.result[0] <= 1.0
    assert await redis_client.xlen(STREAM_KEY) == 0  # acked and deleted (SPEC §6.3)


async def test_given_poisoned_item_among_healthy_when_processed_then_only_poison_unacked(
    redis_client: redis.Redis,
) -> None:
    healthy_msg_id = await _seed_queued_analysis(
        redis_client, analysis_id="healthy-1", document="a fox document", query="fox"
    )
    poison_msg_id: str = await redis_client.xadd(
        STREAM_KEY,
        {
            "analysis_id": "poison-1",
            "client_id": "client-a",
            "document": "contains __POISON__ marker",
            "query": "fox",
            "document_truncated": "0",
        },
    )
    created_at = datetime.now(UTC)
    poison_record = Record(
        id="poison-1",
        client_id="client-a",
        status=Status.QUEUED,
        created_at=created_at,
        expires_at=expires_at(created_at, 86_400),
    )
    await _put_record(redis_client, poison_record)

    items = [
        QueueItem(
            msg_id=healthy_msg_id,
            analysis_id="healthy-1",
            client_id="client-a",
            document="a fox document",
            query="fox",
            document_truncated=False,
        ),
        QueueItem(
            msg_id=poison_msg_id,
            analysis_id="poison-1",
            client_id="client-a",
            document="contains __POISON__ marker",
            query="fox",
            document_truncated=False,
        ),
    ]

    await process_batch(
        redis_client,
        FakeEncoder(),
        items,
        stream_key=STREAM_KEY,
        consumer_group=CONSUMER_GROUP,
        max_attempts=3,
        now=datetime.now(UTC),
    )

    healthy = await _get_record(redis_client, "healthy-1")
    poison = await _get_record(redis_client, "poison-1")
    assert healthy.status is Status.DONE
    assert poison.status is Status.PROCESSING
    # the poisoned message is still in the stream, unacked, for later redelivery
    assert await redis_client.xlen(STREAM_KEY) == 1


async def test_given_attempts_exhausted_on_reclaim_when_batch_processed_then_failed(
    redis_client: redis.Redis,
) -> None:
    created_at = datetime.now(UTC)
    record = Record(
        id="exhausted-1",
        client_id="client-a",
        status=Status.PROCESSING,
        created_at=created_at,
        expires_at=expires_at(created_at, 86_400),
        attempts=2,
    )
    await _put_record(redis_client, record)
    msg_id: str = await redis_client.xadd(
        STREAM_KEY,
        {
            "analysis_id": "exhausted-1",
            "client_id": "client-a",
            "document": "contains __POISON__ marker",
            "query": "fox",
            "document_truncated": "0",
        },
    )
    await redis_client.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0")
    item = QueueItem(
        msg_id=msg_id,
        analysis_id="exhausted-1",
        client_id="client-a",
        document="contains __POISON__ marker",
        query="fox",
        document_truncated=False,
    )

    # increment_attempts makes this the 3rd attempt; max_attempts=2 means 3 > 2 -> give up.
    await process_batch(
        redis_client,
        FakeEncoder(),
        [item],
        stream_key=STREAM_KEY,
        consumer_group=CONSUMER_GROUP,
        max_attempts=2,
        now=datetime.now(UTC),
    )

    failed = await _get_record(redis_client, "exhausted-1")
    assert failed.status is Status.FAILED
    assert failed.attempts == 3  # the circuit breaker itself: MAX_ATTEMPTS=2 -> gave up on the 3rd
    assert failed.error is not None
    assert failed.error.code == "PROCESSING_FAILED"
    assert await redis_client.xlen(STREAM_KEY) == 0  # acked and deleted (SPEC §6.3)


async def test_given_missing_record_when_batch_processed_then_message_is_skipped_and_acked(
    redis_client: redis.Redis,
) -> None:
    msg_id: str = await redis_client.xadd(
        STREAM_KEY,
        {
            "analysis_id": "gone-1",
            "client_id": "client-a",
            "document": "doc",
            "query": "q",
            "document_truncated": "0",
        },
    )
    await redis_client.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0")
    item = QueueItem(
        msg_id=msg_id,
        analysis_id="gone-1",
        client_id="client-a",
        document="doc",
        query="q",
        document_truncated=False,
    )

    # No record was ever written for "gone-1" (simulates an expired/deleted record).
    await process_batch(
        redis_client,
        FakeEncoder(),
        [item],
        stream_key=STREAM_KEY,
        consumer_group=CONSUMER_GROUP,
        max_attempts=3,
        now=datetime.now(UTC),
    )

    assert await redis_client.xlen(STREAM_KEY) == 0
    assert await redis_client.exists(analysis_key("gone-1")) == 0
