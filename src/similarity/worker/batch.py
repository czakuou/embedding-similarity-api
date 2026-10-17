"""One batch: increment attempts, encode + pairwise similarity, write results.

process_batch is the single entry point driven directly by BEHAVIOR.md tests
(one loop iteration, no background threads) and by the real consumer loop.
"""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from typing import cast

import redis.asyncio as redis
import structlog

from similarity.analysis import (
    AnalysisError,
    AnalysisRecord,
    Status,
    analysis_key,
    hash_to_record,
    record_to_hash,
)
from similarity.encoder import Encoder
from similarity.observability import (
    analyses_completed_total,
    analysis_end_to_end_seconds,
    analysis_retries_total,
    documents_truncated_total,
)
from similarity.worker.items import QueueItem, should_give_up

logger = structlog.get_logger()


# redis-py's stub shares one signature between the sync and async clients, so several
# methods are typed as `Awaitable[T] | T`; on redis.asyncio.Redis they are always
# awaitable, so these two helpers cast before awaiting (verified against a live Redis).
async def fetch_record_or_none(
    redis_client: redis.Redis, analysis_id: str
) -> AnalysisRecord | None:
    fields = await cast(
        "Awaitable[dict[str, str]]", redis_client.hgetall(analysis_key(analysis_id))
    )
    if not fields:
        return None
    return hash_to_record(fields)


async def increment_attempts(redis_client: redis.Redis, analysis_id: str) -> int:
    """HINCRBY before computing (SPEC §5): a worker killed mid-batch never reaches an
    except block, so counting at the start guarantees MAX_ATTEMPTS is eventually hit.

    Only called once fetch_record_or_none has confirmed the record exists: HINCRBY on
    a missing key creates it (Redis semantics), which would otherwise leave a
    half-written, unparseable hash behind for an analysis that was never queued.
    """
    attempts = await cast(
        "Awaitable[int]", redis_client.hincrby(analysis_key(analysis_id), "attempts", 1)
    )
    await cast(
        "Awaitable[int]",
        redis_client.hset(analysis_key(analysis_id), "status", Status.PROCESSING.value),
    )
    return attempts


def done_record(
    record: AnalysisRecord, *, result: list[float], attempts: int, now: datetime
) -> AnalysisRecord:
    return AnalysisRecord(
        id=record.id,
        client_id=record.client_id,
        status=Status.DONE,
        created_at=record.created_at,
        expires_at=record.expires_at,
        attempts=attempts,
        updated_at=now,
        result=result,
        document_truncated=record.document_truncated,
    )


def failed_record(
    record: AnalysisRecord, *, attempts: int, now: datetime, message: str
) -> AnalysisRecord:
    return AnalysisRecord(
        id=record.id,
        client_id=record.client_id,
        status=Status.FAILED,
        created_at=record.created_at,
        expires_at=record.expires_at,
        attempts=attempts,
        updated_at=now,
        error=AnalysisError(code="PROCESSING_FAILED", message=message, attempts=attempts),
    )


def observe_terminal(record: AnalysisRecord, *, now: datetime) -> None:
    """SPEC §8.1 metrics for a record reaching a terminal state (done or failed)."""
    analyses_completed_total.labels(status=record.status.value).inc()
    analysis_end_to_end_seconds.observe((now - record.created_at).total_seconds())
    if record.attempts > 1:
        analysis_retries_total.inc()
    if record.document_truncated:
        documents_truncated_total.inc()


async def write_and_ack(
    redis_client: redis.Redis,
    record: AnalysisRecord,
    *,
    stream_key: str,
    consumer_group: str,
    msg_id: str,
    now: datetime,
) -> None:
    """Writes the updated record (re-applying its TTL, SPEC §5 expiry invariant) and
    acks + deletes the stream message in one pipeline.
    """
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.hset(analysis_key(record.id), mapping=record_to_hash(record))
        pipe.expireat(analysis_key(record.id), int(record.expires_at.timestamp()))
        pipe.xack(stream_key, consumer_group, msg_id)
        pipe.xdel(stream_key, msg_id)
        await pipe.execute()
    observe_terminal(record, now=now)


async def give_up_on_item(
    redis_client: redis.Redis,
    item: QueueItem,
    *,
    record: AnalysisRecord,
    attempts: int,
    stream_key: str,
    consumer_group: str,
    now: datetime,
) -> None:
    """SPEC §5 poison circuit breaker: attempts exhausted, fail without computing."""
    updated = failed_record(record, attempts=attempts, now=now, message="maximum attempts exceeded")
    await write_and_ack(
        redis_client,
        updated,
        stream_key=stream_key,
        consumer_group=consumer_group,
        msg_id=item.msg_id,
        now=now,
    )
    logger.info("analysis.failed", analysis_id=item.analysis_id, attempts=attempts)


async def skip_missing_item(
    redis_client: redis.Redis, item: QueueItem, *, stream_key: str, consumer_group: str
) -> None:
    """SPEC §5: a record that no longer exists (expired or deleted) is skipped and
    acked -- no resurrection.
    """
    await redis_client.xack(stream_key, consumer_group, item.msg_id)
    await redis_client.xdel(stream_key, item.msg_id)


async def compute_one(encoder: Encoder, item: QueueItem) -> list[float]:
    embeddings = encoder.encode([item.query, item.document])
    return encoder.similarity_pairwise([embeddings[0]], [embeddings[1]])


async def process_item_alone(
    redis_client: redis.Redis,
    encoder: Encoder,
    item: QueueItem,
    *,
    record: AnalysisRecord,
    attempts: int,
    stream_key: str,
    consumer_group: str,
    now: datetime,
) -> None:
    """Used both for a one-item batch and to retry an item after a whole-batch
    encode failure (SPEC §5: one bad document must not fail the other 31).
    """
    try:
        result = await compute_one(encoder, item)
    except Exception:
        logger.exception("analysis.failed", analysis_id=item.analysis_id, attempts=attempts)
        return  # left unacked: XAUTOCLAIM will redeliver it later

    updated = done_record(record, result=result, attempts=attempts, now=now)
    await write_and_ack(
        redis_client,
        updated,
        stream_key=stream_key,
        consumer_group=consumer_group,
        msg_id=item.msg_id,
        now=now,
    )
    logger.info("analysis.completed", analysis_id=item.analysis_id, attempts=attempts)


async def process_batch(
    redis_client: redis.Redis,
    encoder: Encoder,
    items: list[QueueItem],
    *,
    stream_key: str,
    consumer_group: str,
    max_attempts: int,
    now: datetime,
) -> None:
    """Processes a whole batch: items whose record no longer exists are skipped first
    (SPEC §5: no resurrection), then attempts are incremented for the rest before any
    computation. Items with attempts exhausted are resolved immediately, and the rest
    are encoded together in one call (SPEC §6.2). If that call raises, items are
    retried one by one so one poisoned document can't fail the others.
    """
    live_items: list[tuple[QueueItem, AnalysisRecord, int]] = []
    for item in items:
        record = await fetch_record_or_none(redis_client, item.analysis_id)
        if record is None:
            await skip_missing_item(
                redis_client, item, stream_key=stream_key, consumer_group=consumer_group
            )
            continue
        attempts = await increment_attempts(redis_client, item.analysis_id)
        if should_give_up(attempts, max_attempts):
            await give_up_on_item(
                redis_client,
                item,
                record=record,
                attempts=attempts,
                stream_key=stream_key,
                consumer_group=consumer_group,
                now=now,
            )
            continue
        live_items.append((item, record, attempts))

    if not live_items:
        return

    try:
        texts = [text for item, _, _ in live_items for text in (item.query, item.document)]
        embeddings = encoder.encode(texts)
        query_embeddings = embeddings[0::2]
        document_embeddings = embeddings[1::2]
        results = encoder.similarity_pairwise(query_embeddings, document_embeddings)
    except Exception:
        logger.exception("batch.encode_failed", batch_size=len(live_items))
        for item, record, attempts in live_items:
            await process_item_alone(
                redis_client,
                encoder,
                item,
                record=record,
                attempts=attempts,
                stream_key=stream_key,
                consumer_group=consumer_group,
                now=now,
            )
        return

    updated_records: list[AnalysisRecord] = []
    async with redis_client.pipeline(transaction=True) as pipe:
        for (item, record, attempts), result in zip(live_items, results, strict=True):
            updated = done_record(record, result=[result], attempts=attempts, now=now)
            updated_records.append(updated)
            pipe.hset(analysis_key(item.analysis_id), mapping=record_to_hash(updated))
            pipe.expireat(analysis_key(item.analysis_id), int(record.expires_at.timestamp()))
            pipe.xack(stream_key, consumer_group, item.msg_id)
            pipe.xdel(stream_key, item.msg_id)
        await pipe.execute()

    for (item, _, attempts), updated in zip(live_items, updated_records, strict=True):
        observe_terminal(updated, now=now)
        logger.info("analysis.completed", analysis_id=item.analysis_id, attempts=attempts)
