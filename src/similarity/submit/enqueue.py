"""Rate limiting, backpressure, and the MULTI enqueue (I/O side of the POST slice)."""

from __future__ import annotations

import random
from datetime import UTC, datetime

import redis.asyncio as redis
from arq import ArqRedis
from limits import RateLimitItem, parse
from limits.aio.storage import RedisStorage
from limits.aio.strategies import FixedWindowRateLimiter
from limits.storage import storage_from_string
from uuid6 import uuid7

from similarity.analysis import (
    ANALYZE_JOB,
    AnalysisRecord,
    Status,
    analysis_key,
    expires_at,
    record_to_hash,
    validate_client_id,
)
from similarity.config import ApiSettings
from similarity.submit.outcomes import Accepted, Invalid, Outcome, QueueFull, RateLimited
from similarity.submit.validation import (
    decode_document,
    pre_cut_document,
    validate_filename,
    validate_query,
)


def jittered_retry_after(base_s: int, rng: random.Random) -> int:
    """`base + uniform(0, base)`: avoids a thundering herd of retries."""
    return base_s + rng.randint(0, base_s)


def rate_limit_item(rate_limit: str) -> RateLimitItem:
    return parse(rate_limit)


def make_redis_storage(redis_url: str) -> RedisStorage:
    """Backed by redis.asyncio (implementation='redispy'), not coredis: redis-py is
    already a dependency for the rest of the app, so this avoids adding a second
    Redis client library just for rate limiting.
    """
    result = storage_from_string(f"async+{redis_url}", implementation="redispy")
    if not isinstance(result, RedisStorage):
        raise TypeError(f"expected RedisStorage, got {type(result)}")
    return result


async def check_rate_limit(
    limiter: FixedWindowRateLimiter,
    item: RateLimitItem,
    client_id: str,
) -> bool:
    """True if the request is allowed."""
    return await limiter.hit(item, client_id)


async def queue_depth(redis_client: redis.Redis, queue_name: str) -> int:
    """Jobs queued, deferred for a retry, or in progress: arq keeps a job in this sorted set
    until it has finished.
    """
    # redis-py's stub returns Any here (its sync/async overloads share one signature).
    result: int = await redis_client.zcard(queue_name)
    return result


async def enqueue(
    redis_client: redis.Redis,
    arq_pool: ArqRedis,
    *,
    queue_name: str,
    client_id: str,
    document: str,
    query: str,
    document_truncated: bool,
    result_ttl_s: int,
) -> Accepted:
    """Writes the analysis record, then enqueues the job. Two commands on one Redis, not one
    transaction: the record goes first, so a worker never sees a job without a record. If
    the enqueue fails the record is removed again.
    """
    analysis_id = str(uuid7())
    created_at = datetime.now(UTC)
    expiry = expires_at(created_at, result_ttl_s)
    record = AnalysisRecord(
        id=analysis_id,
        client_id=client_id,
        status=Status.QUEUED,
        created_at=created_at,
        expires_at=expiry,
        document_truncated=document_truncated,
    )

    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.hset(analysis_key(analysis_id), mapping=record_to_hash(record))
        pipe.expireat(analysis_key(analysis_id), int(expiry.timestamp()))
        await pipe.execute()

    try:
        job = await arq_pool.enqueue_job(
            ANALYZE_JOB,
            analysis_id,
            document,
            query,
            _job_id=analysis_id,
            _queue_name=queue_name,
            _expires=result_ttl_s,
        )
    except Exception:
        await redis_client.delete(analysis_key(analysis_id))
        raise
    if job is None:  # a fresh UUIDv7 collided with an existing job id
        await redis_client.delete(analysis_key(analysis_id))
        raise RuntimeError(f"job {analysis_id} already exists")

    return Accepted(analysis_id=analysis_id, created_at=created_at, expires_at=expiry)


async def decide_and_enqueue(
    *,
    redis_client: redis.Redis,
    arq_pool: ArqRedis,
    limiter: FixedWindowRateLimiter,
    rate_limit: RateLimitItem,
    settings: ApiSettings,
    client_id: str | None,
    filename: str | None,
    raw_content: bytes,
    content_too_large: bool,
    query: str,
    rng: random.Random,
) -> Outcome:
    """Orchestration of the request checks, cheapest first.

    Order: client id -> rate limit -> file/query validation -> queue depth -> enqueue.
    Rate limiting runs before any body validation, so a 429 costs no upload I/O
    beyond the single Redis round-trip already spent reading the header.
    """
    validated_client_id = validate_client_id(client_id)
    if validated_client_id is None:
        return Invalid(reason="validation_error", detail="missing or invalid X-Client-Id")

    allowed = await check_rate_limit(limiter, rate_limit, validated_client_id)
    if not allowed:
        return RateLimited(retry_after_s=jittered_retry_after(settings.retry_after_base_s, rng))

    if not validate_filename(filename):
        return Invalid(reason="unsupported_media_type", detail="file must have a .txt extension")

    if content_too_large:
        return Invalid(reason="payload_too_large", detail="file exceeds MAX_FILE_SIZE_BYTES")

    document = decode_document(raw_content)
    if document is None:
        return Invalid(reason="validation_error", detail="file is not valid UTF-8")
    if len(document) == 0:
        return Invalid(reason="validation_error", detail="file is empty")

    validated_query = validate_query(query)
    if validated_query is None:
        return Invalid(
            reason="validation_error", detail="query must be 1-1000 chars after trimming"
        )

    depth = await queue_depth(redis_client, settings.queue_name)
    if depth >= settings.max_queue_depth:
        return QueueFull(retry_after_s=jittered_retry_after(settings.retry_after_base_s, rng))

    pre_cut = pre_cut_document(document, settings.max_doc_chars)
    return await enqueue(
        redis_client,
        arq_pool,
        queue_name=settings.queue_name,
        client_id=validated_client_id,
        document=pre_cut.text,
        query=validated_query,
        document_truncated=pre_cut.truncated,
        result_ttl_s=settings.result_ttl_s,
    )
