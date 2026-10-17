"""Consumer loop: XREADGROUP, periodic XAUTOCLAIM, heartbeat, warm-up, SIGTERM."""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

import redis.asyncio as redis
import structlog
from redis.exceptions import ResponseError

from similarity.config import WorkerSettings
from similarity.encoder import Encoder
from similarity.observability import (
    inference_batch_duration_seconds,
    inference_batch_size,
    queue_oldest_age_seconds,
)
from similarity.observability import queue_depth as queue_depth_metric
from similarity.worker.batch import process_batch
from similarity.worker.items import QueueItem, parse_xautoclaim_response, parse_xreadgroup_response

logger = structlog.get_logger()


async def ensure_consumer_group(
    redis_client: redis.Redis, stream_key: str, consumer_group: str
) -> None:
    try:
        await redis_client.xgroup_create(stream_key, consumer_group, id="0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def write_heartbeat(heartbeat_path: str) -> None:
    """Updated on every loop iteration, including empty reads (SPEC §8.3), so an idle
    worker never looks dead.
    """
    Path(heartbeat_path).write_text(str(time.time()))


async def observe_queue_metrics(redis_client: redis.Redis, stream_key: str) -> None:
    """queue_oldest_age_seconds is the primary alert signal (SPEC §8.1): a stream
    entry ID's leading part is a millisecond timestamp, so age is free to compute.
    """
    depth = await redis_client.xlen(stream_key)
    queue_depth_metric.set(depth)

    oldest = await redis_client.xrange(stream_key, min="-", max="+", count=1)
    if oldest:
        oldest_id = oldest[0][0]
        oldest_ms = int(oldest_id.split("-")[0])
        queue_oldest_age_seconds.set(max(0.0, time.time() - oldest_ms / 1000))
    else:
        queue_oldest_age_seconds.set(0.0)


async def read_batch(
    redis_client: redis.Redis,
    *,
    stream_key: str,
    consumer_group: str,
    consumer_name: str,
    batch_max_size: int,
    read_block_ms: int,
) -> list[QueueItem]:
    response = await redis_client.xreadgroup(
        consumer_group,
        consumer_name,
        {stream_key: ">"},
        count=batch_max_size,
        block=read_block_ms,
    )
    return parse_xreadgroup_response(response)


async def reclaim_idle(
    redis_client: redis.Redis,
    *,
    stream_key: str,
    consumer_group: str,
    consumer_name: str,
    visibility_timeout_s: int,
    batch_max_size: int,
) -> list[QueueItem]:
    response = await redis_client.xautoclaim(
        stream_key,
        consumer_group,
        consumer_name,
        min_idle_time=visibility_timeout_s * 1000,
        start_id="0-0",
        count=batch_max_size,
    )
    return parse_xautoclaim_response(response)


async def run_one_iteration(
    redis_client: redis.Redis,
    encoder: Encoder,
    settings: WorkerSettings,
    *,
    consumer_name: str,
    force_reclaim: bool = False,
) -> list[QueueItem]:
    """One pass of the consumer loop body: read, maybe reclaim, process, observe.

    Pulled out of run() so BEHAVIOR.md scenarios can drive the worker
    deterministically -- no background threads or sleeps (SPEC §13) -- by
    calling this once per expected batch instead of running the real loop.
    Returns the items processed, so a test can assert on batch composition.
    """
    items = await read_batch(
        redis_client,
        stream_key=settings.stream_key,
        consumer_group=settings.consumer_group,
        consumer_name=consumer_name,
        batch_max_size=settings.batch_max_size,
        read_block_ms=settings.read_block_ms,
    )

    if force_reclaim:
        items += await reclaim_idle(
            redis_client,
            stream_key=settings.stream_key,
            consumer_group=settings.consumer_group,
            consumer_name=consumer_name,
            visibility_timeout_s=settings.visibility_timeout_s,
            batch_max_size=settings.batch_max_size,
        )

    if items:
        inference_batch_size.observe(len(items))
        batch_started = time.monotonic()
        await process_batch(
            redis_client,
            encoder,
            items,
            stream_key=settings.stream_key,
            consumer_group=settings.consumer_group,
            max_attempts=settings.max_attempts,
            now=datetime.now(UTC),
        )
        inference_batch_duration_seconds.observe(time.monotonic() - batch_started)
        logger.info("batch.processed", batch_size=len(items))

    await observe_queue_metrics(redis_client, settings.stream_key)
    return items


async def run(
    redis_client: redis.Redis,
    encoder: Encoder,
    settings: WorkerSettings,
    *,
    consumer_name: str,
    shutdown_event: asyncio.Event,
) -> None:
    await ensure_consumer_group(redis_client, settings.stream_key, settings.consumer_group)
    logger.info("worker.model_loaded")

    last_reclaim = time.monotonic()
    while not shutdown_event.is_set():
        force_reclaim = time.monotonic() - last_reclaim >= settings.reclaim_interval_s
        if force_reclaim:
            last_reclaim = time.monotonic()
        await run_one_iteration(
            redis_client,
            encoder,
            settings,
            consumer_name=consumer_name,
            force_reclaim=force_reclaim,
        )
        write_heartbeat(settings.heartbeat_path)


def install_sigterm_handler(shutdown_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, shutdown_event.set)
