"""arq worker wiring: startup/shutdown, queue metrics, and the Worker keyword arguments.

`worker_kwargs` is import-safe (no model load, no env read); `entrypoint.py` supplies the
real encoder, and the behaviour tests supply a pre-loaded one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import redis.asyncio as redis
import structlog
from arq import ArqRedis
from prometheus_client import start_http_server

from similarity.analysis import arq_redis_settings, deserialize_job, serialize_job
from similarity.config import WorkerSettings
from similarity.encoder import Encoder
from similarity.observability import configure_logging, queue_oldest_age_seconds
from similarity.observability import queue_depth as queue_depth_metric
from similarity.worker.batcher import Batcher
from similarity.worker.jobs import analyze

logger = structlog.get_logger()

QUEUE_METRICS_INTERVAL_S = 1.0
HEALTH_CHECK_INTERVAL_S = 15


async def observe_queue_metrics(pool: ArqRedis, queue_name: str) -> None:
    """queue_oldest_age_seconds is the primary alert signal. arq's queue is a sorted set
    scored by the time a job becomes runnable (its enqueue time in ms), so the first member
    is the oldest job. It stays in the set while it runs, so a job stuck after a crash keeps
    the age growing. A job deferred for a retry has a future score and counts as age 0.
    """
    queue_depth_metric.set(await pool.zcard(queue_name))
    oldest = await pool.zrange(queue_name, 0, 0, withscores=True)
    if oldest:
        queue_oldest_age_seconds.set(max(0.0, time.time() - oldest[0][1] / 1000))
    else:
        queue_oldest_age_seconds.set(0.0)


async def observe_queue_forever(pool: ArqRedis, queue_name: str) -> None:
    while True:
        try:
            await observe_queue_metrics(pool, queue_name)
        except Exception:
            logger.exception("worker.queue_metrics_failed")
        await asyncio.sleep(QUEUE_METRICS_INTERVAL_S)


def worker_kwargs(
    settings: WorkerSettings,
    load_encoder: Callable[[], Encoder],
    *,
    serve_metrics: bool = True,
) -> dict[str, Any]:
    async def on_startup(ctx: dict[str, Any]) -> None:
        configure_logging(settings.log_level)
        if serve_metrics:
            start_http_server(settings.metrics_port)
        encoder = await asyncio.to_thread(load_encoder)
        batcher = Batcher(
            encoder,
            max_size=settings.batch_max_size,
            linger_s=settings.batch_linger_ms / 1000,
        )
        ctx["settings"] = settings
        ctx["records"] = redis.from_url(settings.redis_url, decode_responses=True)
        ctx["batcher"] = batcher
        ctx["background"] = [
            asyncio.create_task(batcher.run()),
            asyncio.create_task(observe_queue_forever(ctx["redis"], settings.queue_name)),
        ]
        logger.info("worker.model_loaded")

    async def on_shutdown(ctx: dict[str, Any]) -> None:
        for task in ctx.get("background", []):
            task.cancel()
        records: redis.Redis | None = ctx.get("records")
        if records is not None:
            await records.aclose()

    return {
        "functions": [analyze],
        "queue_name": settings.queue_name,
        "redis_settings": arq_redis_settings(settings.redis_url),
        "on_startup": on_startup,
        "on_shutdown": on_shutdown,
        "job_serializer": serialize_job,
        "job_deserializer": deserialize_job,
        # Pipelining: while one batch is encoding, the next one is already collecting.
        "max_jobs": 2 * settings.batch_max_size,
        "poll_delay": settings.poll_delay_s,
        # arq treats a job whose in-progress marker expired (timeout + 10 s) as lost.
        "job_timeout": settings.visibility_timeout_s,
        # One extra try so an analysis whose attempts ran out gets failed by `analyze`
        # instead of silently dropped by arq.
        "max_tries": settings.max_attempts + 1,
        "retry_jobs": True,
        "keep_result": 0,
        "job_completion_wait": settings.shutdown_grace_s,
        "health_check_interval": HEALTH_CHECK_INTERVAL_S,
    }
