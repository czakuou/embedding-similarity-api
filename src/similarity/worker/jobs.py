"""The arq job: one analysis.

arq owns the queue, redelivery after a crash (a job whose in-progress marker expires is
picked up again) and the retry bookkeeping (`ctx["job_try"]` counts every start, including
the ones that ended in a crash). This module owns the analysis record the client polls.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

import redis.asyncio as redis
import structlog
from arq import Retry

from similarity.analysis import (
    AnalysisError,
    AnalysisRecord,
    Status,
    analysis_key,
    hash_to_record,
    record_to_hash,
)
from similarity.observability import (
    analyses_completed_total,
    analysis_end_to_end_seconds,
    analysis_retries_total,
    documents_truncated_total,
)

logger = structlog.get_logger()

GAVE_UP = "maximum attempts exceeded"


async def fetch_record(records: redis.Redis, analysis_id: str) -> AnalysisRecord | None:
    # redis-py's stub shares one signature between the sync and async clients, so HGETALL
    # is typed `Awaitable[dict] | dict`; on redis.asyncio.Redis it is always awaitable.
    fields = await cast("Awaitable[dict[str, str]]", records.hgetall(analysis_key(analysis_id)))
    if not fields:
        return None
    return hash_to_record(fields)


async def write_record(records: redis.Redis, record: AnalysisRecord) -> None:
    """Writes the whole record and re-applies its expiry. EXPIREAT with a time in the past
    deletes the key, so a record that expired in the meantime is not left behind.
    """
    async with records.pipeline(transaction=True) as pipe:
        pipe.hset(analysis_key(record.id), mapping=record_to_hash(record))
        pipe.expireat(analysis_key(record.id), int(record.expires_at.timestamp()))
        await pipe.execute()


def observe_terminal(record: AnalysisRecord, *, now: datetime) -> None:
    """Metrics for a record reaching a terminal state (done or failed)."""
    analyses_completed_total.labels(status=record.status.value).inc()
    analysis_end_to_end_seconds.observe((now - record.created_at).total_seconds())
    if record.attempts > 1:
        analysis_retries_total.inc()
    if record.document_truncated:
        documents_truncated_total.inc()


async def finish(records: redis.Redis, record: AnalysisRecord) -> None:
    await write_record(records, record)
    observe_terminal(record, now=record.updated_at or datetime.now(UTC))


def failed(record: AnalysisRecord, *, attempts: int, message: str) -> AnalysisRecord:
    return replace(
        record,
        status=Status.FAILED,
        attempts=attempts,
        updated_at=datetime.now(UTC),
        error=AnalysisError(code="PROCESSING_FAILED", message=message, attempts=attempts),
    )


async def analyze(ctx: dict[str, Any], analysis_id: str, document: str, query: str) -> None:
    settings, records, batcher = ctx["settings"], ctx["records"], ctx["batcher"]

    record = await fetch_record(records, analysis_id)
    if record is None:
        return  # expired or deleted by the client: nothing to write back

    attempts: int = ctx["job_try"]
    if attempts > settings.max_attempts:
        # Redelivered after crashes until the attempts ran out: fail without computing.
        await finish(records, failed(record, attempts=attempts, message=GAVE_UP))
        logger.info("analysis.failed", analysis_id=analysis_id, attempts=attempts)
        return

    await write_record(records, replace(record, status=Status.PROCESSING, attempts=attempts))

    try:
        score = await batcher.submit(query, document)
    except Exception:
        logger.exception("analysis.compute_failed", analysis_id=analysis_id, attempts=attempts)
        if attempts >= settings.max_attempts:
            await finish(records, failed(record, attempts=attempts, message=GAVE_UP))
            logger.info("analysis.failed", analysis_id=analysis_id, attempts=attempts)
            return
        raise Retry(defer=settings.visibility_timeout_s) from None

    done = replace(
        record,
        status=Status.DONE,
        attempts=attempts,
        updated_at=datetime.now(UTC),
        result=[score],
    )
    await finish(records, done)
    logger.info("analysis.completed", analysis_id=analysis_id, attempts=attempts)
