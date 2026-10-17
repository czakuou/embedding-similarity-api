"""Structured logging and worker metrics registry.

client_id is never a metric label (SPEC §8.1): unbounded cardinality would
bloat Prometheus RAM. It goes to logs instead.
"""

from __future__ import annotations

import logging
import sys

import structlog
from prometheus_client import Counter, Gauge, Histogram


def configure_logging(log_level: str) -> None:
    """JSON logs on stdout (SPEC §8.2): one line per event, no document text or query."""
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stdout)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        processors=[
            structlog.processors.TimeStamper(fmt="iso", key="ts"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


analysis_end_to_end_seconds = Histogram(
    "analysis_end_to_end_seconds", "Time from analysis creation to a done/failed result"
)
queue_depth = Gauge("queue_depth", "Outstanding (unacked) messages in the analyses stream")
queue_oldest_age_seconds = Gauge(
    "queue_oldest_age_seconds", "Age of the oldest message still in the analyses stream"
)
inference_batch_duration_seconds = Histogram(
    "inference_batch_duration_seconds", "Wall-clock time to encode and score one batch"
)
inference_batch_size = Histogram(
    "inference_batch_size",
    "Number of items processed together in one batch",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
analyses_completed_total = Counter(
    "analyses_completed_total", "Analyses that reached a terminal state", ["status"]
)
analysis_retries_total = Counter(
    "analysis_retries_total", "Times an analysis was retried after a failed attempt"
)
documents_truncated_total = Counter(
    "documents_truncated_total", "Analyses whose document was pre-cut to MAX_DOC_CHARS"
)
