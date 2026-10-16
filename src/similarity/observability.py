"""Structured logging setup, shared by the API and worker processes."""

from __future__ import annotations

import logging
import sys

import structlog


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
