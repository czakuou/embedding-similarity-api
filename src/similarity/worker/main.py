"""Worker process entrypoint: load the model, warm up, run the consumer loop."""

from __future__ import annotations

import asyncio
import os

import redis.asyncio as redis
import structlog
from prometheus_client import start_http_server

from similarity.config import WorkerSettings
from similarity.encoder import SentenceTransformerEncoder
from similarity.observability import configure_logging
from similarity.worker.loop import install_sigterm_handler, run

logger = structlog.get_logger()


async def main() -> None:
    settings = WorkerSettings()
    configure_logging(settings.log_level)
    start_http_server(settings.metrics_port)

    model_name = os.environ["MODEL_NAME"]
    model_revision = os.environ["MODEL_REVISION"]
    encoder = SentenceTransformerEncoder(model_name, model_revision)
    encoder.warm_up()

    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    shutdown_event = asyncio.Event()
    install_sigterm_handler(shutdown_event)

    try:
        await run(
            redis_client,
            encoder,
            settings,
            consumer_name=f"worker-{os.getpid()}",
            shutdown_event=shutdown_event,
        )
    finally:
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
