"""`arq similarity.worker.entrypoint.ArqSettings` runs the worker;
`arq --check similarity.worker.entrypoint.ArqSettings` is its liveness probe.
"""

from __future__ import annotations

import os

from similarity.config import WorkerSettings
from similarity.encoder import SentenceTransformerEncoder
from similarity.worker.app import worker_kwargs


def load_encoder() -> SentenceTransformerEncoder:
    encoder = SentenceTransformerEncoder(os.environ["MODEL_NAME"], os.environ["MODEL_REVISION"])
    encoder.warm_up()
    return encoder


ArqSettings = worker_kwargs(WorkerSettings(), load_encoder)
