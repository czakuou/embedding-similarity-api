# syntax=docker/dockerfile:1
# Pinned by digest, not tag: reproducible builds (SPEC §11: model delivery).
FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS base

COPY --from=ghcr.io/astral-sh/uv@sha256:4faec156e35a5f345d57804d8858c6ba1cf6352ce5f4bffc11b7fdebdef46a38 /uv /uvx /usr/local/bin/

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home --shell /bin/bash app
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./


# ---- API image: no torch, no sentence-transformers -------------------------
FROM base AS api

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-group worker --no-group dev

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-group worker --no-group dev

USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "similarity.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]


# ---- Worker image: CPU torch + sentence-transformers, model baked in ------
FROM base AS worker

ARG MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ARG MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ENV MODEL_NAME=${MODEL_NAME} \
    MODEL_REVISION=${MODEL_REVISION} \
    HF_HOME=/app/.cache/huggingface

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group worker --no-group dev

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group worker --no-group dev

# Baked in at build time (SPEC §11): no runtime dependency on Hugging Face,
# reproducible across replicas. HF_HUB_OFFLINE=1 at runtime enforces this.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${MODEL_NAME}', revision='${MODEL_REVISION}')"

RUN mkdir -p /app/.cache && chown -R app:app /app/.cache
ENV HF_HUB_OFFLINE=1

USER 1000:1000
EXPOSE 9100
CMD ["python", "-m", "similarity.worker.main"]
