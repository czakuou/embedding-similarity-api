"""Greedy micro-batching in front of the encoder.

arq runs one job per analysis; the model is far faster per item on a batch. Each job hands
its (query, document) pair to the Batcher and waits; a single task takes whatever is waiting
(up to `max_size`), lingers once for a few milliseconds if the batch is not full, then encodes
the whole batch in a worker thread so the event loop (arq's polling and health check) stays
responsive.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import structlog

from similarity.encoder import Encoder
from similarity.observability import inference_batch_duration_seconds, inference_batch_size

logger = structlog.get_logger()


@dataclass(slots=True)
class _Pending:
    query: str
    document: str
    future: asyncio.Future[float]


def score_pairs(encoder: Encoder, pairs: list[tuple[str, str]]) -> list[float]:
    """Pairwise cosine similarity of (query, document) pairs: one score per pair, never an
    N x N matrix. Queries and documents are encoded in one interleaved list.
    """
    texts = [text for pair in pairs for text in pair]
    embeddings = encoder.encode(texts)
    return encoder.similarity_pairwise(embeddings[0::2], embeddings[1::2])


class Batcher:
    def __init__(self, encoder: Encoder, *, max_size: int, linger_s: float) -> None:
        self._encoder = encoder
        self._max_size = max_size
        self._linger_s = linger_s
        self._pending: asyncio.Queue[_Pending] = asyncio.Queue()

    async def submit(self, query: str, document: str) -> float:
        """Returns the similarity score, or raises what the encoder raised for this pair."""
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        self._pending.put_nowait(_Pending(query, document, future))
        return await future

    async def run(self) -> None:
        while True:
            batch = [await self._pending.get()]
            self._drain(batch)
            if len(batch) < self._max_size and self._linger_s > 0:
                await asyncio.sleep(self._linger_s)
                self._drain(batch)
            await self._flush(batch)

    def _drain(self, batch: list[_Pending]) -> None:
        while len(batch) < self._max_size:
            try:
                batch.append(self._pending.get_nowait())
            except asyncio.QueueEmpty:
                return

    async def _flush(self, batch: list[_Pending]) -> None:
        # A job cancelled while waiting (timeout, shutdown) has a cancelled future: skip it.
        batch = [p for p in batch if not p.future.done()]
        if not batch:
            return
        inference_batch_size.observe(len(batch))
        started = time.monotonic()
        try:
            scores = await asyncio.to_thread(
                score_pairs, self._encoder, [(p.query, p.document) for p in batch]
            )
        except Exception:
            logger.exception("batch.encode_failed", batch_size=len(batch))
            await self._flush_one_by_one(batch)
        else:
            for pending, score in zip(batch, scores, strict=True):
                if not pending.future.done():
                    pending.future.set_result(score)
        inference_batch_duration_seconds.observe(time.monotonic() - started)
        logger.info("batch.processed", batch_size=len(batch))

    async def _flush_one_by_one(self, batch: list[_Pending]) -> None:
        """After a whole-batch failure, retry each pair alone: one bad item can't fail the rest."""
        for pending in batch:
            try:
                (score,) = await asyncio.to_thread(
                    score_pairs, self._encoder, [(pending.query, pending.document)]
                )
            except Exception as exc:
                if not pending.future.done():
                    pending.future.set_exception(exc)
            else:
                if not pending.future.done():
                    pending.future.set_result(score)
