from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from similarity.worker.batcher import Batcher

from .fake_encoder import POISON_MARKER, FakeEncoder


class RecordingEncoder(FakeEncoder):
    """Remembers how many texts each encode call received."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return super().encode(texts)


@pytest.fixture
async def running() -> AsyncIterator[tuple[Batcher, RecordingEncoder]]:
    encoder = RecordingEncoder()
    batcher = Batcher(encoder, max_size=8, linger_s=0.05)
    task = asyncio.create_task(batcher.run())
    yield batcher, encoder
    task.cancel()


async def test_given_concurrent_submissions_when_batching_then_encoded_together(
    running: tuple[Batcher, RecordingEncoder],
) -> None:
    batcher, encoder = running

    scores = await asyncio.gather(*(batcher.submit("fox", f"document {i}") for i in range(5)))

    assert len(scores) == 5
    assert encoder.calls == [10]  # five (query, document) pairs, one encode call


async def test_given_more_submissions_than_batch_size_when_batching_then_split_by_max_size(
    running: tuple[Batcher, RecordingEncoder],
) -> None:
    batcher, encoder = running

    await asyncio.gather(*(batcher.submit("fox", f"document {i}") for i in range(12)))

    assert max(encoder.calls) <= 16  # at most 8 pairs per call
    assert sum(encoder.calls) == 24


async def test_given_one_bad_pair_in_a_batch_when_batching_then_only_it_fails(
    running: tuple[Batcher, RecordingEncoder],
) -> None:
    batcher, _ = running

    results = await asyncio.gather(
        batcher.submit("fox", "a healthy document"),
        batcher.submit("fox", f"{POISON_MARKER} a document the encoder rejects"),
        batcher.submit("fox", "another healthy document"),
        return_exceptions=True,
    )

    assert isinstance(results[0], float)
    assert isinstance(results[1], ValueError)
    assert isinstance(results[2], float)


async def test_given_waiter_cancelled_before_flush_when_batching_then_others_still_scored(
    running: tuple[Batcher, RecordingEncoder],
) -> None:
    batcher, _ = running
    cancelled = asyncio.create_task(batcher.submit("fox", "abandoned"))
    kept = asyncio.create_task(batcher.submit("fox", "kept"))
    await asyncio.sleep(0)
    cancelled.cancel()

    assert isinstance(await kept, float)
