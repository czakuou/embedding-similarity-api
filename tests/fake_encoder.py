"""Fake Encoder for tests: deterministic vectors from a text hash, no torch.

Implements the similarity.encoder.Encoder Protocol. A text containing
__POISON__ raises, to simulate a document that crashes the real model.
"""

from __future__ import annotations

import hashlib
import math

_DIMENSIONS = 8
POISON_MARKER = "__POISON__"


def _deterministic_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i] for i in range(_DIMENSIONS)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            if POISON_MARKER in text:
                raise ValueError(f"poisoned input: {text!r}")
        return [_deterministic_vector(text) for text in texts]

    def similarity_pairwise(
        self, embeddings1: list[list[float]], embeddings2: list[list[float]]
    ) -> list[float]:
        return [_cosine_similarity(a, b) for a, b in zip(embeddings1, embeddings2, strict=True)]
