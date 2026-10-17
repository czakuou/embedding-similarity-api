"""Encoder Protocol + sentence-transformers implementation.

The only module that imports torch / sentence-transformers (SPEC §3.5): a
unit test asserts that importing similarity.api never pulls this module in.
"""

from __future__ import annotations

from typing import Protocol

Embedding = list[float]


class Encoder(Protocol):
    """Implemented by the real model (below) and by tests/fake_encoder.py.

    Plain lists, not numpy.ndarray: the fake encoder used by unit/BDD tests
    needs no numpy/torch dependency to satisfy this Protocol.
    """

    def encode(self, texts: list[str]) -> list[Embedding]: ...

    def similarity_pairwise(
        self, embeddings1: list[Embedding], embeddings2: list[Embedding]
    ) -> list[float]: ...


class SentenceTransformerEncoder:
    """Wraps sentence-transformers/all-MiniLM-L6-v2, pinned to MODEL_REVISION."""

    def __init__(self, model_name: str, model_revision: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, revision=model_revision)

    def encode(self, texts: list[str]) -> list[Embedding]:
        embeddings = self._model.encode(texts)
        return [row.tolist() for row in embeddings]

    def similarity_pairwise(
        self, embeddings1: list[Embedding], embeddings2: list[Embedding]
    ) -> list[float]:
        import numpy as np

        result: list[float] = self._model.similarity_pairwise(
            np.asarray(embeddings1), np.asarray(embeddings2)
        ).tolist()
        return result

    def warm_up(self) -> None:
        """One throwaway encode call, so the first real batch doesn't pay lazy-init cost."""
        self.encode(["warm up"])
