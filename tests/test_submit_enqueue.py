from __future__ import annotations

import random

from similarity.submit.enqueue import jittered_retry_after


def test_given_fixed_rng_when_computing_retry_after_then_within_base_to_twice_base() -> None:
    rng = random.Random(42)
    for _ in range(100):
        value = jittered_retry_after(base_s=5, rng=rng)
        assert 5 <= value <= 10
