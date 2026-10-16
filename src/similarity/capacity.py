"""Pure capacity formula from SPEC §7.1. Used by benchmark.py and its tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

SECONDS_PER_DAY = 86_400
TARGET_UTILISATION = 0.7


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    requests_per_day: float
    average_rps: float
    peak_rps: float
    peak_embeddings_per_s: float
    worker_replicas: int
    max_queue_depth: int


def peak_embeddings_per_s(
    requests_per_day: float,
    peak_factor: float,
    query_cache_hit_rate: float = 0.0,
) -> float:
    """SPEC §7.1: average -> peak req/s -> embeddings/s (query + doc, minus cache hits)."""
    average_rps = requests_per_day / SECONDS_PER_DAY
    peak_rps = average_rps * peak_factor
    return peak_rps * 2 * (1 - query_cache_hit_rate / 2)


def worker_replicas(peak_emb_s: float, replica_throughput: float) -> int:
    """`ceil(peak_emb_s / (T * 0.7)) + 1`; the +1 tolerates the loss of one replica."""
    return math.ceil(peak_emb_s / (replica_throughput * TARGET_UTILISATION)) + 1


def max_queue_depth(
    active_replicas: int, replica_throughput: float, max_acceptable_wait_s: float
) -> int:
    """`active_replicas * (T / 2) * MAX_ACCEPTABLE_WAIT_S`."""
    return math.ceil(active_replicas * (replica_throughput / 2) * max_acceptable_wait_s)


def plan_capacity(
    requests_per_day: float,
    peak_factor: float,
    replica_throughput: float,
    max_acceptable_wait_s: float,
    query_cache_hit_rate: float = 0.0,
) -> CapacityPlan:
    average_rps = requests_per_day / SECONDS_PER_DAY
    peak_rps = average_rps * peak_factor
    peak_emb_s = peak_embeddings_per_s(requests_per_day, peak_factor, query_cache_hit_rate)
    replicas = worker_replicas(peak_emb_s, replica_throughput)
    queue_depth = max_queue_depth(replicas, replica_throughput, max_acceptable_wait_s)
    return CapacityPlan(
        requests_per_day=requests_per_day,
        average_rps=average_rps,
        peak_rps=peak_rps,
        peak_embeddings_per_s=peak_emb_s,
        worker_replicas=replicas,
        max_queue_depth=queue_depth,
    )
