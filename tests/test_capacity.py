from __future__ import annotations

from similarity.capacity import (
    peak_embeddings_per_s,
    plan_capacity,
    worker_replicas,
)


def test_given_500k_requests_per_day_when_computing_peak_embeddings_then_matches_spec() -> None:
    # SPEC §7.1 worked example: 500k/day, peak factor 10, H=0 -> 116 emb/s.
    result = peak_embeddings_per_s(requests_per_day=500_000, peak_factor=10)
    assert round(result) == 116


def test_given_full_cache_hit_rate_when_computing_peak_embeddings_then_halved() -> None:
    no_cache = peak_embeddings_per_s(
        requests_per_day=500_000, peak_factor=10, query_cache_hit_rate=0.0
    )
    full_cache = peak_embeddings_per_s(
        requests_per_day=500_000, peak_factor=10, query_cache_hit_rate=1.0
    )
    assert full_cache == no_cache / 2


def test_given_spec_worked_example_values_when_computing_replicas_then_matches_table() -> None:
    # SPEC §7.2 table, placeholder T=200: 100k->2, 500k->2, 1M->3, 5M->10.
    cases = {100_000: 2, 500_000: 2, 1_000_000: 3, 5_000_000: 10}
    for requests_per_day, expected_replicas in cases.items():
        peak_emb_s = peak_embeddings_per_s(requests_per_day, peak_factor=10)
        assert worker_replicas(peak_emb_s, replica_throughput=200) == expected_replicas


def test_given_known_inputs_when_planning_capacity_then_fields_are_internally_consistent() -> None:
    plan = plan_capacity(
        requests_per_day=500_000,
        peak_factor=10,
        replica_throughput=200,
        max_acceptable_wait_s=5,
    )
    assert plan.average_rps == 500_000 / 86_400
    assert plan.peak_rps == plan.average_rps * 10
    assert plan.worker_replicas == 2
    assert plan.max_queue_depth > 0
