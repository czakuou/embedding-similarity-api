"""Measures worker throughput (T) and batch latency with the real model.

Run inside the worker container/environment (needs the `worker` dependency
group: torch + sentence-transformers). Prints p50/p95 batch latency per
batch size, T (embeddings/s), and a replica table computed from the measured
T -- results are local to this machine, re-run on the target SKU.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass

from similarity.capacity import max_queue_depth as compute_max_queue_depth
from similarity.capacity import peak_embeddings_per_s, worker_replicas
from similarity.encoder import SentenceTransformerEncoder

QUERY_TEXT = "fox over dog"  # ~5 tokens, representative of a short query
DOC_TEXT = ("The quick brown fox jumps over the lazy dog. " * 100)[:4096]  # pre-cut length; the
# model truncates internally at 256 tokens regardless of how much of this is fed in.

BATCH_SIZES = (1, 8, 16, 32, 64)
WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 20


@dataclass(frozen=True, slots=True)
class BatchMeasurement:
    batch_size: int
    p50_s: float
    p95_s: float
    throughput_emb_s: float


def percentile(samples: list[float], pct: float) -> float:
    """pct in (0, 100). Uses inclusive quantiles since these are the samples
    themselves, not a population estimate.
    """
    if len(samples) == 1:
        return samples[0]
    quantiles = statistics.quantiles(samples, n=100, method="inclusive")
    return quantiles[round(pct) - 1]


def measure_batch(
    encoder: SentenceTransformerEncoder, batch_size: int, iterations: int
) -> list[float]:
    """Returns the wall-clock seconds for each iteration of encode+similarity
    on a batch of `batch_size` query/document pairs.
    """
    queries = [QUERY_TEXT] * batch_size
    docs = [DOC_TEXT] * batch_size
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        texts = [text for pair in zip(queries, docs, strict=True) for text in pair]
        embeddings = encoder.encode(texts)
        query_embeddings = embeddings[0::2]
        document_embeddings = embeddings[1::2]
        encoder.similarity_pairwise(query_embeddings, document_embeddings)
        durations.append(time.perf_counter() - started)
    return durations


def benchmark_batch_sizes(
    encoder: SentenceTransformerEncoder, batch_sizes: tuple[int, ...], iterations: int
) -> list[BatchMeasurement]:
    results = []
    for batch_size in batch_sizes:
        measure_batch(encoder, batch_size, WARMUP_ITERATIONS)  # warm up, discard
        durations = measure_batch(encoder, batch_size, iterations)
        p50 = percentile(durations, 50)
        p95 = percentile(durations, 95)
        # T is embeddings/s counting both halves of the pair: peak_embeddings_per_s
        # multiplies peak_rps by 2 for exactly this reason -- one analysis costs one
        # query embedding plus one document embedding.
        throughput = (batch_size * 2) / p50
        results.append(
            BatchMeasurement(
                batch_size=batch_size, p50_s=p50, p95_s=p95, throughput_emb_s=throughput
            )
        )
    return results


def measure_query_vs_document_cost(
    encoder: SentenceTransformerEncoder, iterations: int
) -> tuple[float, float]:
    """Returns (query_only_p50_s, document_only_p50_s) for a single-item batch,
    isolating the cost of a ~5-token query from a 256-token (truncated) document.
    """
    query_durations = []
    doc_durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        encoder.encode([QUERY_TEXT])
        query_durations.append(time.perf_counter() - started)

        started = time.perf_counter()
        encoder.encode([DOC_TEXT])
        doc_durations.append(time.perf_counter() - started)
    return percentile(query_durations, 50), percentile(doc_durations, 50)


def print_replica_table(best_measurement: BatchMeasurement, max_acceptable_wait_s: float) -> None:
    print()
    print(f"Replica table (T = {best_measurement.throughput_emb_s:.1f} emb/s, measured locally,")
    print("peak factor = 10x, cache hit rate H = 0):")
    print(f"{'Requests/day':>15} | {'Peak emb/s':>12} | {'Replicas':>8} | {'Queue depth':>11}")
    for requests_per_day in (100_000, 500_000, 1_000_000, 5_000_000):
        peak_emb_s = peak_embeddings_per_s(requests_per_day, peak_factor=10)
        replicas = worker_replicas(peak_emb_s, best_measurement.throughput_emb_s)
        queue_depth = compute_max_queue_depth(
            replicas, best_measurement.throughput_emb_s, max_acceptable_wait_s
        )
        print(f"{requests_per_day:>15,} | {peak_emb_s:>12.1f} | {replicas:>8} | {queue_depth:>11}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=MEASURED_ITERATIONS,
        help="measured iterations per batch size",
    )
    parser.add_argument(
        "--max-acceptable-wait-s",
        type=float,
        default=5.0,
        help="MAX_ACCEPTABLE_WAIT_S for the queue-depth formula (SLA default: 5s)",
    )
    args = parser.parse_args()

    model_name = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    model_revision = os.environ.get("MODEL_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    torch_num_threads = int(os.environ.get("TORCH_NUM_THREADS", "2"))

    import torch

    torch.set_num_threads(torch_num_threads)

    print(f"Loading {model_name}@{model_revision} (TORCH_NUM_THREADS={torch_num_threads})...")
    encoder = SentenceTransformerEncoder(model_name, model_revision)
    encoder.warm_up()

    print(f"\n{'Batch size':>10} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'T (emb/s)':>10}")
    measurements = benchmark_batch_sizes(encoder, BATCH_SIZES, args.iterations)
    for m in measurements:
        print(
            f"{m.batch_size:>10} | {m.p50_s * 1000:>10.1f} | {m.p95_s * 1000:>10.1f} "
            f"| {m.throughput_emb_s:>10.1f}"
        )

    query_p50, doc_p50 = measure_query_vs_document_cost(encoder, args.iterations)
    print(
        f"\nSingle-item cost: query p50 = {query_p50 * 1000:.1f} ms, "
        f"document p50 = {doc_p50 * 1000:.1f} ms"
    )

    best = max(measurements, key=lambda m: m.throughput_emb_s)
    worst_p95_s = max(m.p95_s for m in measurements)
    print(
        f"\nWorst observed p95 batch time (across all batch sizes): {worst_p95_s * 1000:.1f} ms"
        f"\nVISIBILITY_TIMEOUT_S must exceed this -- set from the measured p95 batch time; "
        f"round up generously for headroom under real load."
    )
    print_replica_table(best, args.max_acceptable_wait_s)


if __name__ == "__main__":
    main()
