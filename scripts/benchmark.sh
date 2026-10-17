#!/usr/bin/env bash
# Runs benchmark.py inside the worker image, with the same CPU limit the
# worker service uses in docker-compose.yml (SPEC §7.3: measure under the
# same CPU limit as production, since torch throughput depends on it).
set -euo pipefail

cd "$(dirname "$0")/.."

docker build --target worker -t similarity-worker:benchmark .

docker run --rm \
  --cpus 2 \
  -e TORCH_NUM_THREADS=2 \
  -e OMP_NUM_THREADS=2 \
  -e HF_HUB_OFFLINE=1 \
  -v "$(pwd)/scripts:/app/scripts:ro" \
  similarity-worker:benchmark \
  python /app/scripts/benchmark.py "$@"
