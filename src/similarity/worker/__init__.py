"""Worker slice: consumer loop, greedy batching, per-item isolation, attempts,
reclaim, heartbeat, warm-up, SIGTERM.

A package from the start (not split later): the loop, batch processing, and
item parsing are naturally separate concerns and together would exceed a
reasonable single-file size.
"""

from __future__ import annotations

from similarity.worker.batch import process_batch
from similarity.worker.items import QueueItem

__all__ = ["QueueItem", "process_batch"]
