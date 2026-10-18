"""POST slice: validation, pre-cut, rate limiting, backpressure, MULTI enqueue.

Split into a package because the slice outgrew a reasonable single-file size,
not because of a layering concern: validation.py (pure), outcomes.py (pure sum
type + response mapping), enqueue.py (Redis I/O), router.py (HTTP handler).
Other slices still only reach into this package's public API below.
"""

from __future__ import annotations

from similarity.submit.enqueue import make_redis_storage, rate_limit_item
from similarity.submit.router import router

__all__ = ["make_redis_storage", "rate_limit_item", "router"]
