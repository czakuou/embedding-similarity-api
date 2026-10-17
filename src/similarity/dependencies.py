"""FastAPI Depends() getters for objects stored on app.state at startup.

This is the one place that reads request.app.state; slice handlers declare
these as Annotated parameters instead (SPEC §3.5: dependencies are passed
as function arguments -- this just lets FastAPI supply them).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated

import redis.asyncio as redis
from fastapi import Depends, Request
from limits import RateLimitItem
from limits.aio.strategies import FixedWindowRateLimiter

from similarity.config import ApiSettings

Clock = Callable[[], datetime]


def get_settings(request: Request) -> ApiSettings:
    settings: ApiSettings = request.app.state.settings
    return settings


def get_clock() -> Clock:
    """Default clock: real wall-clock time. SPEC §3.5: clock is injected so
    404/410 logic is deterministic in tests, which override this dependency
    (app.dependency_overrides) instead of sleeping for real TTLs.
    """
    return lambda: datetime.now(UTC)


def get_redis(request: Request) -> redis.Redis:
    client: redis.Redis = request.app.state.redis
    return client


def get_rate_limiter(request: Request) -> FixedWindowRateLimiter:
    limiter: FixedWindowRateLimiter = request.app.state.rate_limiter
    return limiter


def get_rate_limit_item(request: Request) -> RateLimitItem:
    item: RateLimitItem = request.app.state.rate_limit_item
    return item


SettingsDep = Annotated[ApiSettings, Depends(get_settings)]
RedisDep = Annotated[redis.Redis, Depends(get_redis)]
RateLimiterDep = Annotated[FixedWindowRateLimiter, Depends(get_rate_limiter)]
RateLimitItemDep = Annotated[RateLimitItem, Depends(get_rate_limit_item)]
ClockDep = Annotated[Clock, Depends(get_clock)]
