import asyncio
import logging
from types import TracebackType
from typing import Any

from redis.exceptions import ResponseError

from app.persistence.redis_store import redis_store

logger = logging.getLogger(__name__)

LUA_ACQUIRE_CONCURRENCY = """
local key = KEYS[1]
local max_cap = tonumber(ARGV[1])
local current = tonumber(redis.call('GET', key) or '0')
if current < max_cap then
    redis.call('INCR', key)
    return 1
else
    return 0
end
"""

LUA_RELEASE_CONCURRENCY = """
local key = KEYS[1]
local current = tonumber(redis.call('GET', key) or '0')
if current > 0 then
    redis.call('DECR', key)
    return 1
else
    return 0
end
"""


class InProcessConcurrencyLimiter:
    """In-process concurrency limiter using asyncio.Semaphore."""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        self._semaphore.release()

    @property
    def current_available(self) -> int:
        return self._semaphore._value


class CrossProcessConcurrencyLimiter:
    """
    Cross-process concurrency limiter using Redis Lua scripting for atomic acquire/release.
    Note: Requires real Redis with Lua scripting support (fakeredis without lupa does not support eval).
    """

    def __init__(self, key: str, max_concurrent: int, client: Any | None = None):
        self.key = key
        self.max_concurrent = max_concurrent
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        return redis_store.get_client()

    def acquire(self) -> bool:
        """Atomically checks and increments concurrency counter if below max_concurrent."""
        try:
            result = self.client.eval(LUA_ACQUIRE_CONCURRENCY, 1, self.key, self.max_concurrent)
            return bool(result == 1)
        except ResponseError as e:
            if "unknown command 'eval'" in str(e).lower():
                raise NotImplementedError("Lua eval is unsupported in this Redis client (fakeredis without lupa).")
            raise e

    def release(self) -> bool:
        """Atomically decrements concurrency counter if above 0."""
        try:
            result = self.client.eval(LUA_RELEASE_CONCURRENCY, 1, self.key)
            return bool(result == 1)
        except ResponseError as e:
            if "unknown command 'eval'" in str(e).lower():
                raise NotImplementedError("Lua eval is unsupported in this Redis client (fakeredis without lupa).")
            raise e
