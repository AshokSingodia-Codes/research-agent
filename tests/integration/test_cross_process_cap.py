import pytest

from app.persistence.concurrency import CrossProcessConcurrencyLimiter
from app.persistence.redis_store import redis_store


@pytest.mark.redis_real
@pytest.mark.integration
def test_cross_process_concurrency_cap_real_redis():
    """
    Validates atomic cross-process concurrency capping using Redis Lua scripts.
    Requires real Redis supporting EVAL (skipped on local fakeredis, executed in CI).
    """
    client = redis_store.get_client()
    key = "test:concurrency:cross_process_cap"
    client.delete(key)

    limiter = CrossProcessConcurrencyLimiter(key=key, max_concurrent=2, client=client)

    # 1. Acquire slot 1
    assert limiter.acquire() is True
    # 2. Acquire slot 2
    assert limiter.acquire() is True
    # 3. Third acquire exceeds cap of 2 -> rejected
    assert limiter.acquire() is False

    # 4. Release one slot
    assert limiter.release() is True
    # 5. Acquire succeeds again
    assert limiter.acquire() is True
    # 6. Exceeds cap again
    assert limiter.acquire() is False

    # Cleanup
    limiter.release()
    limiter.release()
    client.delete(key)
