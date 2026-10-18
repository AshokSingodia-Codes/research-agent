import asyncio

import pytest

from app.persistence.concurrency import CrossProcessConcurrencyLimiter, InProcessConcurrencyLimiter
from app.persistence.redis_store import redis_store


@pytest.mark.asyncio
@pytest.mark.unit
async def test_in_process_concurrency_limiter():
    """Validates in-process asyncio semaphore strictly caps concurrent workers."""
    limiter = InProcessConcurrencyLimiter(max_concurrent=2)
    active_workers = 0
    max_observed_active = 0

    async def worker():
        nonlocal active_workers, max_observed_active
        async with limiter:
            active_workers += 1
            max_observed_active = max(max_observed_active, active_workers)
            await asyncio.sleep(0.05)
            active_workers -= 1

    await asyncio.gather(*[worker() for _ in range(6)])
    assert max_observed_active <= 2


@pytest.mark.unit
def test_cross_process_concurrency_limiter_fakeredis_notice():
    """
    Validates that CrossProcessConcurrencyLimiter identifies lack of Lua support
    when ResponseError("unknown command 'eval'") occurs and raises NotImplementedError.
    """
    from unittest.mock import MagicMock
    from redis.exceptions import ResponseError

    mock_client = MagicMock()
    mock_client.eval.side_effect = ResponseError("unknown command 'eval'")

    limiter = CrossProcessConcurrencyLimiter(
        key="test:cap:fakeredis", max_concurrent=2, client=mock_client, fallback_on_unsupported=False
    )

    with pytest.raises(NotImplementedError) as exc_info:
        limiter.acquire()

    assert "Lua eval is unsupported" in str(exc_info.value)
