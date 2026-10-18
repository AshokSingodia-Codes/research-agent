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


@pytest.mark.redis_real
@pytest.mark.integration
def test_lua_check_and_incr_real_redis():
    """Validates atomic check-and-increment Lua script under real Redis."""
    from app.budget.guard import LUA_CHECK_AND_INCR

    client = redis_store.get_client()
    key = "test:budget:lua_check_incr"
    client.delete(key)

    # 1. Increment 300 with cap 500 -> 300
    res1 = client.eval(LUA_CHECK_AND_INCR, 1, key, 300, 500)
    assert res1 == 300

    # 2. Increment 150 with cap 500 -> 450
    res2 = client.eval(LUA_CHECK_AND_INCR, 1, key, 150, 500)
    assert res2 == 450

    # 3. Increment 100 with cap 500 -> -1 (rejected)
    res3 = client.eval(LUA_CHECK_AND_INCR, 1, key, 100, 500)
    assert res3 == -1

    # Counter remains at 450
    assert int(client.get(key)) == 450
    client.delete(key)


@pytest.mark.redis_real
@pytest.mark.integration
def test_lua_check_and_incr_concurrent_two_clients_real_redis():
    """
    Validates that 20 concurrent charges from 2 distinct Redis clients executing the
    LUA_CHECK_AND_INCR script never exceed the specified ceiling.
    """
    import concurrent.futures
    import redis
    from app.budget.guard import LUA_CHECK_AND_INCR
    from app.config import settings

    client1 = redis_store.get_client()
    client2 = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    key = "test:budget:lua_concurrent_charges"
    client1.delete(key)

    cap = 1000
    charge_amount = 75  # 13 charges * 75 = 975 <= 1000; 14th charge would be 1050 > 1000

    results = []

    def make_charge(client_instance):
        try:
            res = client_instance.eval(LUA_CHECK_AND_INCR, 1, key, charge_amount, cap)
            return res
        except Exception as e:
            return f"error: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(20):
            target_client = client1 if i % 2 == 0 else client2
            futures.append(executor.submit(make_charge, target_client))
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    successful_charges = [r for r in results if isinstance(r, int) and r != -1]
    rejected_charges = [r for r in results if r == -1]

    final_val = int(client1.get(key) or 0)
    assert final_val <= cap
    assert final_val == len(successful_charges) * charge_amount
    assert len(successful_charges) + len(rejected_charges) == 20
    assert len(successful_charges) == cap // charge_amount  # exactly 13 successful

    client1.delete(key)
    client2.close()
