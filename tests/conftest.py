import asyncio
import os
import time
from typing import Any

import pytest

from app.config import settings
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.persistence.redis_store import redis_store

_REDIS_REACHABLE: bool | None = None


def is_redis_reachable() -> bool:
    global _REDIS_REACHABLE
    if _REDIS_REACHABLE is not None:
        return _REDIS_REACHABLE

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url or os.environ.get("USE_FAKE_REDIS", "").lower() in ("true", "1", "yes"):
        _REDIS_REACHABLE = False
        return False

    try:
        import redis

        client = redis.Redis.from_url(redis_url, socket_timeout=1.0, decode_responses=True)
        client.ping()
        _REDIS_REACHABLE = True
    except Exception:
        _REDIS_REACHABLE = False
    return _REDIS_REACHABLE


def pytest_runtest_setup(item):
    if "redis_real" in item.keywords:
        if not is_redis_reachable():
            pytest.skip(
                "Skipping redis_real test: Real Redis instance is not available at REDIS_URL. "
                "Set REDIS_URL to a running Redis server or run in CI to execute this test."
            )


def pytest_collection_modifyitems(items):
    """
    Auto-applies markers based on test folder path:
    tests/unit -> unit
    tests/integration -> integration
    tests/chaos -> chaos
    tests/e2e -> e2e
    """
    for item in items:
        if "live" in item.keywords:
            continue
        fspath = str(item.fspath).replace("\\", "/")
        if "/tests/unit/" in fspath:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in fspath:
            item.add_marker(pytest.mark.integration)
        elif "/tests/chaos/" in fspath:
            item.add_marker(pytest.mark.chaos)
        elif "/tests/e2e/" in fspath:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(autouse=True)
def reset_rate_limiter_and_fake_llm(monkeypatch):
    """
    Resets Redis rate limiter keys and installs FakeProvider for deterministic test execution.
    Prevents API tests from hitting 10 req/min limit.
    """
    monkeypatch.setattr(settings, "mock_llm", "true")
    monkeypatch.setattr(settings, "mock_search", "true")
    monkeypatch.setattr(settings, "run_worker_inprocess", True)

    client = redis_store.get_client()
    try:
        keys = (
            list(client.keys("research:ratelimit:*"))
            + list(client.keys("budget:*"))
            + list(client.keys("research:worker:*"))
            + list(client.keys("research:cancel:*"))
        )
        if keys:
            client.delete(*keys)
        # Flush or trim stream
        client.delete("research:jobs")
        client.delete("research:jobs:dlq")
    except Exception:
        pass

    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    yield fake
    llm_client.set_override_provider(None)
    global _INPROCESS_WORKER, _INPROCESS_TASK
    if _INPROCESS_WORKER is not None:
        _INPROCESS_WORKER.stop()
        _INPROCESS_WORKER.unregister_fake_worker()
        _INPROCESS_WORKER = None
        _INPROCESS_TASK = None


@pytest.fixture
def setup_fake_llm(reset_rate_limiter_and_fake_llm):
    """Alias for backwards compatibility with tests expecting setup_fake_llm fixture."""
    return reset_rate_limiter_and_fake_llm


_INPROCESS_WORKER: Any = None
_INPROCESS_TASK: asyncio.Task | None = None
_INPROCESS_WORKER_LOCK = asyncio.Lock()


async def _ensure_inprocess_worker():
    global _INPROCESS_WORKER, _INPROCESS_TASK
    if not getattr(settings, "run_worker_inprocess", False):
        return
    async with _INPROCESS_WORKER_LOCK:
        if _INPROCESS_TASK is not None and not _INPROCESS_TASK.done():
            return
        from app.worker import ResearchStreamWorker

        _INPROCESS_WORKER = ResearchStreamWorker(
            consumer_name=f"inprocess-test-{os.getpid()}-{int(time.time() * 1000)}"
        )
        _INPROCESS_TASK = asyncio.create_task(_INPROCESS_WORKER.run_loop())



async def wait_for_status(
    client: Any,
    run_id: str,
    expected_statuses: tuple,
    timeout: float = 60.0,
    headers: dict | None = None,
) -> dict:
    """Polls GET /research/{id} with exponential backoff until status matches expected_statuses."""
    await _ensure_inprocess_worker()
    start_time = time.time()
    delay = 0.1
    expected_values = {s.value if hasattr(s, "value") else str(s) for s in expected_statuses}
    while time.time() - start_time < timeout:
        if _INPROCESS_TASK is not None and _INPROCESS_TASK.done():
            exc = _INPROCESS_TASK.exception()
            if exc:
                raise RuntimeError(f"Background worker task failed with: {exc}") from exc

        status_resp = await client.get(f"/research/{run_id}", headers=headers or {})
        if status_resp.status_code == 200:
            data = status_resp.json()
            status_val = data.get("status")
            if status_val in expected_values or status_val in expected_statuses:
                return data
        elif status_resp.status_code == 429:
            await asyncio.sleep(0.5)
            continue
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 1.0)
    raise TimeoutError(f"Run {run_id} did not reach {expected_statuses} within {timeout}s (last={status_val})")




