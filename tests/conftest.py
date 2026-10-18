import asyncio
import time
from typing import Any

import pytest

from app.config import settings
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.persistence.redis_store import redis_store


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

    client = redis_store.get_client()
    try:
        keys = client.keys("research:ratelimit:*")
        if keys:
            client.delete(*keys)
    except Exception:
        pass

    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    yield fake
    llm_client.set_override_provider(None)


@pytest.fixture
def setup_fake_llm(reset_rate_limiter_and_fake_llm):
    """Alias for backwards compatibility with tests expecting setup_fake_llm fixture."""
    return reset_rate_limiter_and_fake_llm


async def wait_for_status(
    client: Any,
    run_id: str,
    expected_statuses: tuple,
    timeout: float = 60.0,
    headers: dict | None = None,
) -> dict:
    """Polls GET /research/{id} with exponential backoff until status matches expected_statuses."""
    start_time = time.time()
    delay = 0.1
    expected_values = {s.value if hasattr(s, "value") else str(s) for s in expected_statuses}
    while time.time() - start_time < timeout:
        status_resp = await client.get(f"/research/{run_id}", headers=headers or {})
        if status_resp.status_code == 200:
            data = status_resp.json()
            status_val = data.get("status")
            if status_val in expected_values or status_val in expected_statuses:
                return data
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 0.5)
    raise TimeoutError(f"Run {run_id} did not reach {expected_statuses} within {timeout}s")
