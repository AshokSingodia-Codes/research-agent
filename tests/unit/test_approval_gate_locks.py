import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes import _resume_approved_workflow
from app.config import settings
from app.main import app
from app.models.enums import RunStatus
from app.persistence.redis_store import redis_store


@pytest.mark.unit
def test_owner_checked_lock_release():
    """Validates that a non-owner cannot release a distributed lock."""
    run_id = f"test_run_owner_lock_{uuid.uuid4().hex[:6]}"
    # Acquire with owner token
    token_a = redis_store.acquire_lock(run_id, timeout_seconds=10.0)
    assert token_a is not None

    # Attempt to release with wrong token B -> returns False
    released_b = redis_store.release_lock(run_id, token="wrong_token_b")
    assert released_b is False
    # Lock is still held: cannot be acquired
    assert redis_store.acquire_lock(run_id, timeout_seconds=10.0) is None

    # Attempt to release without token from non-owner (clear local tracking to simulate another process)
    saved = redis_store._owned_locks.pop(run_id, None)
    released_none = redis_store.release_lock(run_id, token=None)
    assert released_none is False
    redis_store._owned_locks[run_id] = saved

    # Release with legitimate token A -> returns True
    released_a = redis_store.release_lock(run_id, token=token_a)
    assert released_a is True

    # Now lock can be acquired again
    token_c = redis_store.acquire_lock(run_id, timeout_seconds=10.0)
    assert token_c is not None
    redis_store.release_lock(run_id, token=token_c)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_double_approve_starts_exactly_one_resume():
    """Validates that duplicate approve calls atomically transition once and start NO second resume."""
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}
    run_id = f"test_run_double_approve_{uuid.uuid4().hex[:6]}"

    # Set run in awaiting_approval state
    redis_store.save_run_state(
        run_id,
        {"run_id": run_id, "status": RunStatus.AWAITING_APPROVAL, "topic": "Consensus"},
    )

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. First approval: transitions awaiting_approval -> running
        resp1 = await client.post(f"/research/{run_id}/approve", headers=headers)
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "approved"

        # 2. Second approval: state is already running
        resp2 = await client.post(f"/research/{run_id}/approve", headers=headers)
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "running"

        # Exactly one resume was dispatched to stream
        redis_client = redis_store.get_client()
        entries = redis_client.xrange("research:jobs", "-", "+")
        resume_entries = [e for e in entries if e[1].get("run_id") == run_id and e[1].get("type") == "resume"]
        assert len(resume_entries) == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_resume_while_another_holder_owns_lock_fails_typed():
    """Validates that resume waits/fails typed when another holder owns the lock, never stealing it."""
    run_id = f"test_run_lock_held_{uuid.uuid4().hex[:6]}"
    # External worker holds lock with foreign token
    foreign_token = redis_store.acquire_lock(run_id, timeout_seconds=30.0)
    assert foreign_token is not None

    with patch("app.api.routes.build_research_graph") as mock_build_graph:
        # Calling resume should fail to acquire lock and exit without executing the graph
        await _resume_approved_workflow(run_id)
        mock_build_graph.assert_not_called()

    # Verify foreign token is still the legitimate owner of the lock
    client = redis_store.get_client()
    assert client.get(f"research:lock:{run_id}") == foreign_token

    # Cleanup
    redis_store.release_lock(run_id, token=foreign_token)
