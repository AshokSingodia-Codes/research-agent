import asyncio
import json
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.events import event_broker, register_stream_queue
from app.config import settings
from app.main import app
from app.persistence.redis_store import redis_store


@pytest.mark.asyncio
async def test_sse_progress_streaming_lifecycle():
    """Validates real-time Server-Sent Events (SSE) streaming endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a mock run state in Redis
        run_id = "test_sse_run_001"
        redis_store.save_run_state(
            run_id=run_id, status="in_progress", current_node="planner", total_tokens=100, total_subquestions=2
        )

        queue = register_stream_queue(run_id)

        # Background coroutine to push events into the stream queue
        async def push_events():
            await asyncio.sleep(0.05)
            await queue.put({"type": "status", "data": {"node": "planner", "message": "Planning..."}})
            await asyncio.sleep(0.05)
            await queue.put({"type": "status", "data": {"node": "researcher", "message": "Searching..."}})
            await asyncio.sleep(0.05)
            await queue.put({"type": "complete", "data": {"status": "completed"}})
            await queue.put(None)  # stream end

        asyncio.create_task(push_events())

        # Read stream
        events_received = []
        async with client.stream(
            "GET", f"/research/{run_id}/stream", headers={"X-API-Key": settings.api_key}
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    raw_data = line[len("data:") :].strip()
                    payload = json.loads(raw_data)
                    events_received.append(payload)
                if len(events_received) >= 4:
                    break

        assert len(events_received) >= 3
        types = [e.get("type") for e in events_received]
        assert "init" in types
        assert "status" in types


@pytest.mark.asyncio
async def test_sse_ordering_and_reconnection_playback():
    """Validates SSE event ordering and historical event replay via Last-Event-ID."""
    run_id = f"test_sse_recon_{int(time.time() * 1000)}"

    # Publish 3 sequential events to Redis
    event_broker.publish_event(run_id, "node_started", {"node": "planner"})
    event_broker.publish_event(run_id, "finding_added", {"claim": "Claim 1"})
    event_broker.publish_event(run_id, "completed", {"status": "completed"})

    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key, "Last-Event-ID": "1"}

    # Connect to SSE stream starting after event ID 1
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/research/{run_id}/events", headers=headers)
        assert resp.status_code == 200
        text = resp.text
        # Event 1 was skipped; Events 2 and 3 must be present in exact order
        assert "event: finding_added" in text
        assert "event: completed" in text
        idx2 = text.find("event: finding_added")
        idx3 = text.find("event: completed")
        assert idx2 < idx3, "SSE events must be ordered chronologically"


@pytest.mark.asyncio
async def test_sse_stream_token_acquisition_and_streaming(monkeypatch):
    """Validates acquiring a run-scoped stream token via POST /stream-token and connecting via ?token=."""
    test_key = "sec-test-key-32chars-for-test!!"
    monkeypatch.setattr(settings, "api_key", test_key)
    monkeypatch.setattr(settings, "stream_token_secret", "secret-stream-token-xyz")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        run_id = f"test_token_sse_{int(time.time() * 1000)}"
        redis_store.save_run_state(run_id=run_id, status="in_progress")

        # 1. Acquire token with X-API-Key header
        resp = await client.post(
            f"/research/{run_id}/stream-token",
            headers={"X-API-Key": test_key},
        )
        assert resp.status_code == 200
        token_data = resp.json()
        assert "token" in token_data
        token = token_data["token"]
        assert token_data["expires_in"] == 60

        # 2. Connect with ?token= (no header)
        queue = register_stream_queue(run_id)
        await queue.put({"type": "complete", "data": {"status": "completed"}})
        await queue.put(None)
        async with client.stream("GET", f"/research/{run_id}/stream?token={token}") as stream_resp:
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers["content-type"]
            lines = [line async for line in stream_resp.aiter_lines() if line]
            assert any("complete" in line for line in lines)

        # 3. Connecting again with the same token fails (single use)
        async with client.stream("GET", f"/research/{run_id}/stream?token={token}") as replay_resp:
            assert replay_resp.status_code == 403

        # 4. Connecting with ?api_key= is rejected (no query api_key leakage)
        async with client.stream(
            "GET", f"/research/{run_id}/stream?api_key={test_key}"
        ) as bad_resp:
            assert bad_resp.status_code == 401
