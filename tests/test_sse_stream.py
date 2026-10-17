import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.graph.redis_store import redis_store
from src.services.runner import register_stream_queue


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.asyncio
async def test_sse_progress_streaming_lifecycle():
    """Validates real-time Server-Sent Events (SSE) streaming endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a mock run state in Redis
        run_id = "test_sse_run_001"
        redis_store.save_run_state(
            run_id=run_id,
            status="in_progress",
            current_node="planner",
            total_tokens=100,
            total_subquestions=2
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
        async with client.stream("GET", f"/research/{run_id}/stream") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    raw_data = line[len("data:"):].strip()
                    payload = json.loads(raw_data)
                    events_received.append(payload)
                if len(events_received) >= 4:
                    break

        assert len(events_received) >= 3
        types = [e.get("type") for e in events_received]
        assert "init" in types
        assert "status" in types
