import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_research_runs_isolation():
    """Validates that concurrent research runs maintain completely isolated state and run IDs."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Launch 2 concurrent runs
        topics = ["Topic A: Raft Consensus", "Topic B: Paxos Protocol"]

        responses = await asyncio.gather(*[client.post("/research", json={"topic": t}) for t in topics])

        run_ids = [r.json()["run_id"] for r in responses]
        assert len(set(run_ids)) == len(topics)  # All run IDs must be unique

        # Poll status for all runs until completed (wait up to 30s)
        for _ in range(120):
            status_responses = await asyncio.gather(*[client.get(f"/research/{rid}") for rid in run_ids])
            all_done = all(sr.json()["status"] in ("completed", "failed") for sr in status_responses)
            if all_done:
                break
            await asyncio.sleep(0.25)

        # Inspect that all completed with isolated reports
        for idx, rid in enumerate(run_ids):
            sr = await client.get(f"/research/{rid}")
            data = sr.json()
            assert data["status"] == "completed"
            assert data["report"] is not None
            assert data["report"]["topic"] == topics[idx]
