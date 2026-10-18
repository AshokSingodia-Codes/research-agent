import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.models.enums import RunStatus
from tests.conftest import wait_for_status


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_e2e_full_research_workflow():
    """
    End-to-End full workflow test:
    1. POST /research to enqueue research topic
    2. Poll status until completion
    3. Retrieve final structured report and citation verification
    4. Retrieve execution trace and metrics
    """
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Enqueue research run
        topic = "Zero Knowledge Proofs in Distributed Systems"
        resp = await client.post(
            "/research",
            json={"question": topic, "require_approval": False},
            headers=headers,
        )
        assert resp.status_code == 202
        run_data = resp.json()
        assert "run_id" in run_data
        run_id = run_data["run_id"]

        # 2. Wait for background execution to complete
        completed_data = await wait_for_status(
            client,
            run_id,
            (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW),
            timeout=60.0,
            headers=headers,
        )
        assert completed_data["status"] in ("completed", "needs_review")

        # 3. Verify final report
        report = completed_data.get("report")
        assert report is not None
        assert "topic" in report
        assert "sections" in report
        assert len(report["sections"]) > 0

        # 4. Verify execution trace
        trace_resp = await client.get(f"/research/{run_id}/trace", headers=headers)
        assert trace_resp.status_code == 200
        trace = trace_resp.json()
        assert trace["run_id"] == run_id
        assert trace["total_steps"] > 0
        assert len(trace["steps"]) == trace["total_steps"]

        # 5. Verify Prometheus / JSON metrics endpoint
        metrics_resp = await client.get("/metrics", headers=headers)
        assert metrics_resp.status_code == 200
        assert "total_requests" in metrics_resp.text or "http_requests" in metrics_resp.text
