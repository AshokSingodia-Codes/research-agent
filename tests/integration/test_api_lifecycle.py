import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.models.enums import RunStatus
from tests.conftest import wait_for_status


@pytest.mark.asyncio
async def test_full_api_research_lifecycle_and_trace():
    """Validates full REST API lifecycle: POST /research, GET /status, and GET /trace."""
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Post new research run
        post_resp = await client.post(
            "/research",
            json={"question": "Distributed Consensus Under Asymmetric Network Latency"},
            headers=headers,
        )
        assert post_resp.status_code == 202
        run_data = post_resp.json()
        run_id = run_data["run_id"]
        assert run_id.startswith("run_")

        # 2. Poll until workflow completes (exponential backoff helper)
        data = await wait_for_status(
            client,
            run_id,
            (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW),
            timeout=60.0,
            headers=headers,
        )
        assert data["report"] is not None
        assert len(data["report"]["sections"]) >= 1

        # 3. Inspect audit trace endpoint
        trace_resp = await client.get(f"/research/{run_id}/trace", headers=headers)
        assert trace_resp.status_code == 200
        trace_data = trace_resp.json()
        assert trace_data["run_id"] == run_id
        assert trace_data["total_steps"] >= 1
        assert trace_data["total_tokens"] > 0
        assert len(trace_data["steps"]) >= 1
        step0 = trace_data["steps"][0]
        assert "agent" in step0
        assert "provider" in step0


@pytest.mark.asyncio
async def test_post_research_request_schema_alignment():
    """Validates POST /research with question, topic, both, and neither (422)."""
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. POST with question
        r1 = await client.post("/research", json={"question": "Valid Question One"}, headers=headers)
        assert r1.status_code == 202
        assert r1.json()["run_id"].startswith("run_")

        # 2. POST with topic
        r2 = await client.post("/research", json={"topic": "Valid Topic Two"}, headers=headers)
        assert r2.status_code == 202
        assert r2.json()["run_id"].startswith("run_")

        # 3. POST with both
        r3 = await client.post(
            "/research", json={"question": "Question Three", "topic": "Topic Three"}, headers=headers
        )
        assert r3.status_code == 202
        assert r3.json()["run_id"].startswith("run_")

        # 4. POST with neither -> 422
        r4 = await client.post("/research", json={}, headers=headers)
        assert r4.status_code == 422

        # Clean up / cancel dispatched runs so background workers do not linger
        for r in (r1, r2, r3):
            await client.post(f"/research/{r.json()['run_id']}/cancel", headers=headers)


@pytest.mark.asyncio
async def test_api_auth_and_metrics():
    """Validates API authentication rejection on missing keys and health/metrics responses."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Missing key should be unauthorized
        unauth = await client.post("/research", json={"question": "Test Auth"})
        assert unauth.status_code in (401, 403)

        # Health endpoint should be publicly accessible
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] in ("healthy", "degraded")

        # Metrics endpoint should return Prometheus format text
        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "research_runs_total" in metrics_resp.text


@pytest.mark.asyncio
async def test_dashboard_ui_html_structure():
    """Validates that GET / delivers the rich Tailwind dashboard HTML with all required interactive element IDs."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text

        # Verify key interactive elements exist
        assert 'id="topicInput"' in html
        assert 'id="runBtn"' in html
        assert 'id="progressBox"' in html
        assert 'id="progressBar"' in html
        assert 'id="resultsCard"' in html
        assert 'id="reportPane"' in html
        assert 'id="tracePane"' in html
        assert 'id="node-planner"' in html
        assert 'id="node-researcher"' in html
        assert 'id="node-supervisor"' in html
        assert 'id="node-writer"' in html

        # Verify Tailwind and font imports
        assert "cdn.tailwindcss.com" in html
        assert "JetBrains+Mono" in html
        assert "Outfit" in html


@pytest.mark.asyncio
async def test_static_assets_delivery():
    """Validates delivery of static CSS and JS files."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Check styles.css
        css_resp = await client.get("/static/css/styles.css")
        assert css_resp.status_code == 200
        assert "text/css" in css_resp.headers["content-type"]
        assert "--bg-primary" in css_resp.text

        # Check app.js
        js_resp = await client.get("/static/js/app.js")
        assert js_resp.status_code == 200
        assert (
            "application/javascript" in js_resp.headers["content-type"]
            or "text/javascript" in js_resp.headers["content-type"]
        )
        assert "startResearch" in js_resp.text


@pytest.mark.asyncio
async def test_fastapi_research_lifecycle_and_trace_endpoints():
    """Validates complete API lifecycle: POST /research -> GET /research/{id} -> GET /research/{id}/trace."""
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] in ("healthy", "degraded")

        # 2. POST /research (returns run_id immediately)
        post_resp = await client.post(
            "/research",
            json={"topic": "Consensus in Distributed Databases"},
            headers=headers,
        )
        assert post_resp.status_code == 202
        body = post_resp.json()
        assert "run_id" in body
        run_id = body["run_id"]
        assert body["status"] == "queued"

        # 3. GET /research/{id}
        get_resp = await client.get(f"/research/{run_id}", headers=headers)
        assert get_resp.status_code == 200
        status_body = get_resp.json()
        assert status_body["run_id"] == run_id

        # 4. Wait for background task completion
        data = await wait_for_status(
            client,
            run_id,
            (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW),
            timeout=60.0,
            headers=headers,
        )
        assert data["status"] in ("completed", "needs_review")

        # 5. GET /research/{id}/trace
        trace_resp = await client.get(f"/research/{run_id}/trace", headers=headers)
        assert trace_resp.status_code == 200
        trace_data = trace_resp.json()
        assert trace_data["run_id"] == run_id
        assert trace_data["total_steps"] > 0
        assert len(trace_data["steps"]) == trace_data["total_steps"]
