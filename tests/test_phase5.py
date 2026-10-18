import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.events import event_broker
from app.config import settings
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.main import app
from app.models.enums import RunStatus
from app.models.schemas import CitationKey, Finding, Report, ReportSection
from app.services.citation_validator import validate_citations


@pytest.fixture(autouse=True)
def setup_fake_llm():
    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    yield fake
    llm_client.set_override_provider(None)


def test_citation_validator_rejects_fabricated_citations():
    """Validates that citation validator detects and strips fabricated citations not in evidence."""
    verified_findings = [
        Finding(
            id="fnd_real_1",
            sub_question_id="sq_1",
            claim="Raft provides leader leases to avoid split-brain.",
            source_url="https://raft.github.io/raft.pdf",
            domain="raft.github.io",
            snippet="Leader leases enforce quorum safety.",
            retrieval_timestamp=time.time(),
            content_hash="h_real",
        )
    ]

    report = Report(
        topic="Raft Consensus",
        summary="Overview of Raft [1].",
        sections=[
            ReportSection(
                title="Leader Safety",
                content="Raft uses leader leases [1]. It also claims 99.999% throughput without consensus [2].",
                citations=["[1]", "[2]"],
            )
        ],
        citations={
            "[1]": CitationKey(
                key="[1]",
                finding_id="fnd_real_1",
                source_url="https://raft.github.io/raft.pdf",
                domain="raft.github.io",
                claim="Raft provides leader leases.",
            ),
            # Fabricated citation [2] pointing to nonexistent finding
            "[2]": CitationKey(
                key="[2]",
                finding_id="fnd_fake_999",
                source_url="https://hallucinated-source.com/fake.pdf",
                domain="hallucinated-source.com",
                claim="Fabricated claim without evidence.",
            ),
        },
    )

    validated, coverage, warnings = validate_citations(report, verified_findings, threshold=0.80)
    assert coverage == 0.50  # 1 out of 2 valid
    assert "[1]" in validated.citations
    assert "[2]" not in validated.citations  # Fabricated citation stripped
    assert len(warnings) >= 1
    assert "Fabricated citation" in warnings[0]


async def wait_for_status(
    client, run_id: str, expected_statuses: tuple, timeout: float = 90.0, headers: dict | None = None
) -> dict:
    """Polls GET /research/{id} with exponential backoff until status matches expected_statuses."""
    start_time = time.time()
    delay = 0.2
    while time.time() - start_time < timeout:
        status_resp = await client.get(f"/research/{run_id}", headers=headers or {})
        if status_resp.status_code == 200:
            data = status_resp.json()
            if data.get("status") in expected_statuses:
                return data
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 1.0)
    raise TimeoutError(f"Run {run_id} did not reach {expected_statuses} within {timeout}s")


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
            timeout=90.0,
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
async def test_sse_ordering_and_reconnection_playback():
    """Validates SSE event ordering and historical event replay via Last-Event-ID."""
    run_id = f"test_sse_{int(time.time())}"

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
async def test_human_approval_gate_pauses_and_resumes():
    """Validates that require_approval=True pauses graph before write, and POST /approve resumes it."""
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Start run requiring approval
        post_resp = await client.post(
            "/research",
            json={"question": "Fault Tolerance Patterns", "require_approval": True},
            headers=headers,
        )
        assert post_resp.status_code == 202
        run_id = post_resp.json()["run_id"]

        # 2. Wait until graph pauses at awaiting_approval (exponential backoff helper)
        await wait_for_status(
            client,
            run_id,
            (RunStatus.AWAITING_APPROVAL,),
            timeout=90.0,
            headers=headers,
        )

        # 3. Post approval
        approve_resp = await client.post(f"/research/{run_id}/approve", headers=headers)
        assert approve_resp.status_code == 200

        # 4. Wait for resumption to complete (exponential backoff helper)
        resumed_data = await wait_for_status(
            client,
            run_id,
            (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW),
            timeout=90.0,
            headers=headers,
        )
        assert resumed_data["report"] is not None


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
