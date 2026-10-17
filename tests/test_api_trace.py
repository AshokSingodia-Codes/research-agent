import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.models.schemas import Citation, EvidenceRecord, ReportSection, ResearchReport
from src.services.validator import validate_report_citations


@pytest.mark.unit
@pytest.mark.asyncio
async def test_citation_validator_filters_hallucinated_sources():
    """Validates that citation validator confirms real sources and flags hallucinated citations."""
    real_evidence = [
        EvidenceRecord(
            claim="Multi-Raft shards data into consensus partitions.",
            source_url="https://arxiv.org/abs/2401.0001",
            domain="arxiv.org",
            snippet="Partitions reduce write bottlenecks."
        )
    ]

    report = ResearchReport(
        topic="Multi-Raft Sharding",
        executive_summary="Summary",
        sections=[ReportSection(heading="Sec 1", content="Content", citation_ids=["[1]", "[2]"])],
        citations=[
            Citation(
                citation_id="[1]",
                source_url="https://arxiv.org/abs/2401.0001",
                domain="arxiv.org",
                claim_supported="Multi-Raft shards data",
                snippet_reference="Partitions reduce write bottlenecks"
            ),
            Citation(
                citation_id="[2]",
                source_url="https://fake-hallucinated-source.com/paper",
                domain="fake-hallucinated-source.com",
                claim_supported="Hallucinated claim",
                snippet_reference="Fake snippet"
            )
        ]
    )

    validated_report, warnings = validate_report_citations(report, real_evidence)

    # Valid citation is preserved and marked verified
    assert len(validated_report.citations) == 1
    assert validated_report.citations[0].citation_id == "[1]"
    assert validated_report.citations[0].is_verified is True

    # Hallucinated citation is filtered out with warning
    assert len(warnings) == 1
    assert "fake-hallucinated-source.com" in warnings[0]


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.asyncio
async def test_fastapi_research_lifecycle_and_trace_endpoints():
    """Validates complete API lifecycle: POST /research -> GET /research/{id} -> GET /research/{id}/trace."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # 2. POST /research (returns run_id immediately)
        post_resp = await client.post(
            "/research",
            json={"topic": "Consensus in Distributed Databases"}
        )
        assert post_resp.status_code == 202
        body = post_resp.json()
        assert "run_id" in body
        run_id = body["run_id"]
        assert body["status"] == "queued"

        # 3. GET /research/{id}
        get_resp = await client.get(f"/research/{run_id}")
        assert get_resp.status_code == 200
        status_body = get_resp.json()
        assert status_body["run_id"] == run_id
        assert status_body["status"] in ("queued", "in_progress", "completed")

        # 4. Wait for background task completion
        for _ in range(100):
            poll_resp = await client.get(f"/research/{run_id}")
            data = poll_resp.json()
            if data["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.25)

        # 5. GET /research/{id}/trace (interview proof debugging trace)
        trace_resp = await client.get(f"/research/{run_id}/trace")
        assert trace_resp.status_code == 200
        trace_data = trace_resp.json()
        assert trace_data["run_id"] == run_id
        assert trace_data["total_steps"] > 0
        assert trace_data["total_latency_ms"] >= 0
        assert len(trace_data["steps"]) == trace_data["total_steps"]

        step_nodes = [s["node_name"] for s in trace_data["steps"]]
        assert "planner" in step_nodes
