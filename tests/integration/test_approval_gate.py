import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.models.enums import RunStatus
from tests.conftest import wait_for_status


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
            timeout=60.0,
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
            timeout=60.0,
            headers=headers,
        )
        assert resumed_data["report"] is not None
