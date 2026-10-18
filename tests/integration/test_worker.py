import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.models.enums import RunStatus
from app.persistence.redis_store import redis_store
from app.persistence.worker import BackgroundWorker


@pytest.mark.asyncio
@pytest.mark.integration
async def test_worker_resume_interrupted_runs_empty():
    """Validates resume_interrupted_runs when there are no active runs."""
    w = BackgroundWorker()
    redis_store.get_client().delete("research:running_runs")
    count = await w.resume_interrupted_runs()
    assert count == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_worker_resume_interrupted_runs_detects_running():
    """Validates resume_interrupted_runs resumes runs left in incomplete states."""
    w = BackgroundWorker()
    redis_store.get_client().delete("research:running_runs")
    run_id = "test_run_interrupted_worker"

    redis_store.save_run_state(
        run_id,
        {"run_id": run_id, "status": RunStatus.RESEARCHING, "question": "Quantum Cryptography"},
    )

    with patch.object(w, "run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = None
        count = await w.resume_interrupted_runs()
        assert count == 1
        await asyncio.sleep(0.01)
        mock_run.assert_any_call(run_id, None, False)

    redis_store.get_client().delete("research:running_runs")
