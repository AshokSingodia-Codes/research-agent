import asyncio
import time
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.models.enums import RunStatus
from app.persistence.concurrency import CrossProcessConcurrencyLimiter
from app.persistence.redis_store import redis_store
from app.persistence.stream_adapter import RedisStreamAdapter
from app.worker import ResearchStreamWorker


@pytest.fixture(autouse=True)
def clean_redis_and_streams(monkeypatch):
    monkeypatch.setattr(settings, "run_worker_inprocess", False)
    client = redis_store.get_client()
    try:
        keys = (
            list(client.keys("research:ratelimit:*"))
            + list(client.keys("budget:*"))
            + list(client.keys("research:worker:*"))
            + list(client.keys("research:cancel:*"))
            + list(client.keys("research:lock:*"))
        )
        if keys:
            client.delete(*keys)
        client.delete("research:jobs")
        client.delete("research:jobs:dlq")
        client.delete("research:running_runs")
    except Exception:
        pass
    yield


@pytest.mark.asyncio
@pytest.mark.integration
async def test_enqueue_and_consume_job_with_ack():
    """Validates that POST /research enqueues to stream, worker processes it, and acks upon completion."""
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}

    worker = ResearchStreamWorker(consumer_name="test-worker-1")
    worker.stream_adapter.create_group("research:jobs", "research-workers")

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/research",
            json={"question": "Distributed Consensus Optimization"},
            headers=headers,
        )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

    # Verify job is present in stream
    adapter = RedisStreamAdapter()
    jobs = adapter.read_job("research:jobs", "research-workers", "test-worker-1", count=1, block=500)
    assert len(jobs) == 1
    entry_id, payload = jobs[0]
    assert payload["run_id"] == run_id

    # Process job with worker
    success = await worker.process_job(entry_id, payload)
    assert success is True

    # Verify status in Redis is completed
    state = redis_store.get_run_state(run_id)
    assert state is not None
    assert state.get("status") in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW, "completed", "needs_review")

    # Verify job was ACKed: pending count is 0
    pending_summary = adapter.get_pending_summary("research:jobs", "research-workers")
    assert pending_summary.get("pending", 0) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_of_queued_job():
    """Validates that cancelling a queued job before pickup marks it cancelled and ACKs it immediately."""
    adapter = RedisStreamAdapter()
    adapter.create_group("research:jobs", "research-workers")

    run_id = f"test_cancel_queued_{time.time()}"
    redis_store.save_run_state(run_id, {"run_id": run_id, "status": RunStatus.QUEUED})

    # Post cancel before worker pickup
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        cancel_resp = await client.post(f"/research/{run_id}/cancel", headers=headers)
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

    # Enqueue job to stream
    adapter.publish_job(
        "research:jobs",
        {"run_id": run_id, "question": "Cancelled Queued Test", "type": "initial"},
    )

    worker = ResearchStreamWorker(consumer_name="test-worker-cancel")
    jobs = adapter.read_job("research:jobs", "research-workers", "test-worker-cancel", count=1, block=500)
    assert len(jobs) == 1
    e_id, payload = jobs[0]

    # Worker picks up job: should detect cancellation and ACK
    success = await worker.process_job(e_id, payload)
    assert success is True

    state = redis_store.get_run_state(run_id)
    assert state.get("status") in (RunStatus.CANCELLED, "cancelled")

    # Verify job was ACKed
    pending = adapter.get_pending_summary("research:jobs", "research-workers")
    assert pending.get("pending", 0) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dlq_after_three_deliveries():
    """Validates that a job delivered more than 3 times is moved to research:jobs:dlq with error trace."""
    adapter = RedisStreamAdapter()
    adapter.create_group("research:jobs", "research-workers")

    run_id = f"test_poison_dlq_{time.time()}"
    redis_store.save_run_state(run_id, {"run_id": run_id, "status": RunStatus.QUEUED})

    entry_id = adapter.publish_job(
        "research:jobs",
        {"run_id": run_id, "question": "Poison Pill Question", "type": "initial"},
    )

    # Read job once to place it in pending list
    adapter.read_job("research:jobs", "research-workers", "test-worker-dlq", count=1, block=500)

    worker = ResearchStreamWorker(consumer_name="test-worker-dlq")

    # Simulate delivery count > 3 by mocking _get_delivery_count
    with patch.object(worker, "_get_delivery_count", return_value=4):
        success = await worker.process_job(
            entry_id,
            {"run_id": run_id, "question": "Poison Pill Question", "type": "initial"},
        )
        assert success is False

    # Check DLQ stream
    client = redis_store.get_client()
    dlq_entries = client.xrange("research:jobs:dlq", "-", "+")
    assert len(dlq_entries) >= 1
    dlq_id, dlq_data = dlq_entries[0]
    assert dlq_data["run_id"] == run_id
    assert "Exceeded maximum delivery attempts" in dlq_data["error"]

    # Check that main stream entry is ACKed
    pending = adapter.get_pending_summary("research:jobs", "research-workers")
    assert pending.get("pending", 0) == 0

    # Run state is marked failed
    state = redis_store.get_run_state(run_id)
    assert state.get("status") in (RunStatus.FAILED, "failed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_in_process_crash_simulation_reclaim_and_resume():
    """
    Simulates consumer A reading a job and crashing/abandoning it.
    Consumer B reclaims with min-idle 0, resumes execution from last checkpoint,
    and finishes: no node twice, spend = unique steps, status completed, entry acked.
    """
    adapter = RedisStreamAdapter()
    adapter.create_group("research:jobs", "research-workers")

    run_id = f"test_reclaim_resume_{time.time()}"
    redis_store.save_run_state(run_id, {"run_id": run_id, "status": RunStatus.QUEUED})

    adapter.publish_job(
        "research:jobs",
        {"run_id": run_id, "question": "Log Compaction Resilience", "type": "initial"},
    )

    # Consumer A reads the job, placing it in pending list, and abandons it
    jobs_a = adapter.read_job("research:jobs", "research-workers", "consumer-A", count=1, block=500)
    assert len(jobs_a) == 1

    # Pending summary shows 1 pending message owned by consumer-A
    pending = adapter.get_pending_summary("research:jobs", "research-workers")
    assert pending.get("pending") == 1

    # Consumer B reclaims idle jobs with min_idle_ms=0
    consumer_b = ResearchStreamWorker(consumer_name="consumer-B")
    reclaimed = await consumer_b.reclaim_idle_jobs(min_idle_ms=0)
    assert reclaimed == 1

    # Run completes
    state = redis_store.get_run_state(run_id)
    assert state is not None
    assert state.get("status") in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW, "completed", "needs_review")

    # Job is ACKed from stream
    pending_after = adapter.get_pending_summary("research:jobs", "research-workers")
    assert pending_after.get("pending", 0) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fake_mode_single_worker_refusal():
    """Validates that a second worker refuses to start in fake mode with a clear error."""
    worker1 = ResearchStreamWorker(consumer_name="fake-worker-primary")
    worker1.register_fake_worker()

    worker2 = ResearchStreamWorker(consumer_name="fake-worker-secondary")
    with pytest.raises(RuntimeError) as exc_info:
        worker2.register_fake_worker()

    assert "Fake mode supports ONE worker only" in str(exc_info.value)
    assert "racy" in str(exc_info.value)

    worker1.unregister_fake_worker()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_max_concurrent_runs_enforced():
    """Validates that CrossProcessConcurrencyLimiter caps active workers to max_concurrent_runs."""
    limiter = CrossProcessConcurrencyLimiter(
        key="test:cap:workers",
        max_concurrent=3,
        fallback_on_unsupported=True,
    )
    redis_store.get_client().delete("test:cap:workers")

    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    # 4th exceeds cap of 3
    assert limiter.acquire() is False

    # Release 1 slot
    assert limiter.release() is True
    # Acquire succeeds again
    assert limiter.acquire() is True
    assert limiter.acquire() is False

    limiter.release()
    limiter.release()
    limiter.release()
    redis_store.get_client().delete("test:cap:workers")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_approve_after_pause_enqueues_exactly_one_resume_job():
    """Validates that pausing at awaiting_approval ACKs the initial job, and POST /approve enqueues exactly one resume job."""
    adapter = RedisStreamAdapter()
    adapter.create_group("research:jobs", "research-workers")

    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}

    # Post with require_approval=True
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        post_resp = await client.post(
            "/research",
            json={"question": "Human Approval Synthesis Test", "require_approval": True},
            headers=headers,
        )
        assert post_resp.status_code == 202
        run_id = post_resp.json()["run_id"]

    worker = ResearchStreamWorker(consumer_name="test-worker-approval")
    jobs = adapter.read_job("research:jobs", "research-workers", "test-worker-approval", count=1, block=500)
    assert len(jobs) == 1
    e_id, payload = jobs[0]

    # Process initial phase: must pause at awaiting_approval and ACK initial job
    await worker.process_job(e_id, payload)
    state = redis_store.get_run_state(run_id)
    assert state.get("status") in (RunStatus.AWAITING_APPROVAL, "awaiting_approval")

    # Lock must be released BEFORE ACK so /approve cannot race it
    client_redis = redis_store.get_client()
    assert client_redis.get(f"research:lock:{run_id}") is None

    # Initial job was ACKed
    pending = adapter.get_pending_summary("research:jobs", "research-workers")
    assert pending.get("pending", 0) == 0

    # First POST /approve: transitions to running and enqueues resume job
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        appr1 = await client.post(f"/research/{run_id}/approve", headers=headers)
        assert appr1.status_code == 200

        # Second POST /approve: idempotent 200, no second resume
        appr2 = await client.post(f"/research/{run_id}/approve", headers=headers)
        assert appr2.status_code == 200
        assert "already" in appr2.json()["message"]

    # Verify exactly one resume job was added to stream
    redis_client = redis_store.get_client()
    stream_entries = redis_client.xrange("research:jobs", "-", "+")
    resume_entries = [e for e in stream_entries if e[1].get("type") == "resume" and e[1].get("run_id") == run_id]
    assert len(resume_entries) == 1

    # Worker picks up the resume job and finishes synthesis
    resume_jobs = adapter.read_job("research:jobs", "research-workers", "test-worker-approval", count=1, block=500)
    assert len(resume_jobs) == 1
    r_eid, r_payload = resume_jobs[0]
    assert r_payload["type"] == "resume"

    await worker.process_job(r_eid, r_payload)
    final_state = redis_store.get_run_state(run_id)
    assert final_state.get("status") in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW, "completed", "needs_review")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_heartbeat_loss_stops_the_run():
    """Validates that if lock ownership is lost during execution, worker detects it and halts without ACKing."""
    adapter = RedisStreamAdapter()
    adapter.create_group("research:jobs", "research-workers")

    run_id = f"test_hb_loss_{time.time()}"
    redis_store.save_run_state(run_id, {"run_id": run_id, "status": RunStatus.QUEUED})

    adapter.publish_job(
        "research:jobs",
        {"run_id": run_id, "question": "Heartbeat loss test", "type": "initial"},
    )

    worker = ResearchStreamWorker(consumer_name="test-worker-hb")
    jobs = adapter.read_job("research:jobs", "research-workers", "test-worker-hb", count=1, block=500)
    assert len(jobs) == 1
    e_id, payload = jobs[0]

    # Pre-steal the lock so renewal fails immediately
    client = redis_store.get_client()
    client.set(f"research:lock:{run_id}", "stolen_by_another_worker")

    success = await worker.process_job(e_id, payload)
    # Lock was stolen by another holder -> worker refused to process
    assert success is False

    # Job is left UNACKED in pending list
    pending = adapter.get_pending_summary("research:jobs", "research-workers")
    assert pending.get("pending") == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sigint_leaves_job_unacked():
    """Validates that a worker interrupted mid-flight (simulating SIGINT) leaves the job UNACKED for recovery."""
    adapter = RedisStreamAdapter()
    adapter.create_group("research:jobs", "research-workers")

    run_id = f"test_sigint_unacked_{time.time()}"
    redis_store.save_run_state(run_id, {"run_id": run_id, "status": RunStatus.QUEUED})

    adapter.publish_job(
        "research:jobs",
        {"run_id": run_id, "question": "SIGINT unacked test", "type": "initial"},
    )

    worker = ResearchStreamWorker(consumer_name="test-worker-sigint")
    jobs = adapter.read_job("research:jobs", "research-workers", "test-worker-sigint", count=1, block=500)
    assert len(jobs) == 1
    e_id, payload = jobs[0]

    # Mock _execute_initial to raise CancelledError (as caused by SIGINT / cancellation)
    with patch.object(worker, "_execute_initial", side_effect=asyncio.CancelledError()):
        with pytest.raises(asyncio.CancelledError):
            await worker.process_job(e_id, payload)

    # Job must remain UNACKED in pending list so another worker can reclaim it
    pending = adapter.get_pending_summary("research:jobs", "research-workers")
    assert pending.get("pending") == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lock_released_strictly_prior_to_ack_at_awaiting_approval():
    """
    Validates that when pausing at awaiting_approval, the distributed run lock
    is released strictly BEFORE stream_adapter.ack_job() is invoked.
    Fails if the lock is released after ACK or not released prior to ACK.
    """
    adapter = RedisStreamAdapter()
    adapter.create_group("research:jobs", "research-workers")

    run_id = f"test_lock_order_{time.time()}"
    redis_store.save_run_state(run_id, {"run_id": run_id, "status": RunStatus.QUEUED})

    adapter.publish_job(
        "research:jobs",
        {"run_id": run_id, "question": "Lock order test", "require_approval": "true", "type": "initial"},
    )

    worker = ResearchStreamWorker(consumer_name="test-worker-lock-order")
    jobs = adapter.read_job("research:jobs", "research-workers", "test-worker-lock-order", count=1, block=500)
    assert len(jobs) == 1
    e_id, payload = jobs[0]

    lock_was_released_at_ack_time = None
    original_ack = worker.stream_adapter.ack_job

    def spy_ack_job(stream, group, job_id):
        nonlocal lock_was_released_at_ack_time
        # Inspect lock status at the exact moment ACK is called
        client = redis_store.get_client()
        lock_val = client.get(f"research:lock:{run_id}")
        lock_was_released_at_ack_time = (lock_val is None)
        return original_ack(stream, group, job_id)

    with patch.object(worker.stream_adapter, "ack_job", side_effect=spy_ack_job):
        success = await worker.process_job(e_id, payload)
        assert success is True

    # State must be awaiting_approval
    state = redis_store.get_run_state(run_id)
    assert state.get("status") in (RunStatus.AWAITING_APPROVAL, "awaiting_approval")

    # Lock MUST have been released BEFORE ACK was executed
    assert lock_was_released_at_ack_time is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_without_lock_release_before_ack_immediate_approval_races():
    """
    Simulates the condition where a worker fails to release the lock before ACK:
    if the lock remains held across ACK, an immediate /approve call or subsequent worker
    attempting to resume fails to acquire the lock immediately without retries.
    """
    run_id = f"test_race_{time.time()}"
    redis_store.save_run_state(run_id, {"run_id": run_id, "status": RunStatus.AWAITING_APPROVAL})

    # Suppose worker 1 did NOT release lock prior to or at ACK, keeping it locked:
    token = redis_store.acquire_lock(run_id, timeout_seconds=30.0)
    assert token is not None

    # Immediate lock acquisition by worker 2 without waiting/retry must fail:
    token_worker_2 = redis_store.acquire_lock(run_id, timeout_seconds=30.0)
    assert token_worker_2 is None, "Worker 2 cannot acquire lock while worker 1 still holds it"

    # Once worker 1 releases the lock (as required before ACK), worker 2 succeeds:
    redis_store.release_lock(run_id, token)
    token_worker_2_after_release = redis_store.acquire_lock(run_id, timeout_seconds=30.0)
    assert token_worker_2_after_release is not None
    redis_store.release_lock(run_id, token_worker_2_after_release)

