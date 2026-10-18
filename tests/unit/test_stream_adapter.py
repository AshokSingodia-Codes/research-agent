from unittest.mock import MagicMock

import pytest
from redis.exceptions import ResponseError

from app.persistence.redis_store import redis_store
from app.persistence.stream_adapter import RedisStreamAdapter


@pytest.mark.unit
def test_stream_adapter_basic_operations():
    """Validates stream group creation, job publishing, reading, and acknowledgment."""
    client = redis_store.get_client()
    adapter = RedisStreamAdapter(client)

    stream = "test:stream:basic"
    group = "test_group_basic"

    # 1. Create group idempotently
    assert adapter.create_group(stream, group, mkstream=True) is True
    # Second call should not error (BUSYGROUP handled)
    assert adapter.create_group(stream, group, mkstream=True) is True

    # 2. Publish job
    msg_id = adapter.publish_job(stream, {"run_id": "run_job_1", "action": "research"})
    assert msg_id is not None

    # 3. Read job via consumer
    jobs = adapter.read_job(stream, group, "worker_1", count=1)
    assert len(jobs) == 1
    assert jobs[0][0] == msg_id
    assert jobs[0][1]["run_id"] == "run_job_1"

    # 4. Check pending
    pending = adapter.get_pending_summary(stream, group)
    assert pending.get("pending", 0) >= 1

    # 5. Acknowledge job
    acked = adapter.ack_job(stream, group, msg_id)
    assert acked >= 1

    # Pending count should drop
    pending_after = adapter.get_pending_summary(stream, group)
    assert pending_after.get("pending", 0) == 0


@pytest.mark.unit
def test_stream_adapter_autoclaim_supported():
    """Validates autoclaim_jobs when xautoclaim is supported by the client."""
    client = redis_store.get_client()
    adapter = RedisStreamAdapter(client)

    stream = "test:stream:autoclaim"
    group = "test_group_autoclaim"
    adapter.create_group(stream, group, mkstream=True)

    msg_id = adapter.publish_job(stream, {"run_id": "run_abandoned_1"})
    # Worker A reads but never acks
    jobs = adapter.read_job(stream, group, "worker_A", count=1)
    assert len(jobs) == 1

    # Worker B reclaims it via autoclaim (idle time 0)
    next_id, claimed, _ = adapter.autoclaim_jobs(stream, group, "worker_B", min_idle_time=0)
    assert len(claimed) >= 1
    assert claimed[0][0] == msg_id


@pytest.mark.unit
def test_stream_adapter_autoclaim_fallback_on_unsupported_command():
    """Validates fallback to xpending_range + xclaim when xautoclaim raises ResponseError."""
    mock_client = MagicMock()
    adapter = RedisStreamAdapter(mock_client)

    # Simulate unsupported XAUTOCLAIM
    mock_client.xautoclaim.side_effect = ResponseError("unknown command 'XAUTOCLAIM'")

    # Mock xpending_range returning one abandoned message
    mock_client.xpending_range.return_value = [{"message_id": "1000-0", "idle": 5000, "consumer": "worker_A"}]
    # Mock xclaim claiming the message
    mock_client.xclaim.return_value = [("1000-0", {"run_id": "run_reclaimed"})]

    next_id, claimed, deleted = adapter.autoclaim_jobs(
        stream="test:stream:fallback",
        group="fallback_group",
        consumer="worker_B",
        min_idle_time=1000,
    )

    assert next_id == "1000-0"
    assert len(claimed) == 1
    assert claimed[0][0] == "1000-0"
    assert claimed[0][1]["run_id"] == "run_reclaimed"
    mock_client.xclaim.assert_called_once_with(
        "test:stream:fallback",
        "fallback_group",
        "worker_B",
        min_idle_time=1000,
        message_ids=["1000-0"],
    )
