from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.models.enums import RunStatus
from app.persistence.redis_store import RedisStore


@pytest.mark.unit
def test_redis_store_debounced_disk_sync(tmp_path):
    """
    Validates that rapid save_run_state calls debounce disk writes,
    writing at most 2 times across 10 rapid calls, and that flush() forces an immediate write.
    """
    store = RedisStore()
    test_db_file = tmp_path / "test_fake_redis.json"

    with patch("app.persistence.redis_store.FAKEDB_FILE", test_db_file):
        # Initial state
        store.write_count = 0
        store._last_sync_time = 0.0
        store._sync_debounce_seconds = 1.0

        # Rapid 10 calls
        for i in range(10):
            store.save_run_state(
                f"run_debounce_{i}",
                {"status": RunStatus.PLANNING, "progress_percentage": float(i * 10)},
            )

        # Due to debounce window (1s) and instantaneous loop execution,
        # at most 2 writes should have occurred (first write + at most 1 debounced)
        assert store.write_count <= 2
        initial_write_count = store.write_count

        # Force flush
        store.flush()
        assert store.write_count == initial_write_count + 1
        assert test_db_file.exists()


@pytest.mark.unit
def test_redis_store_serialization_and_load_from_disk(tmp_path):
    """Validates saving strings, sets, and lists to disk and restoring them."""
    store = RedisStore()
    test_db_file = tmp_path / "test_db_persist.json"

    with patch("app.persistence.redis_store.FAKEDB_FILE", test_db_file):
        client = store.get_client()
        client.set("key_str", "hello_world")
        client.sadd("key_set", "item1", "item2")
        client.delete("key_list")
        client.rpush("key_list", "a", "b", "c")

        store.sync_to_disk(force=True)
        assert test_db_file.exists()

        # Create fresh store and load from snapshot
        fresh_store = RedisStore()
        fresh_client = fresh_store.get_client()
        fresh_store.load_from_disk()

        assert fresh_client.get("key_str") == "hello_world"
        assert set(fresh_client.smembers("key_set")) == {"item1", "item2"}
        assert fresh_client.lrange("key_list", 0, -1) == ["a", "b", "c"]


@pytest.mark.unit
def test_redis_store_get_run_status():
    """Validates get_run_status parsing and progress calculation."""
    store = RedisStore()
    run_id = "run_status_calc_test"

    # State with completed subquestions
    store.save_run_state(
        run_id,
        {
            "run_id": run_id,
            "status": RunStatus.RESEARCHING,
            "completed_subquestion_ids": ["sq_1", "sq_2"],
            "total_subquestions": 4,
            "progress_percentage": 0.0,
            "total_tokens": 1500,
            "current_node": "research",
        },
    )

    status = store.get_run_status(run_id)
    assert isinstance(status, SimpleNamespace)
    assert status.run_id == run_id
    assert status.status == RunStatus.RESEARCHING.value
    assert status.progress_percentage == 50.0  # (2/4) * 100
    assert status.completed_subquestions == 2
    assert status.total_subquestions == 4
    assert status.total_tokens == 1500
    assert status.current_node == "research"

    # Non-existent run
    assert store.get_run_status("non_existent_run_xyz") is None


@pytest.mark.unit
def test_redis_store_locks():
    """Validates distributed lock acquisition, renewal, and release."""
    store = RedisStore()
    lock_id = "test_resource_lock"

    assert bool(store.acquire_lock(lock_id, timeout_seconds=10.0)) is True
    # Re-acquisition fails while locked
    assert store.acquire_lock(lock_id, timeout_seconds=10.0) is None

    # Release
    store.release_lock(lock_id)
    # Now acquisition succeeds again
    assert bool(store.acquire_lock(lock_id, timeout_seconds=10.0)) is True
    store.release_lock(lock_id)
