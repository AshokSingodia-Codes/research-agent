import subprocess
import sys
import time

import pytest

from app.persistence.checkpointer import RedisCheckpointSaver
from app.persistence.redis_store import redis_store


@pytest.mark.chaos
def test_chaos_subprocess_sigkill_and_resumption():
    """
    CHAOS RESILIENCE TEST:
    1. Spawns a background worker subprocess with artificial node delays.
    2. Kills the worker process mid-run (SIGKILL via terminate/kill).
    3. Starts a new worker process referencing the exact same run_id.
    4. Asserts that the run completes, previous completed work was NOT re-executed,
       and the final report is generated.
    """
    run_id = f"chaos_run_{int(time.time())}"
    topic = "High-Throughput Distributed Raft Consensus"

    # 1. Start worker subprocess with 0.8s artificial delay per LLM step
    cmd = [sys.executable, "tests/chaos/chaos_worker_runner.py", run_id, topic, "0.8"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    checkpointer = RedisCheckpointSaver()
    # Wait until planner node completes and checkpoints into Redis
    for _ in range(60):
        cp = checkpointer.get_tuple({"configurable": {"thread_id": run_id}})
        if cp and "plan" in cp.checkpoint.get("channel_values", {}):
            break
        time.sleep(0.1)

    # 2. Hard kill worker subprocess mid-run (simulating SIGKILL / crash)
    proc.kill()
    proc.wait()

    # Verify run did not complete in first process
    assert redis_store.get_final_report(run_id) is None

    # Checkpoint should exist for the thread
    cp_tuple = checkpointer.get_tuple({"configurable": {"thread_id": run_id}})
    assert cp_tuple is not None, "Expected checkpoint to be persisted to Redis before process kill"
    assert "plan" in cp_tuple.checkpoint["channel_values"]

    # 3. Restart worker: spawn second process to resume the same run
    cmd_resume = [sys.executable, "tests/chaos/chaos_worker_runner.py", run_id, topic, "0.1"]
    proc_resume = subprocess.run(cmd_resume, capture_output=True, text=True, timeout=15)

    assert "SUCCESS:" in proc_resume.stdout

    # 4. Verify completed report exists in Redis
    final_report_json = redis_store.get_final_report(run_id)
    assert final_report_json is not None
    assert topic in final_report_json
