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
    1. Spawns a baseline clean run to obtain the canonical final report.
    2. Spawns a background worker subprocess with artificial node delays for the chaos run.
    3. Kills the worker process mid-run (Popen.kill()).
    4. Restarts worker in a new process referencing the exact same run_id.
    5. Asserts:
       - Run finishes successfully.
       - Zero duplicate LLM calls are executed across the crash boundary.
       - Final report is strictly identical to the baseline report.
    """
    topic = "High-Throughput Distributed Raft Consensus"
    clean_run_id = f"baseline_run_{int(time.time())}"
    chaos_run_id = f"chaos_run_{int(time.time())}"

    # 1. Run baseline run to completion
    cmd_baseline = [sys.executable, "tests/chaos/chaos_worker_runner.py", clean_run_id, topic, "0.0"]
    res_base = subprocess.run(cmd_baseline, capture_output=True, text=True, timeout=15)
    assert "SUCCESS:" in res_base.stdout, f"Baseline run failed: {res_base.stderr}"
    baseline_report_json = redis_store.get_final_report(clean_run_id)
    assert baseline_report_json is not None

    client = redis_store.get_client()
    baseline_calls = [c.decode("utf-8") if isinstance(c, bytes) else c for c in client.lrange(f"chaos:raw_calls:{clean_run_id}", 0, -1)]

    # 2. Start chaos worker subprocess with 0.8s artificial delay per step
    cmd_chaos = [sys.executable, "tests/chaos/chaos_worker_runner.py", chaos_run_id, topic, "0.8"]
    proc = subprocess.Popen(cmd_chaos, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    checkpointer = RedisCheckpointSaver()
    # Wait until planner node completes and checkpoints into Redis, then immediately kill
    killed = False
    for _ in range(100):
        cp = checkpointer.get_tuple({"configurable": {"thread_id": chaos_run_id}})
        if cp and "plan" in cp.checkpoint.get("channel_values", {}):
            proc.kill()
            proc.wait()
            killed = True
            break
        time.sleep(0.05)

    assert killed, "Subprocess did not reach planner checkpoint within timeout"

    # Verify run did not complete in first process
    assert redis_store.get_final_report(chaos_run_id) is None

    # Checkpoint should exist for the thread
    cp_tuple = checkpointer.get_tuple({"configurable": {"thread_id": chaos_run_id}})
    assert cp_tuple is not None, "Expected checkpoint to be persisted to Redis before process kill"
    assert "plan" in cp_tuple.checkpoint["channel_values"]

    # 4. Restart worker: spawn second process to resume the same run
    cmd_resume = [sys.executable, "tests/chaos/chaos_worker_runner.py", chaos_run_id, topic, "0.0"]
    proc_resume = subprocess.run(cmd_resume, capture_output=True, text=True, timeout=15)

    assert "SUCCESS:" in proc_resume.stdout, f"Resumed worker failed: {proc_resume.stderr}"

    # 5. Verify completed report exists and is identical to baseline report
    chaos_report_json = redis_store.get_final_report(chaos_run_id)
    assert chaos_report_json is not None
    assert chaos_report_json == baseline_report_json, "Final report after recovery must be identical to baseline"

    # 6. Verify zero duplicate LLM calls were made across the crash/restart boundary
    chaos_calls = [c.decode("utf-8") if isinstance(c, bytes) else c for c in client.lrange(f"chaos:raw_calls:{chaos_run_id}", 0, -1)]
    assert len(chaos_calls) > 0
    # Every LLM call recorded should match the sequence of baseline calls without duplicates
    assert len(chaos_calls) == len(set(chaos_calls)), f"Detected duplicate LLM calls: {chaos_calls}"
    assert chaos_calls == baseline_calls, f"Expected exact call match. Chaos: {chaos_calls}, Baseline: {baseline_calls}"
