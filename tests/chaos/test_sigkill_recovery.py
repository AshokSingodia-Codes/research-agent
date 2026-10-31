import subprocess
import sys
import time

import pytest

from app.config import settings
from app.models.schemas import Report, generate_finding_id
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

    if settings.use_fake_redis:
        redis_store.load_from_disk(force=True)

    chaos_calls = [c.decode("utf-8") if isinstance(c, bytes) else c for c in client.lrange(f"chaos:raw_calls:{chaos_run_id}", 0, -1)]
    baseline_searches = [s.decode("utf-8") if isinstance(s, bytes) else s for s in client.lrange(f"chaos:raw_searches:{clean_run_id}", 0, -1)]
    chaos_searches = [s.decode("utf-8") if isinstance(s, bytes) else s for s in client.lrange(f"chaos:raw_searches:{chaos_run_id}", 0, -1)]
    proc_events = [p.decode("utf-8") if isinstance(p, bytes) else p for p in client.lrange(f"chaos:proc_events:{chaos_run_id}", 0, -1)]

    print(f"\n--- CHAOS RECOVERY TRACE ---")
    print(f"Proc events: {proc_events}")
    print(f"Baseline LLM calls ({len(baseline_calls)}): {baseline_calls}")
    print(f"Chaos LLM calls    ({len(chaos_calls)}): {chaos_calls}")
    print(f"Baseline Searches  ({len(baseline_searches)}): {baseline_searches}")
    print(f"Chaos Searches     ({len(chaos_searches)}): {chaos_searches}")

    # 5. Verify completed report exists and matches baseline report (normalizing run-scoped finding IDs)
    chaos_report_json = redis_store.get_final_report(chaos_run_id)
    assert chaos_report_json is not None, "Expected final report for chaos run"
    assert baseline_report_json is not None, "Expected baseline report"

    baseline_report = Report.model_validate_json(baseline_report_json)
    chaos_report = Report.model_validate_json(chaos_report_json)

    # Verify deterministic finding IDs for each run
    for key, citation in baseline_report.citations.items():
        expected_id = generate_finding_id(clean_run_id, citation.source_url, citation.claim)
        assert citation.finding_id == expected_id, f"Baseline citation {key} finding_id must be deterministic"

    for key, citation in chaos_report.citations.items():
        expected_id = generate_finding_id(chaos_run_id, citation.source_url, citation.claim)
        assert citation.finding_id == expected_id, f"Chaos citation {key} finding_id must be deterministic"

    def normalize_report_for_eval(r: Report) -> dict:
        d = r.model_dump()
        for c in d.get("citations", {}).values():
            c.pop("finding_id", None)
        return d

    assert normalize_report_for_eval(chaos_report) == normalize_report_for_eval(baseline_report), (
        "Normalized final report after recovery must be identical to baseline report"
    )

    # 6. Verify zero or at most one replayed node (the one in flight at kill time)
    assert len(chaos_calls) > 0
    assert len(chaos_calls) in (len(baseline_calls), len(baseline_calls) + 1), (
        f"Detected excessive LLM call replays. Chaos: {len(chaos_calls)}, Baseline: {len(baseline_calls)}"
    )
    assert set(chaos_calls) == set(baseline_calls), (
        f"Chaos calls missing baseline steps: {set(baseline_calls) - set(chaos_calls)}"
    )

    assert len(chaos_searches) in (len(baseline_searches), len(baseline_searches) + 1), (
        f"Detected excessive search replays. Chaos: {len(chaos_searches)}, Baseline: {len(baseline_searches)}"
    )
