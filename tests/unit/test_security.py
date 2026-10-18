import re
from pathlib import Path

import pytest

from app.config import settings
from app.persistence.checkpointer import RedisCheckpointSaver


def test_zero_pickle_in_codebase():
    """Validates that zero pickle imports, calls, or serialized artifacts exist across app/, tests/, and fixtures."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate_dirs = [
        repo_root / "app",
        repo_root / "tests",
        repo_root / "fixtures",
    ]
    targets = [d for d in candidate_dirs if d.exists()]

    # Ensure no .pkl or .pickle files exist in repo (ignoring virtualenv and git)
    pickle_files = [
        f
        for f in list(repo_root.rglob("*.pkl")) + list(repo_root.rglob("*.pickle"))
        if ".venv" not in f.parts and ".git" not in f.parts
    ]
    assert len(pickle_files) == 0, f"Disallowed pickle artifacts found: {pickle_files}"

    pickle_matches = []
    # Match 'import pickle' or 'from pickle' or 'pickle.'
    pattern = re.compile(r"^\s*(?:import\s+pickle|from\s+pickle|.*pickle\.(?:loads|dumps|load|dump))")

    for target in targets:
        for py_file in target.rglob("*.py"):
            # Exclude this test file itself from the search
            if py_file.name == "test_security.py":
                continue
            lines = py_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith("#") or "Zero pickle" in line:
                    continue
                if pattern.search(line):
                    pickle_matches.append(f"{py_file}:{line_no} -> {line.strip()}")

    assert len(pickle_matches) == 0, "Disallowed pickle usage found:\n" + "\n".join(pickle_matches)


def test_checkpoint_hmac_tamper_detection(monkeypatch):
    """Validates that RedisCheckpointSaver detects and rejects tampered checkpoint data when HMAC key is enabled."""
    monkeypatch.setattr(settings, "checkpoint_hmac_key", "super-secret-hmac-key")
    checkpointer = RedisCheckpointSaver()

    obj = {"run_id": "test_hmac_001", "tokens": 1200}
    serialized_envelope = checkpointer._serialize_blob(obj)

    # 1. Valid signature deserializes correctly
    deserialized = checkpointer._deserialize_blob(serialized_envelope)
    assert deserialized == obj

    # 2. Tampered signature is rejected
    import json

    data = json.loads(serialized_envelope)
    data["s"] = "tampered_fake_signature_hex"
    tampered_envelope = json.dumps(data)

    with pytest.raises(ValueError, match="HMAC signature verification failed"):
        checkpointer._deserialize_blob(tampered_envelope)


def test_each_agent_requests_own_configured_model(monkeypatch):
    """Validates that Planner, Researcher, Writer, Supervisor, and Critic each request their own configured model."""
    from app.agents.critic import critique_report
    from app.agents.planner import plan_research
    from app.agents.researcher import research_subquestion
    from app.agents.supervisor import review_findings
    from app.agents.writer import write_report
    from app.models.schemas import Finding, Report, ReportSection, ResearchPlan, SubQuestion

    monkeypatch.setattr(settings, "mock_llm", "true")
    monkeypatch.setattr(settings, "planner_model", "test-planner-custom-v1")
    monkeypatch.setattr(settings, "researcher_model", "test-researcher-custom-v1")
    monkeypatch.setattr(settings, "writer_model", "test-writer-custom-v1")
    monkeypatch.setattr(settings, "supervisor_model", "test-supervisor-custom-v1")
    monkeypatch.setattr(settings, "critic_model", "test-critic-custom-v1")

    # 1. Planner
    _, plan_trace = plan_research("Consensus Protocols")
    assert plan_trace.model == "test-planner-custom-v1"

    # 2. Researcher
    sq = SubQuestion(id="sq_1", text="What is Paxos throughput in practice?", rationale="Throughput eval", priority=1)
    _, res_trace = research_subquestion(sq)
    assert res_trace.model == "test-researcher-custom-v1"

    # 3. Supervisor
    finding = Finding(
        id="f1",
        sub_question_id="sq_1",
        claim="Paxos achieves high throughput.",
        source_url="https://example.com/paxos",
        domain="example.com",
        snippet="Snippet text",
        retrieval_timestamp=100.0,
        content_hash="h1",
        confidence=0.9,
    )
    _, sup_trace = review_findings(sq, [finding], current_retry_count=0)
    assert sup_trace.model == "test-supervisor-custom-v1"

    # 4. Writer
    _, writer_trace = write_report("Consensus Protocols", [finding])
    assert writer_trace.model == "test-writer-custom-v1"

    # 5. Critic
    plan = ResearchPlan(topic="Consensus Protocols", sub_questions=[sq])
    draft = Report(
        title="Draft",
        topic="Consensus Protocols",
        summary="Summary",
        sections=[ReportSection(title="Sec 1", content="Content [1]", citations=["[1]"])],
    )
    _, critic_trace = critique_report(plan, [finding], draft)
    assert critic_trace.model == "test-critic-custom-v1"
