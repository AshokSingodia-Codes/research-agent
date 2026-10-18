import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.auth import check_research_rate_limit
from app.config import settings
from app.models.schemas import ResearchRequest
from app.persistence.checkpointer import RedisCheckpointSaver
from app.tools.fetch_extract import is_safe_url


def test_zero_pickle_in_codebase():
    """Validates that zero pickle imports, calls, or serialized artifacts exist across app/, tests/, and fixtures."""
    repo_root = Path(__file__).resolve().parent.parent
    candidate_dirs = [
        repo_root / "app",
        repo_root / "tests",
        repo_root / "fixtures",
        repo_root / "tests" / "fixtures",
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
            if py_file.name == "test_security_phase_a.py":
                continue
            lines = py_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith("#") or "Zero pickle" in line:
                    continue
                if pattern.search(line):
                    pickle_matches.append(f"{py_file}:{line_no} -> {line.strip()}")

    assert len(pickle_matches) == 0, "Disallowed pickle usage found:\n" + "\n".join(pickle_matches)


def test_ssrf_protection_rules():
    """Validates SSRF protection blocking private, loopback, link-local, cloud metadata IPs, and non-http schemes."""
    # 1. Non-http schemes
    safe, reason = is_safe_url("file:///etc/passwd")
    assert safe is False
    assert "scheme" in reason.lower()

    safe, reason = is_safe_url("ftp://example.com/file")
    assert safe is False

    safe, reason = is_safe_url("gopher://example.com:70/1")
    assert safe is False

    # 2. Loopback and internal IPs
    safe, reason = is_safe_url("http://127.0.0.1/admin")
    assert safe is False
    assert "blocked" in reason.lower() or "private" in reason.lower()

    safe, reason = is_safe_url("http://localhost/secret")
    assert safe is False

    # 3. Cloud metadata service
    safe, reason = is_safe_url("http://169.254.169.254/latest/meta-data/")
    assert safe is False

    # 4. RFC 1918 Private ranges
    safe, reason = is_safe_url("http://10.0.0.1/status")
    assert safe is False

    safe, reason = is_safe_url("http://192.168.1.1/router")
    assert safe is False

    safe, reason = is_safe_url("http://172.16.0.1/internal")
    assert safe is False

    # 5. Public URL allowed
    safe, reason = is_safe_url("https://example.com/docs")
    assert safe is True


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


def test_research_request_sanitization_and_length_limits():
    """Validates that ResearchRequest question is sanitized (whitespace stripped) and rejects out-of-bound lengths."""
    # 1. Strips leading and trailing whitespace
    req = ResearchRequest(question="   Distributed Consensus Architecture   ")
    assert req.question == "Distributed Consensus Architecture"

    # 2. Rejects questions shorter than 3 characters
    with pytest.raises(ValidationError):
        ResearchRequest(question="ab")

    # 3. Rejects questions longer than 500 characters
    long_q = "A" * 501
    with pytest.raises(ValidationError):
        ResearchRequest(question=long_q)

    # 4. Accepts valid question up to 500 chars
    valid_q = "A" * 500
    req_valid = ResearchRequest(question=valid_q)
    assert len(req_valid.question) == 500


def test_research_rate_limiter_blocks_excessive_submissions():
    """Validates that research submission rate limiter trips after 10 requests per minute."""
    caller_id = "test_rate_limited_caller"
    from fastapi import HTTPException

    # First 10 requests must succeed
    for _ in range(10):
        check_research_rate_limit(caller_id)

    # 11th request must raise 429
    with pytest.raises(HTTPException) as exc_info:
        check_research_rate_limit(caller_id)
    assert exc_info.value.status_code == 429
    assert "rate limit exceeded" in exc_info.value.detail.lower()


def test_each_agent_requests_own_configured_model(monkeypatch):
    """Validates that Planner, Researcher, Writer, Supervisor, and Critic each request their own configured model."""
    from app.agents.critic import critique_report
    from app.agents.planner import plan_research
    from app.agents.researcher import research_subquestion
    from app.agents.supervisor import review_findings
    from app.agents.writer import write_report
    from app.config import settings
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


def test_ssrf_redirect_to_loopback_and_metadata(monkeypatch):
    """Validates that extract_full_page blocks redirects to 127.0.0.1 and 169.254.169.254."""
    import httpx

    from app.tools.fetch_extract import ToolStatus, extract_full_page

    # 1. Redirect to 127.0.0.1
    resp_redirect_loopback = httpx.Response(
        status_code=302,
        headers={"Location": "http://127.0.0.1/admin"},
        request=httpx.Request("GET", "https://public-source.org/login"),
    )
    monkeypatch.setattr("httpx.Client.get", lambda *args, **kwargs: resp_redirect_loopback)
    text, status = extract_full_page("https://public-source.org/login")
    assert status == ToolStatus.BLOCKED
    assert text == ""

    # 2. Redirect to 169.254.169.254
    resp_redirect_metadata = httpx.Response(
        status_code=302,
        headers={"Location": "http://169.254.169.254/latest/meta-data/"},
        request=httpx.Request("GET", "https://public-source.org/api"),
    )
    monkeypatch.setattr("httpx.Client.get", lambda *args, **kwargs: resp_redirect_metadata)
    text, status = extract_full_page("https://public-source.org/api")
    assert status == ToolStatus.BLOCKED
    assert text == ""


def test_ssrf_ip_resolution_and_mapped_addresses(monkeypatch):
    """Validates SSRF prevention on private DNS resolution, IPv6 loopback, and IPv4-mapped IPv6."""
    import socket

    from app.tools.fetch_extract import is_safe_url

    # 1. Hostname resolving to private IP
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.4.2", 80))],
    )
    safe, reason = is_safe_url("https://deceptive-internal-dns.org/report")
    assert safe is False
    assert "blocked" in reason.lower()

    # 2. IPv6 loopback ::1
    safe, reason = is_safe_url("http://[::1]/debug")
    assert safe is False
    assert "blocked" in reason.lower()

    # 3. IPv4-mapped IPv6 ::ffff:127.0.0.1
    safe, reason = is_safe_url("http://[::ffff:127.0.0.1]/secret")
    assert safe is False
    assert "blocked" in reason.lower()


def test_extract_full_page_truncates_oversized_response(monkeypatch):
    """Validates that extract_full_page handles responses > 2 MB by truncating without failure."""
    import httpx

    from app.tools.fetch_extract import MAX_RESPONSE_BYTES, ToolStatus, extract_full_page

    large_body = b"<html><body><p>" + (b"A" * (3 * 1024 * 1024)) + b"</p></body></html>"
    mock_resp = httpx.Response(
        status_code=200,
        content=large_body,
        request=httpx.Request("GET", "https://public-benchmark.org/large"),
    )
    monkeypatch.setattr("httpx.Client.get", lambda *args, **kwargs: mock_resp)

    text, status = extract_full_page("https://public-benchmark.org/large")
    assert status == ToolStatus.OK
    assert len(text) > 0
    assert len(text) <= MAX_RESPONSE_BYTES
