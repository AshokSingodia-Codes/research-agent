import os
import re
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.api.auth import check_research_rate_limit, verify_api_key
from app.config import settings
from app.main import app
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
        f for f in list(repo_root.rglob("*.pkl")) + list(repo_root.rglob("*.pickle"))
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

    assert len(pickle_matches) == 0, f"Disallowed pickle usage found:\n" + "\n".join(pickle_matches)


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
