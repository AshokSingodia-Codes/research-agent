import pytest
from fastapi import HTTPException

from app.api.auth import check_research_rate_limit, verify_api_key
from app.config import settings


def test_research_rate_limiter_blocks_excessive_submissions():
    """Validates that research submission rate limiter trips after 10 requests per minute."""
    caller_id = "test_rate_limited_caller"

    # First 10 requests must succeed
    for _ in range(10):
        check_research_rate_limit(caller_id)

    # 11th request must raise 429
    with pytest.raises(HTTPException) as exc_info:
        check_research_rate_limit(caller_id)
    assert exc_info.value.status_code == 429
    assert "rate limit exceeded" in exc_info.value.detail.lower()


def test_api_key_auth_verification(monkeypatch):
    """Validates that verify_api_key enforces authentication and detects invalid keys."""
    monkeypatch.setattr(settings, "api_key", "secret-test-key-xyz")

    # Missing key raises 401
    with pytest.raises(HTTPException) as exc:
        verify_api_key(api_key="")
    assert exc.value.status_code == 401

    # Invalid key raises 403
    with pytest.raises(HTTPException) as exc:
        verify_api_key(api_key="wrong-key")
    assert exc.value.status_code == 403

    # Valid key returns caller_id
    caller = verify_api_key(api_key="secret-test-key-xyz")
    assert caller == "secret-test-key-xyz"
