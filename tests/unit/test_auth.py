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


def test_stream_token_lifecycle_and_single_use(monkeypatch):
    """Validates that stream tokens are run-scoped, time-bounded, and single-use."""
    from app.api.auth import create_stream_token, verify_stream_access, verify_stream_token

    monkeypatch.setattr(settings, "api_key", "test-secure-api-key-32charslong!")
    monkeypatch.setattr(settings, "stream_token_secret", "custom-stream-secret-token")

    run_id = "run_test_stream_123"
    token = create_stream_token(run_id, ttl_seconds=60)

    # 1. First verification must succeed
    assert verify_stream_token(run_id, token) is True

    # 2. Replay must fail (single-use token enforcement)
    assert verify_stream_token(run_id, token) is False

    # 3. Token for different run_id must fail
    another_token = create_stream_token("run_diff_abc", ttl_seconds=60)
    assert verify_stream_token(run_id, another_token) is False

    # 4. Expired token must fail
    expired_token = create_stream_token(run_id, ttl_seconds=-10)
    assert verify_stream_token(run_id, expired_token) is False

    # 5. verify_stream_access allows valid fresh token
    fresh_token = create_stream_token(run_id, ttl_seconds=60)
    caller = verify_stream_access(run_id=run_id, token=fresh_token, header_key=None)
    assert caller == f"token:{run_id}"

    # 6. verify_stream_access rejects when no token and no header provided while api_key configured
    with pytest.raises(HTTPException) as exc:
        verify_stream_access(run_id=run_id, token=None, header_key=None)
    assert exc.value.status_code == 401


def test_production_lifespan_refuses_insecure_keys(monkeypatch):
    """Validates that production environment refuses to start with missing, short, or known test keys."""
    from app.main import validate_production_security

    monkeypatch.setattr(settings, "env", "production")

    # Insecure / default key
    monkeypatch.setattr(settings, "api_key", "test-key-12345")
    with pytest.raises(RuntimeError, match="CRITICAL: Production deployment requires a secure"):
        validate_production_security()

    # Short key
    monkeypatch.setattr(settings, "api_key", "short_key_123")
    with pytest.raises(RuntimeError, match="CRITICAL: Production deployment requires a secure"):
        validate_production_security()

    # CORS wildcard forbidden in production
    monkeypatch.setattr(settings, "api_key", "super-secure-production-api-key-32chars!")
    monkeypatch.setattr(settings, "cors_origins", ["*"])
    with pytest.raises(RuntimeError, match="CRITICAL: CORS allow_origins must not contain"):
        validate_production_security()
