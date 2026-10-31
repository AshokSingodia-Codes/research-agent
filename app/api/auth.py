import hashlib
import hmac
import secrets
import time

from fastapi import HTTPException, Path, Query, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings
from app.persistence.redis_store import redis_store

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# General API rate limit: 120 requests per minute
RATE_LIMIT_MAX_REQUESTS = 120
RATE_LIMIT_WINDOW_SECONDS = 60.0

# Research submission rate limit: 10 requests per minute
RESEARCH_RATE_LIMIT_MAX = 10
RESEARCH_RATE_LIMIT_WINDOW = 60.0


def _get_stream_secret() -> bytes:
    """Returns the secret bytes used for signing HMAC stream tokens."""
    secret = settings.stream_token_secret or settings.api_key
    if not secret:
        secret = "multiagent-stream-token-fallback-secret"
    return secret.encode("utf-8")


def create_stream_token(run_id: str, ttl_seconds: int = 60) -> str:
    """Generates a short-lived (default 60s), run-scoped, single-use HMAC token for SSE streaming."""
    exp = int(time.time()) + ttl_seconds
    nonce = secrets.token_hex(8)
    payload = f"{run_id}:{exp}:{nonce}"
    sig = hmac.new(_get_stream_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def verify_stream_token(run_id: str, token: str) -> bool:
    """
    Validates the run-scoped HMAC stream token, checks expiration,
    and enforces single-use via Redis.
    """
    try:
        parts = token.split(":")
        if len(parts) != 4:
            return False
        token_run_id, exp_str, nonce, sig = parts

        if token_run_id != run_id:
            return False

        exp = int(exp_str)
        if time.time() > exp:
            return False

        payload = f"{token_run_id}:{exp_str}:{nonce}"
        expected_sig = hmac.new(_get_stream_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(sig, expected_sig):
            return False

        # Single-use enforcement via Redis
        client = redis_store.get_client()
        used_key = f"research:stream_token:used:{nonce}"
        remaining_ttl = max(int(exp - time.time()) + 60, 60)
        is_new = bool(client.set(used_key, "1", nx=True, ex=remaining_ttl))
        if not is_new:
            return False

        return True
    except Exception:
        return False


def verify_api_key(
    header_key: str | None = Security(api_key_header),
    api_key: str | None = None,
) -> str:
    """Validates API Key from X-API-Key header using constant-time comparison and enforces per-key rate limiting."""
    configured_key = settings.api_key
    # When invoked directly in tests, header_key may retain its FastAPI Security default object
    resolved_header = header_key if isinstance(header_key, str) else None
    resolved_key = api_key if isinstance(api_key, str) else resolved_header

    # Production safety: reject access if production environment lacks configured API key
    if settings.env.lower() == "production" and (
        not configured_key or len(configured_key) < 16 or configured_key.startswith("your_")
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Production server misconfigured: secure API_KEY required.",
        )

    if configured_key:
        if not resolved_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-API-Key authentication header.",
            )
        # Constant-time comparison to prevent timing side-channel attacks
        if not hmac.compare_digest(resolved_key.encode("utf-8"), configured_key.encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API Key provided.",
            )

    caller_id = resolved_key or "anonymous"
    _check_general_rate_limit(caller_id)
    return caller_id


def verify_stream_access(
    run_id: str = Path(...),
    token: str | None = Query(None),
    header_key: str | None = Security(api_key_header),
) -> str:
    """
    Authorizes SSE stream connections.
    Accepts either an X-API-Key header or a short-lived run-scoped HMAC token (?token=).
    Does NOT accept ?api_key=.
    """
    configured_key = settings.api_key

    # Production safety
    if settings.env.lower() == "production" and (
        not configured_key or len(configured_key) < 16 or configured_key.startswith("your_")
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Production server misconfigured: secure API_KEY required.",
        )

    resolved_header = header_key if isinstance(header_key, str) else None

    # 1. Header authentication
    if resolved_header:
        if configured_key and not hmac.compare_digest(resolved_header.encode("utf-8"), configured_key.encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API Key provided.",
            )
        caller_id = resolved_header
        _check_general_rate_limit(caller_id)
        return caller_id

    # 2. Token authentication
    if token:
        if not verify_stream_token(run_id, token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid, expired, or already-used stream token.",
            )
        return f"token:{run_id}"

    # 3. Neither header nor token provided
    if configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Stream authentication required: provide X-API-Key header or valid ?token=.",
        )

    return "anonymous"


def check_research_rate_limit(caller_id: str) -> None:
    """Enforces Redis-backed token bucket / sliding window rate limiting for research submissions (10 req/min)."""
    client = redis_store.get_client()
    key = f"research:ratelimit:create:{caller_id}"
    now = time.time()
    cutoff = now - RESEARCH_RATE_LIMIT_WINDOW

    try:
        # Clean timestamps older than sliding window
        client.zremrangebyscore(key, 0, cutoff)
        current_count = client.zcard(key)

        if current_count >= RESEARCH_RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Research dispatch rate limit exceeded: maximum {RESEARCH_RATE_LIMIT_MAX} runs per minute.",
            )

        # Record this request with microsecond score for uniqueness
        client.zadd(key, {f"{now}_{time.perf_counter()}": now})
        client.expire(key, int(RESEARCH_RATE_LIMIT_WINDOW + 5))
    except HTTPException:
        raise
    except Exception:
        # If Redis command fails, fallback to passing
        pass


def _check_general_rate_limit(caller_id: str) -> None:
    """In-memory sliding window rate limiter for general endpoint calls."""
    client = redis_store.get_client()
    key = f"research:ratelimit:general:{caller_id}"
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    try:
        client.zremrangebyscore(key, 0, cutoff)
        current_count = client.zcard(key)

        if current_count >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {RATE_LIMIT_MAX_REQUESTS} requests per {int(RATE_LIMIT_WINDOW_SECONDS)}s.",
            )

        client.zadd(key, {f"{now}_{time.perf_counter()}": now})
        client.expire(key, int(RATE_LIMIT_WINDOW_SECONDS + 5))
    except HTTPException:
        raise
    except Exception:
        pass
