import hmac
import time

from fastapi import HTTPException, Security, status
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


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validates API Key using constant-time comparison and enforces per-key rate limiting."""
    configured_key = settings.api_key

    # Production safety: reject access if production environment lacks configured API key
    if settings.env.lower() == "production" and (not configured_key or configured_key == "test-key-12345"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Production server misconfigured: secure API_KEY required.",
        )

    if configured_key:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-API-Key authentication header.",
            )
        # Constant-time comparison to prevent timing side-channel attacks
        if not hmac.compare_digest(api_key.encode("utf-8"), configured_key.encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API Key provided.",
            )

    caller_id = api_key or "anonymous"
    _check_general_rate_limit(caller_id)
    return caller_id


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
