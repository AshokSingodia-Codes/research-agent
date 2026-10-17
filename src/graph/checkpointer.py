import json
import logging
from typing import Any, Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from src.config import settings

logger = logging.getLogger(__name__)


def get_checkpointer() -> BaseCheckpointSaver:
    """
    Returns a configured checkpointer for LangGraph state persistence.
    Uses Redis if available and configured, or MemorySaver as an in-process fallback.
    """
    if settings.use_fake_redis:
        return MemorySaver()

    try:
        import redis
        client = redis.Redis.from_url(settings.redis_url)
        client.ping()
        # Redis is active and reachable
        from langgraph.checkpoint.redis import RedisSaver
        return RedisSaver(client)
    except Exception as e:
        logger.warning(
            f"Redis checkpointer could not connect to {settings.redis_url} ({e}). "
            "Falling back to MemorySaver."
        )
        return MemorySaver()
