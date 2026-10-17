import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings

logger = logging.getLogger(__name__)


def get_chat_model(temperature: float = 0.2) -> Optional[BaseChatModel]:
    """
    Returns an initialized LLM Chat Model based on available configuration.
    Prioritizes Groq (for fast inference) or Anthropic Claude, or None for mock mode.
    """
    if settings.is_mock_llm:
        return None

    # 1. Try Groq (ultra-fast inference)
    if settings.has_groq:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=settings.groq_model,
                temperature=temperature,
                api_key=settings.groq_api_key,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGroq ({e}). Checking Anthropic fallback.")

    # 2. Try Anthropic Claude
    if settings.has_anthropic:
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=temperature,
                api_key=settings.anthropic_api_key,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatAnthropic ({e}).")

    return None


# Backward-compatible alias
get_anthropic_chat_model = get_chat_model
