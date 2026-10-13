from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base error for all LLM provider failures."""

    def __init__(self, message: str, error_type: str = "generic", retry_after: float | None = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.retry_after = retry_after


class RateLimitedError(LLMError):
    def __init__(self, message: str, retry_after: float | None = 1.0):
        super().__init__(message, error_type="rate_limited", retry_after=retry_after)


class TimeoutError(LLMError):
    def __init__(self, message: str):
        super().__init__(message, error_type="timeout")


class InvalidOutputError(LLMError):
    def __init__(self, message: str, raw_output: str | None = None):
        super().__init__(message, error_type="invalid_output")
        self.raw_output = raw_output


class AuthError(LLMError):
    def __init__(self, message: str):
        super().__init__(message, error_type="auth")


class LLMResponse(BaseModel, Generic[T]):
    parsed: T
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw_text: str = ""
    provider: str
    model: str


class LLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of provider (e.g., 'anthropic', 'groq', 'fake')."""

    @abstractmethod
    def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_schema: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse[T]:
        """
        Generates a validated Pydantic model instance from structured output.
        Must return parsed object, token counts, cost, and latency.
        """
