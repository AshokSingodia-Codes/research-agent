import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM API Keys
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Search API Key
    tavily_api_key: str = ""

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    use_fake_redis: bool = True

    # Budgets & Circuit Breakers
    max_subquestions: int = 4
    max_searches_per_subquestion: int = 3
    max_total_tokens: int = 50000
    wall_clock_timeout_seconds: int = 240
    max_revision_loops: int = 1

    # Mock toggles ("true", "false", or "auto")
    mock_llm: str = "auto"
    mock_search: str = "auto"

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key and not self.groq_api_key.startswith("your_"))

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key and not self.anthropic_api_key.startswith("your_"))

    @property
    def is_mock_llm(self) -> bool:
        if self.mock_llm.lower() == "true":
            return True
        if self.mock_llm.lower() == "false":
            return False
        # auto mode: mock only if neither Groq nor Anthropic is configured
        return not (self.has_groq or self.has_anthropic)

    @property
    def active_llm_provider(self) -> str:
        if self.is_mock_llm:
            return "mock"
        if self.has_groq:
            return f"groq:{self.groq_model}"
        if self.has_anthropic:
            return f"anthropic:{self.anthropic_model}"
        return "mock"

    @property
    def is_mock_search(self) -> bool:
        if self.mock_search.lower() == "true":
            return True
        if self.mock_search.lower() == "false":
            return False
        # auto mode: mock if no key or key is placeholder
        return not bool(self.tavily_api_key and not self.tavily_api_key.startswith("your_"))


settings = Settings()
