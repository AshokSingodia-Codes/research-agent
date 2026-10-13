from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    tavily_api_key: str = ""
    api_key: str = "test-key-12345"  # System API key for endpoint auth

    # Per-Agent Providers ("anthropic", "groq", "fake")
    planner_provider: str = "anthropic"
    planner_model: str = "claude-haiku-5-5"

    researcher_provider: str = "anthropic"
    researcher_model: str = "claude-haiku-5-5"

    writer_provider: str = "anthropic"
    writer_model: str = "claude-sonnet-5-5"

    supervisor_provider: str = "anthropic"
    supervisor_model: str = "claude-haiku-5-5"

    critic_provider: str = "anthropic"
    critic_model: str = "claude-haiku-5-5"

    # Redis Persistence
    redis_url: str = "redis://localhost:6379/0"
    use_fake_redis: bool = True
    redis_ttl_seconds: int = 86400  # 24 hours

    # Concurrency & Networking
    concurrency_limit: int = 3
    request_timeout_seconds: float = 30.0

    # Hard Budgets
    max_subquestions: int = 5
    max_searches_per_subquestion: int = 3
    max_total_tokens: int = 50000
    max_usd_cost: float = 1.00
    wall_clock_timeout_seconds: float = 240.0
    max_repair_attempts: int = 2
    max_revision_loops: int = 1
    max_results_per_domain: int = 2
    extracted_token_cap: int = 2500

    # Citation Thresholds
    citation_coverage_threshold: float = 0.80

    # Mock Overrides for Tests ("true", "false", "auto")
    mock_search: str = "auto"
    mock_llm: str = "auto"

    @property
    def is_mock_llm(self) -> bool:
        val = str(self.mock_llm).lower()
        if val in ("true", "1", "yes"):
            return True
        if val in ("false", "0", "no"):
            return False
        # auto: mock if both Anthropic and Groq keys are absent
        return not bool(self.anthropic_api_key or self.groq_api_key)

    @property
    def is_mock_search(self) -> bool:
        val = str(self.mock_search).lower()
        if val in ("true", "1", "yes"):
            return True
        if val in ("false", "0", "no"):
            return False
        return not bool(self.tavily_api_key)


settings = Settings()
