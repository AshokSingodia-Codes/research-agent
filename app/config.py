from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & Auth
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    tavily_api_key: str = ""
    api_key: str = "test-key-12345"  # System API key for endpoint auth
    env: str = "development"  # "development", "staging", "production"
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # Security & Integrity
    checkpoint_hmac_key: str = ""

    # Global LLM Model Defaults
    groq_model: str = "llama-3.3-70b-versatile"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    # Per-Agent Providers & Models ("groq", "anthropic", "fake")
    planner_provider: str = "groq"
    planner_model: str = "llama-3.3-70b-versatile"

    researcher_provider: str = "groq"
    researcher_model: str = "llama-3.3-70b-versatile"

    writer_provider: str = "groq"
    writer_model: str = "llama-3.3-70b-versatile"

    supervisor_provider: str = "groq"
    supervisor_model: str = "llama-3.3-70b-versatile"

    critic_provider: str = "groq"
    critic_model: str = "llama-3.3-70b-versatile"

    # Redis Persistence
    redis_url: str = "redis://localhost:6379/0"
    use_fake_redis: bool = True
    redis_ttl_seconds: int = 86400  # 24 hours

    # Concurrency & Networking
    concurrency_limit: int = 3
    max_parallel_researchers: int = 3
    max_concurrent_runs: int = 5
    worker_visibility_timeout: float = 60.0
    run_worker_inprocess: bool = False
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
