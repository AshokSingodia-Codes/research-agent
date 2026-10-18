from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from app.config import settings
from app.llm.base import RateLimitedError
from app.llm.client import LLMClient
from app.llm.fake_provider import FakeProvider


class SampleSchema(BaseModel):
    title: str
    count: int


@pytest.mark.unit
def test_get_model_for_agent():
    """Validates that each agent requests its configured per-agent model."""
    client = LLMClient()

    assert client.get_model_for_agent("planner") == settings.planner_model
    assert client.get_model_for_agent("researcher") == settings.researcher_model
    assert client.get_model_for_agent("writer") == settings.writer_model
    assert client.get_model_for_agent("supervisor") == settings.supervisor_model
    assert client.get_model_for_agent("critic") == settings.critic_model
    # Unknown agent falls back to groq_model
    assert client.get_model_for_agent("unknown_agent") == settings.groq_model


@pytest.mark.unit
def test_redact_secrets():
    """Validates secret redaction in LLM trace inputs."""
    client = LLMClient()
    secret_key = "gsk_testsecretkey1234567890"

    with patch.object(settings, "groq_api_key", secret_key):
        messages = [{"role": "user", "content": f"Query with api key {secret_key} included"}]
        redacted = client._redact_secrets(messages)
        assert secret_key not in redacted[0]["content"]
        assert "[REDACTED_API_KEY]" in redacted[0]["content"]


@pytest.mark.unit
def test_llm_client_generate_with_repair_fake_provider():
    """Validates generate_with_repair with FakeProvider returns parsed schema and StepTrace."""
    client = LLMClient()
    fake = FakeProvider()
    fake.queue_response(SampleSchema(title="Sample Test", count=5))
    client.set_override_provider(fake)

    try:
        parsed, trace = client.generate_with_repair(
            agent_name="planner",
            messages=[{"role": "user", "content": "Analyze topic"}],
            output_schema=SampleSchema,
            run_id="run_test_client",
        )
        assert parsed is not None
        assert parsed.title == "Sample Test"
        assert parsed.count == 5
        assert trace is not None
        assert trace.agent == "planner"
        assert trace.run_id == "run_test_client"
        assert trace.total_tokens >= 0
    finally:
        client.set_override_provider(None)


@pytest.mark.unit
def test_call_with_backoff_rate_limit_retry():
    """Validates _call_with_backoff retries on RateLimitedError."""
    client = LLMClient()
    mock_provider = MagicMock()
    mock_provider.provider_name = "mock_provider"

    success_resp = MagicMock()
    # First call raises RateLimitedError, second call succeeds
    mock_provider.generate_structured.side_effect = [
        RateLimitedError("Rate limit exceeded", retry_after=0.01),
        success_resp,
    ]

    with patch("time.sleep", return_value=None):
        resp = client._call_with_backoff(
            provider=mock_provider,
            messages=[{"role": "user", "content": "hello"}],
            output_schema=SampleSchema,
            model="mock-model",
            max_retries=2,
        )
        assert resp == success_resp
        assert mock_provider.generate_structured.call_count == 2
