import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import AuthError, RateLimitedError, TimeoutError
from app.llm.groq_provider import GroqProvider


class ProviderTestSchema(BaseModel):
    summary: str
    score: int


@pytest.mark.unit
def test_groq_provider_missing_key():
    """Validates that GroqProvider raises AuthError if API key is missing."""
    provider = GroqProvider(api_key="")
    with pytest.raises(AuthError):
        provider._get_client()


@pytest.mark.unit
def test_groq_provider_structured_success():
    """Validates GroqProvider parsing tool call into structured schema."""
    provider = GroqProvider(api_key="test_key")
    mock_client = MagicMock()

    mock_tool_call = MagicMock()
    mock_tool_call.function.arguments = json.dumps({"summary": "test passed", "score": 10})
    mock_choice = MagicMock()
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_choice.message.content = None

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 50
    mock_resp.usage.completion_tokens = 25
    mock_resp.usage.total_tokens = 75

    mock_client.chat.completions.create.return_value = mock_resp
    provider._client = mock_client

    res = provider.generate_structured(
        messages=[{"role": "user", "content": "analyze"}],
        output_schema=ProviderTestSchema,
        model="llama-3.3-70b-versatile",
    )

    assert res.parsed.summary == "test passed"
    assert res.parsed.score == 10
    assert res.tokens_in == 50
    assert res.tokens_out == 25
    assert res.cost_usd >= 0.0


@pytest.mark.unit
def test_groq_provider_rate_limit_error():
    """Validates GroqProvider maps RateLimitError into RateLimitedError."""
    import groq

    provider = GroqProvider(api_key="test_key")
    mock_client = MagicMock()

    mock_resp = MagicMock()
    mock_resp.headers = {"retry-after": "5.5"}
    mock_client.chat.completions.create.side_effect = groq.RateLimitError(
        message="rate limit", response=mock_resp, body=None
    )
    provider._client = mock_client

    with pytest.raises(RateLimitedError) as exc_info:
        provider.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            output_schema=ProviderTestSchema,
            model="llama-3.3-70b-versatile",
        )
    assert exc_info.value.retry_after >= 5.5


@pytest.mark.unit
def test_anthropic_provider_missing_key():
    """Validates that AnthropicProvider raises AuthError if API key is missing."""
    provider = AnthropicProvider(api_key="")
    with pytest.raises(AuthError):
        provider._get_client()


@pytest.mark.unit
def test_anthropic_provider_structured_success():
    """Validates AnthropicProvider parsing tool use block into structured schema."""
    provider = AnthropicProvider(api_key="test_key")
    mock_client = MagicMock()

    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "ProviderTestSchema"
    mock_block.input = {"summary": "anthropic output", "score": 9}

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage.input_tokens = 80
    mock_resp.usage.output_tokens = 40

    mock_client.messages.create.return_value = mock_resp
    provider._client = mock_client

    res = provider.generate_structured(
        messages=[{"role": "user", "content": "summarize"}],
        output_schema=ProviderTestSchema,
        model="claude-3-5-sonnet-20241022",
    )

    assert res.parsed.summary == "anthropic output"
    assert res.parsed.score == 9
    assert res.tokens_in == 80
    assert res.tokens_out == 40
    assert res.cost_usd >= 0.0


@pytest.mark.unit
def test_anthropic_provider_timeout_error():
    """Validates AnthropicProvider maps APITimeoutError to TimeoutError."""
    import anthropic

    provider = AnthropicProvider(api_key="test_key")
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = anthropic.APITimeoutError(request=MagicMock())
    provider._client = mock_client

    with pytest.raises(TimeoutError):
        provider.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            output_schema=ProviderTestSchema,
            model="claude-3-5-sonnet-20241022",
        )
