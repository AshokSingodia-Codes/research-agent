from app.llm.pricing import calculate_cost


def test_pricing_and_cost_accounting():
    """Validates dollar cost calculation per provider, model, and input/output tokens."""
    # 1. Anthropic Sonnet ($3.00 / $15.00 per MTok)
    cost_anthropic = calculate_cost(
        "anthropic",
        "claude-3-5-sonnet-20241022",
        tokens_in=1000,
        tokens_out=500,
    )
    # (1000 * 3.0 / 1e6) + (500 * 15.0 / 1e6) = 0.003 + 0.0075 = 0.0105
    assert cost_anthropic == 0.0105

    # 2. Groq Llama 3.3 70B ($0.59 / $0.79 per MTok)
    cost_groq = calculate_cost(
        "groq",
        "llama-3.3-70b-versatile",
        tokens_in=10000,
        tokens_out=2000,
    )
    # (10000 * 0.59 / 1e6) + (2000 * 0.79 / 1e6) = 0.0059 + 0.00158 = 0.00748
    assert cost_groq == 0.00748

    # 3. Fake provider / unknown model is $0.00
    cost_fake = calculate_cost("fake", "fake-model", tokens_in=500_000, tokens_out=500_000)
    assert cost_fake == 0.0


def test_pricing_zero_tokens_and_fallback():
    """Validates cost calculation with 0 tokens and fallback for unregistered models."""
    assert calculate_cost("anthropic", "claude-3-5-sonnet-20241022", 0, 0) == 0.0
    assert calculate_cost("groq", "nonexistent-model", 1000, 1000) == 0.0
    assert calculate_cost("unknown-provider", "model-x", 1000, 1000) == 0.0
