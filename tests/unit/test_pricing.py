import pytest

from app.budget.guard import BudgetExceeded, BudgetGuard
from app.llm.pricing import calculate_cost, get_unknown_model_count, reset_unknown_model_count


@pytest.mark.unit
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

    # 3. Fake provider default model is $0.00
    cost_fake = calculate_cost("fake", "fake-model", tokens_in=500_000, tokens_out=500_000)
    assert cost_fake == 0.0


@pytest.mark.unit
def test_unknown_model_conservative_fallback_pricing():
    """Validates that unknown models cost > 0 using conservative fallback and increment counter."""
    reset_unknown_model_count()
    initial_count = get_unknown_model_count()

    # Unknown model with 1,000 input tokens and 1,000 output tokens
    cost = calculate_cost("groq", "nonexistent-model-xyz", tokens_in=1000, tokens_out=1000)
    # Conservative fallback is $15/M in and $75/M out: (1000*15 + 1000*75)/1e6 = 0.09
    assert cost > 0.0
    assert cost == 0.09
    assert get_unknown_model_count() == initial_count + 1


@pytest.mark.unit
def test_unknown_model_trips_usd_budget():
    """Validates that the conservative fallback pricing on an unknown model trips the USD budget."""
    # Set a small USD budget of $0.05
    guard = BudgetGuard(max_usd_cost=0.05, max_total_tokens=100_000)

    # Unknown model charges 1,000 in, 1,000 out -> costs $0.09 > $0.05
    cost = calculate_cost("unknown-provider", "mystery-llm", tokens_in=1000, tokens_out=1000)
    assert cost > 0.05

    with pytest.raises(BudgetExceeded) as exc_info:
        guard.check_usd_cost(incoming_cost=cost)
    assert "USD cost budget exceeded" in str(exc_info.value)
    assert guard.is_exhausted is True
