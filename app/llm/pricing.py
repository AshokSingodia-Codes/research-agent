# Pricing in USD per 1,000,000 tokens (input_price_per_1m, output_price_per_1m)
MODEL_PRICING: dict[tuple[str, str], tuple[float, float]] = {
    # Anthropic
    ("anthropic", "claude-haiku-5-5"): (0.80, 4.00),
    ("anthropic", "claude-3-5-haiku-20241022"): (0.80, 4.00),
    ("anthropic", "claude-sonnet-5-5"): (3.00, 15.00),
    ("anthropic", "claude-3-5-sonnet-20241022"): (3.00, 15.00),
    ("anthropic", "claude-3-opus-20240229"): (15.00, 75.00),
    # Groq (Free / Low Cost Tier)
    ("groq", "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("groq", "llama-3.1-8b-instant"): (0.05, 0.08),
    ("groq", "mixtral-8x7b-32768"): (0.24, 0.24),
    ("groq", "gemma2-9b-it"): (0.20, 0.20),
    # Fake / Local
    ("fake", "fake-model"): (0.0, 0.0),
}


def calculate_cost(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
    """Calculates exact USD cost for a given provider, model, and token counts."""
    key = (provider.lower(), model.lower())
    # Match exact or prefix/fallback
    if key in MODEL_PRICING:
        in_price, out_price = MODEL_PRICING[key]
    else:
        # Check if model name partially matches any known model
        matched = False
        in_price, out_price = (0.0, 0.0)
        for (p, m), (ip, op) in MODEL_PRICING.items():
            if p == provider.lower() and (m in model.lower() or model.lower() in m):
                in_price, out_price = ip, op
                matched = True
                break
        if not matched:
            # Default fallback for unknown models
            in_price, out_price = (0.0, 0.0)

    cost = (tokens_in / 1_000_000.0) * in_price + (tokens_out / 1_000_000.0) * out_price
    return round(cost, 6)
