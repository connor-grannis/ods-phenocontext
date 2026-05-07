"""
Bedrock price table and USD cost computation.

Prices are per 1,000 tokens (input and output separately).
Update `priced_on` whenever rates are refreshed.
"""

from __future__ import annotations

# Bedrock on-demand prices as of 2026-05-07 for us-east-2.
# Source: https://aws.amazon.com/bedrock/pricing/
BEDROCK_PRICING: dict[str, dict[str, float | str]] = {
    "us.anthropic.claude-sonnet-4-6": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
        "priced_on": "2026-05-07",
    },
    "anthropic.claude-sonnet-4-6-20261231": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
        "priced_on": "2026-05-07",
    },
}


def compute_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a single Bedrock call.

    Raises KeyError if model_id is not in BEDROCK_PRICING.
    """
    pricing = BEDROCK_PRICING[model_id]
    input_cost = (input_tokens / 1000) * float(pricing["input_per_1k"])
    output_cost = (output_tokens / 1000) * float(pricing["output_per_1k"])
    return input_cost + output_cost
