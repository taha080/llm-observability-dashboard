"""Unit tests for token cost calculation and PricingCatalog."""

from __future__ import annotations

import pytest

from llm_obs.pricing import PricingCatalog, estimate_cost, normalized_Provider_is_local


def test_estimate_cost_known_groq_model() -> None:
    # openai/gpt-oss-20b: $0.10 / 1M input, $0.50 / 1M output
    cost = estimate_cost("openai/gpt-oss-20b", prompt_tokens=10_000, completion_tokens=20_000)
    assert cost == pytest.approx(0.001 + 0.01, rel=1e-6)


def test_estimate_cost_local_ollama_is_zero() -> None:
    cost = estimate_cost(
        "llama3.2", prompt_tokens=50_000, completion_tokens=50_000, provider="ollama"
    )
    assert cost == 0.0


def test_pricing_catalog_prefix_matching() -> None:
    catalog = PricingCatalog()
    rate = catalog.get_rate("llama3.2:latest", provider="ollama")
    assert rate.is_local is True
    assert rate.input_per_1m == 0.0
    assert rate.output_per_1m == 0.0


def test_pricing_catalog_custom_registration() -> None:
    catalog = PricingCatalog()
    registered = catalog.register(
        "custom-groq-model",
        provider="groq",
        input_per_1m=0.40,
        output_per_1m=0.80,
    )
    assert registered.provider == "groq"
    cost = catalog.estimate_cost("custom-groq-model", 1_000_000, 500_000)
    assert cost == pytest.approx(0.80, rel=1e-6)


def test_pricing_catalog_rejects_negative_rates() -> None:
    catalog = PricingCatalog()
    with pytest.raises(ValueError, match="non-negative"):
        catalog.register("bad-model", "groq", -0.1, 0.2)


def test_estimate_cloud_savings() -> None:
    catalog = PricingCatalog()
    savings = catalog.estimate_cloud_savings(
        "llama3.2", prompt_tokens=100_000, completion_tokens=100_000, provider="ollama"
    )
    # Benchmark: 0.1 * 2.50 + 0.1 * 10.00 = $1.25
    assert savings == pytest.approx(1.25, rel=1e-4)


def test_unknown_model_fallback_and_local_provider() -> None:
    catalog = PricingCatalog()
    local_rate = catalog.get_rate("my-unlisted-gguf", provider="local")
    assert local_rate.is_local is True
    assert local_rate.input_per_1m == 0.0

    remote_rate = catalog.get_rate("unlisted-remote-llm", provider="groq")
    assert remote_rate.is_local is False
    assert remote_rate.input_per_1m > 0.0
    assert normalized_Provider_is_local("ollama") is True
    assert normalized_Provider_is_local("groq") is False
