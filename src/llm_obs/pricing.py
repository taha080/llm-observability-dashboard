"""Token cost estimation and model pricing catalog for Groq, Ollama, and reference models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRate:
    """USD rate per 1,000,000 tokens for prompt and completion tokens."""

    provider: str
    input_per_1m: float
    output_per_1m: float
    is_local: bool = False


# Built-in rates (USD per 1M tokens) used to estimate API spend or equivalent value
DEFAULT_MODEL_RATES: dict[str, ModelRate] = {
    # Groq models
    "llama-3.3-70b-versatile": ModelRate("groq", 0.59, 0.79),
    "llama-3.1-8b-instant": ModelRate("groq", 0.05, 0.08),
    "openai/gpt-oss-20b": ModelRate("groq", 0.10, 0.50),
    "openai/gpt-oss-120b": ModelRate("groq", 0.15, 0.60),
    "mixtral-8x7b-32768": ModelRate("groq", 0.24, 0.24),
    "gemma2-9b-it": ModelRate("groq", 0.20, 0.20),
    "deepseek-r1-distill-llama-70b": ModelRate("groq", 0.75, 0.99),
    # Local Ollama models ($0.00 actual cost)
    "llama3.2": ModelRate("ollama", 0.0, 0.0, is_local=True),
    "llama3.2:3b": ModelRate("ollama", 0.0, 0.0, is_local=True),
    "qwen2.5-coder:7b": ModelRate("ollama", 0.0, 0.0, is_local=True),
    "mistral:7b": ModelRate("ollama", 0.0, 0.0, is_local=True),
    "phi4:14b": ModelRate("ollama", 0.0, 0.0, is_local=True),
    "nomic-embed-text": ModelRate("ollama", 0.0, 0.0, is_local=True),
    # Reference cloud models (for multi-provider observability)
    "gpt-4o": ModelRate("openai", 2.50, 10.00),
    "gpt-4o-mini": ModelRate("openai", 0.15, 0.60),
    "claude-3-5-sonnet": ModelRate("anthropic", 3.00, 15.00),
}

# Cloud reference rate ($2.50 / $10.00 per 1M tokens) to calculate "Cloud Cost Saved"
CLOUD_BENCHMARK_INPUT_PER_1M = 2.50
CLOUD_BENCHMARK_OUTPUT_PER_1M = 10.00


class PricingCatalog:
    """Registry of model pricing rates with prefix fallback and custom overrides."""

    def __init__(self, initial_rates: dict[str, ModelRate] | None = None) -> None:
        self._rates: dict[str, ModelRate] = dict(DEFAULT_MODEL_RATES)
        if initial_rates:
            self._rates.update(initial_rates)

    def register(
        self,
        model: str,
        provider: str,
        input_per_1m: float,
        output_per_1m: float,
        *,
        is_local: bool = False,
    ) -> ModelRate:
        if input_per_1m < 0 or output_per_1m < 0:
            raise ValueError("Token rates must be non-negative.")
        rate = ModelRate(
            provider=provider.strip().lower(),
            input_per_1m=float(input_per_1m),
            output_per_1m=float(output_per_1m),
            is_local=is_local,
        )
        self._rates[model.strip().lower()] = rate
        return rate

    def get_rate(self, model: str, provider: str | None = None) -> ModelRate:
        normalized_model = (model or "").strip().lower()
        normalized_provider = (provider or "").strip().lower()

        if normalized_model in self._rates:
            return self._rates[normalized_model]

        # Prefix match (e.g., "llama3.2:latest" -> "llama3.2")
        base_name = normalized_model.split(":")[0]
        if base_name in self._rates:
            return self._rates[base_name]

        for known_key, rate in self._rates.items():
            if normalized_model.startswith(known_key):
                return rate

        if normalized_provider == "ollama" or normalized_Provider_is_local(normalized_provider):
            return ModelRate(
                provider=normalized_provider or "ollama",
                input_per_1m=0.0,
                output_per_1m=0.0,
                is_local=True,
            )

        # Default fallback rate for unknown remote models
        return ModelRate(
            provider=normalized_provider or "unknown",
            input_per_1m=0.10,
            output_per_1m=0.30,
            is_local=False,
        )

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider: str | None = None,
    ) -> float:
        p_tokens = max(0, int(prompt_tokens))
        c_tokens = max(0, int(completion_tokens))
        rate = self.get_rate(model, provider=provider)
        cost = (p_tokens / 1_000_000.0) * rate.input_per_1m + (
            c_tokens / 1_000_000.0
        ) * rate.output_per_1m
        return round(cost, 8)

    def estimate_cloud_savings(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider: str | None = None,
    ) -> float:
        """Compare actual estimated cost against a commercial frontier model benchmark."""
        p_tokens = max(0, int(prompt_tokens))
        c_tokens = max(0, int(completion_tokens))
        benchmark = (p_tokens / 1_000_000.0) * CLOUD_BENCHMARK_INPUT_PER_1M + (
            c_tokens / 1_000_000.0
        ) * CLOUD_BENCHMARK_OUTPUT_PER_1M
        actual = self.estimate_cost(model, p_tokens, c_tokens, provider=provider)
        return round(max(0.0, benchmark - actual), 6)

    def list_rates(self) -> dict[str, ModelRate]:
        return dict(self._rates)


def normalized_Provider_is_local(provider: str) -> bool:
    return provider in {"ollama", "local", "llama.cpp", "vllm-local"}


DEFAULT_CATALOG = PricingCatalog()


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    provider: str | None = None,
) -> float:
    """Convenience function to estimate USD cost for a single LLM call."""
    return DEFAULT_CATALOG.estimate_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider=provider,
    )
