"""Additional unit tests covering models, pricing catalog, filtering, and edge cases."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from llm_obs.models import CallEventCreate
from llm_obs.otel import _extract_attr_value, parse_otlp_traces
from llm_obs.pricing import DEFAULT_MODEL_RATES, PricingCatalog, estimate_cost
from llm_obs.sdk import extract_usage_and_preview
from llm_obs.server import _detect_provider, create_app
from llm_obs.storage import SQLiteStore


@pytest.mark.parametrize(
    ("model", "provider", "expected_zero"),
    [
        ("llama-3.3-70b-versatile", "groq", False),
        ("llama-3.1-8b-instant", "groq", False),
        ("openai/gpt-oss-20b", "groq", False),
        ("openai/gpt-oss-120b", "groq", False),
        ("mixtral-8x7b-32768", "groq", False),
        ("gemma2-9b-it", "groq", False),
        ("deepseek-r1-distill-llama-70b", "groq", False),
        ("llama3.2", "ollama", True),
        ("qwen2.5-coder:7b", "ollama", True),
        ("mistral:7b", "ollama", True),
        ("phi4:14b", "ollama", True),
        ("nomic-embed-text", "ollama", True),
    ],
)
def test_builtin_model_rates(model: str, provider: str, expected_zero: bool) -> None:
    cost = estimate_cost(model, prompt_tokens=5_000, completion_tokens=5_000, provider=provider)
    if expected_zero:
        assert cost == 0.0
    else:
        assert cost > 0.0


def test_pricing_catalog_list_rates() -> None:
    catalog = PricingCatalog()
    rates = catalog.list_rates()
    assert len(rates) >= len(DEFAULT_MODEL_RATES)


def test_call_event_create_validation() -> None:
    with pytest.raises(ValidationError):
        CallEventCreate(provider="groq", model="", prompt_tokens=10)

    with pytest.raises(ValidationError):
        CallEventCreate(provider="groq", model="llama3.2", prompt_tokens=-5)


def test_detect_provider_heuristics() -> None:
    assert _detect_provider("llama3.2:latest") == "ollama"
    assert _detect_provider("qwen2.5-coder:7b") == "ollama"
    assert _detect_provider("phi4:14b") == "ollama"
    assert _detect_provider("openai/gpt-oss-20b") == "groq"
    assert _detect_provider("llama3.2", explicit_provider="GROQ") == "groq"


def test_otel_extract_attr_value_variants() -> None:
    assert _extract_attr_value({"stringValue": "abc"}) == "abc"
    assert _extract_attr_value({"intValue": "42"}) == 42
    assert _extract_attr_value({"intValue": "not-an-int"}) == 0
    assert _extract_attr_value({"doubleValue": "3.14"}) == 3.14
    assert _extract_attr_value({"doubleValue": "bad"}) == 0.0
    assert _extract_attr_value({"boolValue": True}) is True
    assert _extract_attr_value(
        {"arrayValue": {"values": [{"stringValue": "x"}, {"intValue": 2}]}}
    ) == ["x", 2]
    assert _extract_attr_value("raw-primitive") == "raw-primitive"
    assert parse_otlp_traces({"resourceSpans": "not-a-list"}) == []


def test_extract_usage_text_completion_and_generate_variants() -> None:
    legacy_choice = {
        "model": "mixtral-8x7b-32768",
        "usage": {"input_tokens": 12, "output_tokens": 28},
        "choices": [{"text": "Legacy completion text"}],
    }
    extracted = extract_usage_and_preview(legacy_choice)
    assert extracted["prompt_tokens"] == 12
    assert extracted["completion_tokens"] == 28
    assert extracted["response_preview"] == "Legacy completion text"

    ollama_gen = {
        "model": "mistral:7b",
        "prompt_eval_count": 9,
        "eval_count": 21,
        "response": "Ollama /api/generate text",
    }
    extracted_gen = extract_usage_and_preview(ollama_gen)
    assert extracted_gen["total_tokens"] == 30
    assert extracted_gen["response_preview"] == "Ollama /api/generate text"


def test_store_pagination_and_model_filter(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=tmp_path / "page.db")
    for i in range(6):
        store.record_call(
            CallEventCreate(
                provider="groq",
                model="model-a" if i % 2 == 0 else "model-b",
                prompt_tokens=10 + i,
                completion_tokens=20 + i,
                latency_ms=100.0 + i * 10,
            )
        )
    page1 = store.list_calls(limit=2, offset=0)
    page2 = store.list_calls(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0].id != page2[0].id

    only_a = store.list_calls(model="model-a")
    assert len(only_a) == 3
    assert store.count_calls(model="model-b") == 3


def test_proxy_network_exception_returns_502_and_logs_error(tmp_path: Path) -> None:
    import asyncio

    async def failing_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by upstream")

    app = create_app(
        db_path=tmp_path / "err502.db",
        http_transport=httpx.MockTransport(failing_transport),
    )

    async def run_check() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "llama3.2", "messages": [{"role": "user", "content": "Hi"}]},
            )
            assert resp.status_code == 502
            calls = await client.get("/api/calls?status=error")
            assert calls.json()["total"] == 1
            assert calls.json()["items"][0]["error_type"] == "ConnectError"

    asyncio.run(run_check())
