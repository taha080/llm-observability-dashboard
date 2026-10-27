"""Unit tests for LLMObserver, decorators, .env loading, and Groq/Ollama client wrappers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llm_obs.sdk import (
    GroqObservedClient,
    LLMObserver,
    OllamaObservedClient,
    extract_prompt_preview,
    extract_usage_and_preview,
    load_dotenv_file,
    observe_llm_call,
)


def test_load_dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\nCUSTOM_OBS_VAR='hello-obs'\nOLLAMA_BASE_URL=http://127.0.0.1:11434\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CUSTOM_OBS_VAR", raising=False)
    loaded = load_dotenv_file(env_file)
    assert loaded["CUSTOM_OBS_VAR"] == "hello-obs"


def test_extract_usage_from_openai_ollama_and_object() -> None:
    groq_dict = {
        "model": "openai/gpt-oss-20b",
        "usage": {"prompt_tokens": 15, "completion_tokens": 35, "total_tokens": 50},
        "choices": [{"message": {"content": "Groq answer"}}],
    }
    res1 = extract_usage_and_preview(groq_dict)
    assert res1["prompt_tokens"] == 15
    assert res1["completion_tokens"] == 35
    assert res1["response_preview"] == "Groq answer"

    ollama_dict = {
        "model": "llama3.2",
        "prompt_eval_count": 22,
        "eval_count": 48,
        "message": {"content": "Ollama local answer"},
    }
    res2 = extract_usage_and_preview(ollama_dict)
    assert res2["prompt_tokens"] == 22
    assert res2["completion_tokens"] == 48
    assert res2["total_tokens"] == 70
    assert res2["response_preview"] == "Ollama local answer"

    sdk_obj = SimpleNamespace(
        model="llama-3.3-70b-versatile",
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=18, total_tokens=30),
        choices=[SimpleNamespace(message=SimpleNamespace(content="SDK object text"))],
    )
    res3 = extract_usage_and_preview(sdk_obj)
    assert res3["prompt_tokens"] == 12
    assert res3["completion_tokens"] == 18
    assert res3["response_preview"] == "SDK object text"


def test_extract_prompt_preview() -> None:
    assert extract_prompt_preview("Direct prompt") == "Direct prompt"
    msgs = [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"}]
    assert extract_prompt_preview(msgs) == "[system] Be brief\n[user] Hi"
    assert extract_prompt_preview(12345) is None


def test_observer_trace_call_success_and_exception(tmp_path: Path) -> None:
    observer = LLMObserver(db_path=tmp_path / "sdk.db", default_tags=["sdk-test"])

    with observer.trace_call(
        provider="groq", model="openai/gpt-oss-20b", prompt_preview="Ping"
    ) as span:
        span.set_tokens(prompt_tokens=10, completion_tokens=20)
        span.response_preview = "Pong"

    assert span.record is not None
    assert span.record.status == "ok"
    assert span.record.total_tokens == 30
    assert "sdk-test" in span.record.tags

    with pytest.raises(RuntimeError, match="rate limit hit"):
        with observer.trace_call(provider="groq", model="openai/gpt-oss-20b"):
            raise RuntimeError("rate limit hit")

    calls = observer.store.list_calls()
    assert len(calls) == 2
    err_call = next(c for c in calls if c.status == "error")
    assert err_call.error_type == "RuntimeError"
    assert "rate limit hit" in (err_call.error_message or "")


def test_sync_and_async_observe_decorator(tmp_path: Path) -> None:
    import asyncio

    observer = LLMObserver(db_path=tmp_path / "dec.db")

    @observe_llm_call(observer=observer, provider="groq", model="llama-3.1-8b-instant")
    def fake_sync_llm(prompt: str) -> dict[str, Any]:
        return {
            "model": "llama-3.1-8b-instant",
            "usage": {"prompt_tokens": 11, "completion_tokens": 29},
            "choices": [{"message": {"content": f"Echo: {prompt}"}}],
        }

    @observer.observe(provider="ollama", model="llama3.2", tags=["async"])
    async def fake_async_llm(messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": "llama3.2",
            "prompt_eval_count": 14,
            "eval_count": 26,
            "message": {"content": "Async local reply"},
        }

    fake_sync_llm("Hello decorator")
    asyncio.run(fake_async_llm([{"role": "user", "content": "Async test"}]))

    stats = observer.store.get_stats()
    assert stats.total_calls == 2
    assert stats.total_tokens == 80


def test_groq_and_ollama_observed_clients_with_mock_transport(tmp_path: Path) -> None:
    observer = LLMObserver(db_path=tmp_path / "clients.db")

    def handler(request: httpx.Request) -> httpx.Response:
        if "groq.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-oss-20b",
                    "usage": {"prompt_tokens": 30, "completion_tokens": 50, "total_tokens": 80},
                    "choices": [{"message": {"content": "Mocked Groq completion"}}],
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "llama3.2",
                "prompt_eval_count": 25,
                "eval_count": 45,
                "message": {"content": "Mocked Ollama completion"},
            },
        )

    mock_http = httpx.Client(transport=httpx.MockTransport(handler))
    groq_client = GroqObservedClient(
        observer, api_key="test-only-not-a-real-key", http_client=mock_http
    )
    ollama_client = OllamaObservedClient(observer, http_client=mock_http)

    g_resp = groq_client.chat_completions(
        messages=[{"role": "user", "content": "Test Groq"}],
        model="openai/gpt-oss-20b",
    )
    assert g_resp["choices"][0]["message"]["content"] == "Mocked Groq completion"

    o_resp = ollama_client.chat(
        messages=[{"role": "user", "content": "Test Ollama"}],
        model="llama3.2",
    )
    assert o_resp["message"]["content"] == "Mocked Ollama completion"

    assert observer.store.count_calls() == 2

    # Missing API key logs an error call and raises ValueError
    no_key_client = GroqObservedClient(observer, api_key="", http_client=mock_http)
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        no_key_client.chat_completions(messages=[{"role": "user", "content": "Fail"}])

    assert observer.store.count_calls(status="error") == 1
