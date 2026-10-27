"""Lightweight Python SDK, decorators, context managers, and Groq/Ollama client wrappers."""

from __future__ import annotations

import functools
import inspect
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from llm_obs.models import CallEventCreate, CallRecord
from llm_obs.storage import SQLiteStore


def load_dotenv_file(env_path: str | Path = ".env") -> dict[str, str]:
    """Load key=value pairs from a local .env file without overwriting existing env vars."""
    loaded: dict[str, str] = {}
    path = Path(env_path)
    if not path.is_file():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


def extract_usage_and_preview(response: Any) -> dict[str, Any]:
    """Extract token counts, model name, and text preview from Groq, OpenAI, or Ollama payloads."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens: int | None = None
    model: str | None = None
    preview: str | None = None

    if isinstance(response, dict):
        model = response.get("model")
        usage = response.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion_tokens = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            if usage.get("total_tokens") is not None:
                total_tokens = int(usage["total_tokens"])
        else:
            # Ollama /api/chat or /api/generate format
            prompt_tokens = int(response.get("prompt_eval_count") or 0)
            completion_tokens = int(response.get("eval_count") or 0)
            if prompt_tokens or completion_tokens:
                total_tokens = prompt_tokens + completion_tokens

        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    preview = msg["content"]
                elif isinstance(first.get("text"), str):
                    preview = first["text"]
        elif isinstance(response.get("message"), dict):
            content = response["message"].get("content")
            if isinstance(content, str):
                preview = content
        elif isinstance(response.get("response"), str):
            preview = response["response"]

    else:
        # Object-style SDK response (e.g., openai/groq Python SDK)
        model = getattr(response, "model", None)
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
            raw_total = getattr(usage_obj, "total_tokens", None)
            if raw_total is not None:
                total_tokens = int(raw_total)
        choices_obj = getattr(response, "choices", None)
        if isinstance(choices_obj, list) and choices_obj:
            first_choice = choices_obj[0]
            msg_obj = getattr(first_choice, "message", None)
            if msg_obj is not None:
                content = getattr(msg_obj, "content", None)
                if isinstance(content, str):
                    preview = content

    return {
        "model": model,
        "prompt_tokens": max(0, prompt_tokens),
        "completion_tokens": max(0, completion_tokens),
        "total_tokens": total_tokens,
        "response_preview": preview[:4000] if isinstance(preview, str) else None,
    }


def extract_prompt_preview(messages_or_prompt: Any) -> str | None:
    """Create a concise human-readable preview from chat messages or a prompt string."""
    if isinstance(messages_or_prompt, str):
        return messages_or_prompt[:4000]
    if isinstance(messages_or_prompt, list):
        parts: list[str] = []
        for item in messages_or_prompt:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(f"[{role}] {content}")
        if parts:
            return "\n".join(parts)[:4000]
    return None


class SpanTracker:
    """Mutable context yielded inside `observer.trace_call(...)`."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        prompt_preview: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens: int | None = None
        self.prompt_preview = prompt_preview
        self.response_preview: str | None = None
        self.tags = list(tags or [])
        self.metadata = dict(metadata or {})
        self.trace_id = trace_id or uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.record: CallRecord | None = None

    def set_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = max(0, int(prompt_tokens))
        self.completion_tokens = max(0, int(completion_tokens))
        self.total_tokens = total_tokens

    def set_response(self, response: Any) -> None:
        extracted = extract_usage_and_preview(response)
        if extracted["model"]:
            self.model = str(extracted["model"])
        self.prompt_tokens = extracted["prompt_tokens"]
        self.completion_tokens = extracted["completion_tokens"]
        self.total_tokens = extracted["total_tokens"]
        if extracted["response_preview"]:
            self.response_preview = extracted["response_preview"]


class LLMObserver:
    """Primary SDK entrypoint for recording LLM calls to local SQLite."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        store: SQLiteStore | None = None,
        default_tags: list[str] | None = None,
    ) -> None:
        load_dotenv_file()
        resolved_db = db_path or os.environ.get("LLM_OBS_DB_PATH", "llm_observability.db")
        self.store = store or SQLiteStore(db_path=resolved_db)
        self.default_tags = list(default_tags or [])

    def log_call(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        latency_ms: float = 0.0,
        status: str = "ok",
        error_type: str | None = None,
        error_message: str | None = None,
        prompt_preview: str | None = None,
        response_preview: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CallRecord:
        combined_tags = [*self.default_tags, *(tags or [])]
        event = CallEventCreate(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status="error" if status == "error" else "ok",
            error_type=error_type,
            error_message=error_message,
            prompt_preview=prompt_preview,
            response_preview=response_preview,
            trace_id=trace_id,
            span_id=span_id,
            tags=combined_tags,
            metadata=metadata or {},
        )
        return self.store.record_call(event)

    @contextmanager
    def trace_call(
        self,
        *,
        provider: str = "groq",
        model: str = "openai/gpt-oss-20b",
        prompt_preview: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Iterator[SpanTracker]:
        tracker = SpanTracker(
            provider=provider,
            model=model,
            prompt_preview=prompt_preview,
            tags=tags,
            metadata=metadata,
            trace_id=trace_id,
        )
        start = time.perf_counter()
        status = "ok"
        error_type: str | None = None
        error_message: str | None = None
        try:
            yield tracker
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            error_message = str(exc)[:2000]
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            tracker.record = self.log_call(
                provider=tracker.provider,
                model=tracker.model,
                prompt_tokens=tracker.prompt_tokens,
                completion_tokens=tracker.completion_tokens,
                total_tokens=tracker.total_tokens,
                latency_ms=elapsed_ms,
                status=status,
                error_type=error_type,
                error_message=error_message,
                prompt_preview=tracker.prompt_preview,
                response_preview=tracker.response_preview,
                trace_id=tracker.trace_id,
                span_id=tracker.span_id,
                tags=tracker.tags,
                metadata=tracker.metadata,
            )

    def observe(
        self,
        *,
        provider: str = "groq",
        model: str = "openai/gpt-oss-20b",
        tags: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that instruments sync or async functions returning an LLM response."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    prompt_prev = extract_prompt_preview(
                        kwargs.get("messages")
                        or kwargs.get("prompt")
                        or (args[0] if args else None)
                    )
                    call_model = str(kwargs.get("model") or model)
                    with self.trace_call(
                        provider=provider,
                        model=call_model,
                        prompt_preview=prompt_prev,
                        tags=tags,
                    ) as span:
                        result = await func(*args, **kwargs)
                        span.set_response(result)
                        return result

                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                prompt_prev = extract_prompt_preview(
                    kwargs.get("messages") or kwargs.get("prompt") or (args[0] if args else None)
                )
                call_model = str(kwargs.get("model") or model)
                with self.trace_call(
                    provider=provider,
                    model=call_model,
                    prompt_preview=prompt_prev,
                    tags=tags,
                ) as span:
                    result = func(*args, **kwargs)
                    span.set_response(result)
                    return result

            return sync_wrapper

        return decorator


def observe_llm_call(
    *,
    observer: LLMObserver,
    provider: str = "groq",
    model: str = "openai/gpt-oss-20b",
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Standalone decorator helper bound to an LLMObserver instance."""
    return observer.observe(provider=provider, model=model, tags=tags)


class GroqObservedClient:
    """Observed HTTP client for Groq's OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        observer: LLMObserver,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        http_client: httpx.Client | None = None,
        default_tags: list[str] | None = None,
    ) -> None:
        load_dotenv_file()
        self.observer = observer
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(timeout=30.0)
        self.default_tags = list(default_tags or ["groq"])

    def chat_completions(
        self,
        *,
        messages: list[dict[str, str]],
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        prompt_preview = extract_prompt_preview(messages)
        merged_tags = [*self.default_tags, *(tags or [])]
        with self.observer.trace_call(
            provider="groq",
            model=model,
            prompt_preview=prompt_preview,
            tags=merged_tags,
        ) as span:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. Add it to your local .env file or pass api_key."
                )
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            span.set_response(data)
            return data


class OllamaObservedClient:
    """Observed HTTP client for 100% free local Ollama inference."""

    def __init__(
        self,
        observer: LLMObserver,
        *,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        default_tags: list[str] | None = None,
    ) -> None:
        load_dotenv_file()
        self.observer = observer
        resolved_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = resolved_url.rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)
        self.default_tags = list(default_tags or ["ollama", "local"])

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str = "llama3.2",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        prompt_preview = extract_prompt_preview(messages)
        merged_tags = [*self.default_tags, *(tags or [])]
        with self.observer.trace_call(
            provider="ollama",
            model=model,
            prompt_preview=prompt_preview,
            tags=merged_tags,
        ) as span:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            span.set_response(data)
            return data
