"""FastAPI web dashboard, REST collector API, OpenTelemetry receiver, and LLM proxy."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse

from llm_obs.models import CallEventCreate, CallRecord, DashboardStats
from llm_obs.otel import parse_otlp_traces
from llm_obs.pricing import normalized_Provider_is_local
from llm_obs.sdk import extract_prompt_preview, extract_usage_and_preview, load_dotenv_file
from llm_obs.storage import SQLiteStore

STATIC_DIR = Path(__file__).parent / "static"


def _detect_provider(model: str, explicit_provider: str | None = None) -> str:
    if explicit_provider:
        return explicit_provider.strip().lower()
    lowered = model.strip().lower()
    if any(
        lowered.startswith(prefix)
        for prefix in ("llama3.2", "qwen", "mistral:", "phi4", "nomic-embed")
    ):
        return "ollama"
    return "groq"


def create_app(
    db_path: str | Path | None = None,
    *,
    store: SQLiteStore | None = None,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create and configure the FastAPI LLM Observability Dashboard application."""
    load_dotenv_file()
    resolved_db = db_path or os.environ.get("LLM_OBS_DB_PATH", "llm_observability.db")
    app_store = store or SQLiteStore(db_path=resolved_db)

    app = FastAPI(
        title="LLM Observability Dashboard",
        version="0.1.0",
        description=(
            "100% free local LLM observability collector, OpenAI/Groq/Ollama proxy, "
            "OpenTelemetry trace receiver, and SQLite analytics dashboard."
        ),
    )
    app.state.store = app_store
    app.state.http_transport = http_transport

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "db_path": app_store.db_path,
            "total_calls": app_store.count_calls(),
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard_index() -> HTMLResponse:
        index_file = STATIC_DIR / "index.html"
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

    @app.get("/api/stats", response_model=DashboardStats)
    def get_stats(
        provider: str | None = Query(default=None, description="Optional provider filter"),
    ) -> DashboardStats:
        return app_store.get_stats(provider=provider)

    @app.get("/api/timeseries")
    def get_timeseries(
        granularity: str = Query(default="hour", pattern="^(hour|day)$"),
        limit: int = Query(default=24, ge=1, le=168),
    ) -> dict[str, Any]:
        buckets = app_store.get_timeseries(granularity=granularity, limit=limit)
        return {
            "granularity": granularity,
            "limit": limit,
            "buckets": [b.model_dump() for b in buckets],
        }

    @app.get("/api/calls")
    def list_calls(
        provider: str | None = Query(default=None),
        model: str | None = Query(default=None),
        status: str | None = Query(default=None),
        tag: str | None = Query(default=None),
        q: str | None = Query(default=None),
        min_latency_ms: float | None = Query(default=None, ge=0.0),
        trace_id: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        items = app_store.list_calls(
            provider=provider,
            model=model,
            status=status,
            tag=tag,
            q=q,
            min_latency_ms=min_latency_ms,
            trace_id=trace_id,
            limit=limit,
            offset=offset,
        )
        total = app_store.count_calls(
            provider=provider,
            model=model,
            status=status,
            tag=tag,
            q=q,
            min_latency_ms=min_latency_ms,
            trace_id=trace_id,
        )
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [item.model_dump() for item in items],
        }

    @app.get("/api/calls/{call_id}", response_model=CallRecord)
    def get_call_detail(call_id: int) -> CallRecord:
        record = app_store.get_call(call_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Call ID {call_id} not found.")
        return record

    @app.post("/api/events", response_model=CallRecord, status_code=201)
    def ingest_event(event: CallEventCreate) -> CallRecord:
        return app_store.record_call(event)

    @app.post("/v1/traces")
    async def ingest_otel_traces(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object payload.")
        events = parse_otlp_traces(payload)
        records = [app_store.record_call(ev) for ev in events]
        return {
            "accepted": len(records),
            "call_ids": [r.id for r in records],
        }

    @app.post("/api/seed")
    def seed_demo(
        count: int = Query(default=60, ge=1, le=500),
        reset: bool = Query(default=False),
    ) -> dict[str, Any]:
        seeded = app_store.seed_demo_data(count=count, reset=reset)
        return {
            "seeded": seeded,
            "total_calls": app_store.count_calls(),
        }

    @app.delete("/api/calls")
    def clear_all_calls() -> dict[str, Any]:
        deleted = app_store.clear_calls()
        return {"deleted": deleted}

    @app.get("/api/pricing")
    def get_pricing_catalog() -> dict[str, Any]:
        rates = app_store.pricing.list_rates()
        return {
            model: {
                "provider": rate.provider,
                "input_per_1m_usd": rate.input_per_1m,
                "output_per_1m_usd": rate.output_per_1m,
                "is_local": rate.is_local,
            }
            for model, rate in rates.items()
        }

    @app.post("/v1/chat/completions")
    async def proxy_chat_completions(
        request: Request,
        response: Response,
        authorization: str | None = Header(default=None),
        x_llm_provider: str | None = Header(default=None),
        x_llm_tags: str | None = Header(default=None),
        x_trace_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        body = await request.json()
        if not isinstance(body, dict) or "model" not in body:
            raise HTTPException(
                status_code=400, detail="Request body must include 'model' and 'messages'."
            )

        model = str(body["model"])
        provider = _detect_provider(model, x_llm_provider)
        tags = (
            [t.strip() for t in x_llm_tags.split(",") if t.strip()]
            if x_llm_tags
            else ["proxy", provider]
        )
        prompt_preview = extract_prompt_preview(body.get("messages"))

        if normalized_Provider_is_local(provider):
            ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            target_url = f"{ollama_base}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
        else:
            groq_base = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip(
                "/"
            )
            target_url = f"{groq_base}/chat/completions"
            auth_header = authorization
            if not auth_header:
                env_key = os.environ.get("GROQ_API_KEY", "").strip()
                if env_key:
                    auth_header = f"Bearer {env_key}"
            if not auth_header:
                record = app_store.record_call(
                    CallEventCreate(
                        provider=provider,
                        model=model,
                        latency_ms=0.5,
                        status="error",
                        error_type="MissingAPIKeyError",
                        error_message="Missing Authorization header and GROQ_API_KEY in .env",
                        prompt_preview=prompt_preview,
                        trace_id=x_trace_id,
                        tags=tags,
                    )
                )
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "Missing Authorization header or GROQ_API_KEY in .env",
                        "call_id": record.id,
                    },
                )
            headers = {
                "Authorization": auth_header,
                "Content-Type": "application/json",
            }

        start = time.perf_counter()
        async with httpx.AsyncClient(transport=app.state.http_transport, timeout=60.0) as client:
            try:
                upstream_resp = await client.post(target_url, headers=headers, json=body)
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
                record = app_store.record_call(
                    CallEventCreate(
                        provider=provider,
                        model=model,
                        latency_ms=elapsed_ms,
                        status="error",
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:2000],
                        prompt_preview=prompt_preview,
                        trace_id=x_trace_id,
                        tags=tags,
                    )
                )
                raise HTTPException(
                    status_code=502,
                    detail={"error": str(exc), "call_id": record.id},
                ) from exc

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if upstream_resp.status_code >= 400:
            err_text = upstream_resp.text[:2000]
            record = app_store.record_call(
                CallEventCreate(
                    provider=provider,
                    model=model,
                    latency_ms=elapsed_ms,
                    status="error",
                    error_type=f"HTTP_{upstream_resp.status_code}",
                    error_message=err_text,
                    prompt_preview=prompt_preview,
                    trace_id=x_trace_id,
                    tags=tags,
                )
            )
            raise HTTPException(
                status_code=upstream_resp.status_code,
                detail={"error": err_text, "call_id": record.id},
            )

        data = upstream_resp.json()
        extracted = extract_usage_and_preview(data)
        record = app_store.record_call(
            CallEventCreate(
                provider=provider,
                model=str(extracted["model"] or model),
                prompt_tokens=extracted["prompt_tokens"],
                completion_tokens=extracted["completion_tokens"],
                total_tokens=extracted["total_tokens"],
                latency_ms=elapsed_ms,
                status="ok",
                prompt_preview=prompt_preview,
                response_preview=extracted["response_preview"],
                trace_id=x_trace_id,
                tags=tags,
            )
        )
        response.headers["X-LLM-Obs-Call-Id"] = str(record.id)
        response.headers["X-LLM-Obs-Trace-Id"] = record.trace_id
        return data

    return app
