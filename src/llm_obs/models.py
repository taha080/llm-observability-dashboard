"""Pydantic schemas and domain models for LLM observability events and metrics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CallEventCreate(BaseModel):
    """Payload for recording a single LLM invocation."""

    provider: str = Field(default="groq", min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=128)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    status: Literal["ok", "error"] = Field(default="ok")
    error_type: str | None = Field(default=None, max_length=128)
    error_message: str | None = Field(default=None, max_length=2000)
    prompt_preview: str | None = Field(default=None, max_length=4000)
    response_preview: str | None = Field(default=None, max_length=4000)
    trace_id: str | None = Field(default=None, max_length=128)
    span_id: str | None = Field(default=None, max_length=128)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    created_at: str | None = Field(default=None)


class CallRecord(BaseModel):
    """Stored LLM invocation record returned by SQLiteStore and REST endpoints."""

    id: int
    created_at: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    status: Literal["ok", "error"]
    error_type: str | None = None
    error_message: str | None = None
    prompt_preview: str | None = None
    response_preview: str | None = None
    trace_id: str
    span_id: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelBreakdown(BaseModel):
    """Aggregated telemetry per model."""

    provider: str
    model: str
    total_calls: int
    success_calls: int
    error_calls: int
    error_rate_pct: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    avg_latency_ms: float
    p95_latency_ms: float


class ProviderBreakdown(BaseModel):
    """Aggregated telemetry per LLM provider."""

    provider: str
    total_calls: int
    error_calls: int
    error_rate_pct: float
    total_tokens: int
    estimated_cost_usd: float
    avg_latency_ms: float


class TimeSeriesBucket(BaseModel):
    """Time-bucketed metrics for dashboard charts."""

    bucket: str
    total_calls: int
    error_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    avg_latency_ms: float


class DashboardStats(BaseModel):
    """Top-level KPIs and breakdowns for the observability dashboard."""

    total_calls: int
    success_calls: int
    error_calls: int
    error_rate_pct: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cloud_savings_usd: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_tokens_per_sec: float
    models: list[ModelBreakdown] = Field(default_factory=list)
    providers: list[ProviderBreakdown] = Field(default_factory=list)
