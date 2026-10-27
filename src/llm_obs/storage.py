"""Local SQLite storage engine for LLM observability telemetry."""

from __future__ import annotations

import json
import math
import random
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from llm_obs.models import (
    CallEventCreate,
    CallRecord,
    DashboardStats,
    ModelBreakdown,
    ProviderBreakdown,
    TimeSeriesBucket,
)
from llm_obs.pricing import DEFAULT_CATALOG, PricingCatalog

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'ok',
    error_type TEXT,
    error_message TEXT,
    prompt_preview TEXT,
    response_preview TEXT,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_created_at ON llm_calls(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_provider ON llm_calls(provider);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls(model);
CREATE INDEX IF NOT EXISTS idx_llm_calls_status ON llm_calls(status);
CREATE INDEX IF NOT EXISTS idx_llm_calls_trace_id ON llm_calls(trace_id);
"""


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute percentile (0..100) from a pre-sorted list of floats."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return round(sorted_values[low], 2)
    weight = rank - low
    interpolated = sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight
    return round(interpolated, 2)


class SQLiteStore:
    """Thread-safe local SQLite store for LLM call logs and aggregated analytics."""

    def __init__(
        self,
        db_path: str | Path = "llm_observability.db",
        pricing: PricingCatalog | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.pricing = pricing or DEFAULT_CATALOG
        self._lock = threading.RLock()
        self._persistent_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
            finally:
                if self._persistent_conn is None:
                    conn.close()

    def close(self) -> None:
        with self._lock:
            if self._persistent_conn is not None:
                self._persistent_conn.close()
                self._persistent_conn = None

    def record_call(self, event: CallEventCreate) -> CallRecord:
        """Persist a single LLM call event and return the stored CallRecord."""
        provider = event.provider.strip().lower()
        model = event.model.strip()
        p_tokens = max(0, int(event.prompt_tokens))
        c_tokens = max(0, int(event.completion_tokens))
        total_tokens = (
            int(event.total_tokens)
            if event.total_tokens is not None and event.total_tokens >= p_tokens + c_tokens
            else p_tokens + c_tokens
        )
        cost = (
            float(event.estimated_cost_usd)
            if event.estimated_cost_usd is not None
            else self.pricing.estimate_cost(model, p_tokens, c_tokens, provider=provider)
        )
        created_at = event.created_at or datetime.now(UTC).isoformat()
        trace_id = event.trace_id or uuid.uuid4().hex
        span_id = event.span_id or uuid.uuid4().hex[:16]
        tags_json = json.dumps(list(dict.fromkeys(event.tags)))
        metadata_json = json.dumps(event.metadata or {})

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO llm_calls (
                        created_at, provider, model,
                        prompt_tokens, completion_tokens, total_tokens,
                        estimated_cost_usd, latency_ms, status,
                        error_type, error_message, prompt_preview, response_preview,
                        trace_id, span_id, tags_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        provider,
                        model,
                        p_tokens,
                        c_tokens,
                        total_tokens,
                        round(cost, 8),
                        round(float(event.latency_ms), 2),
                        event.status,
                        event.error_type,
                        event.error_message,
                        event.prompt_preview,
                        event.response_preview,
                        trace_id,
                        span_id,
                        tags_json,
                        metadata_json,
                    ),
                )
                conn.commit()
                row_id = int(cur.lastrowid or 0)
            finally:
                if self._persistent_conn is None:
                    conn.close()

        record = self.get_call(row_id)
        if record is None:
            raise RuntimeError("Failed to read back inserted call record.")
        return record

    def _row_to_record(self, row: sqlite3.Row) -> CallRecord:
        try:
            tags = json.loads(row["tags_json"] or "[]")
        except json.JSONDecodeError:
            tags = []
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}

        return CallRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            total_tokens=int(row["total_tokens"]),
            estimated_cost_usd=float(row["estimated_cost_usd"]),
            latency_ms=float(row["latency_ms"]),
            status="error" if row["status"] == "error" else "ok",
            error_type=row["error_type"],
            error_message=row["error_message"],
            prompt_preview=row["prompt_preview"],
            response_preview=row["response_preview"],
            trace_id=str(row["trace_id"]),
            span_id=str(row["span_id"]),
            tags=tags if isinstance(tags, list) else [],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def get_call(self, call_id: int) -> CallRecord | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM llm_calls WHERE id = ?", (int(call_id),)
                ).fetchone()
                return self._row_to_record(row) if row else None
            finally:
                if self._persistent_conn is None:
                    conn.close()

    def _build_where_clause(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        q: str | None = None,
        min_latency_ms: float | None = None,
        trace_id: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if provider:
            clauses.append("provider = ?")
            params.append(provider.strip().lower())
        if model:
            clauses.append("model = ?")
            params.append(model.strip())
        if status in {"ok", "error"}:
            clauses.append("status = ?")
            params.append(status)
        if trace_id:
            clauses.append("trace_id = ?")
            params.append(trace_id.strip())
        if min_latency_ms is not None:
            clauses.append("latency_ms >= ?")
            params.append(float(min_latency_ms))
        if tag:
            clauses.append("tags_json LIKE ?")
            params.append(f'%"{tag.strip()}"%')
        if q:
            search = f"%{q.strip()}%"
            clauses.append(
                "(model LIKE ? OR prompt_preview LIKE ? OR response_preview LIKE ? "
                "OR error_message LIKE ? OR trace_id LIKE ?)"
            )
            params.extend([search, search, search, search, search])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where_sql, params

    def list_calls(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        q: str | None = None,
        min_latency_ms: float | None = None,
        trace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[CallRecord]:
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        where_sql, params = self._build_where_clause(
            provider=provider,
            model=model,
            status=status,
            tag=tag,
            q=q,
            min_latency_ms=min_latency_ms,
            trace_id=trace_id,
        )

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT * FROM llm_calls
                    {where_sql}
                    ORDER BY created_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (*params, safe_limit, safe_offset),
                ).fetchall()
                return [self._row_to_record(r) for r in rows]
            finally:
                if self._persistent_conn is None:
                    conn.close()

    def count_calls(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        q: str | None = None,
        min_latency_ms: float | None = None,
        trace_id: str | None = None,
    ) -> int:
        where_sql, params = self._build_where_clause(
            provider=provider,
            model=model,
            status=status,
            tag=tag,
            q=q,
            min_latency_ms=min_latency_ms,
            trace_id=trace_id,
        )
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) AS cnt FROM llm_calls {where_sql}",
                    tuple(params),
                ).fetchone()
                return int(row["cnt"]) if row else 0
            finally:
                if self._persistent_conn is None:
                    conn.close()

    def get_stats(self, *, provider: str | None = None) -> DashboardStats:
        where_sql, params = self._build_where_clause(provider=provider)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT provider, model, prompt_tokens, completion_tokens,
                           total_tokens, estimated_cost_usd, latency_ms, status
                    FROM llm_calls
                    {where_sql}
                    """,
                    tuple(params),
                ).fetchall()
            finally:
                if self._persistent_conn is None:
                    conn.close()

        if not rows:
            return DashboardStats(
                total_calls=0,
                success_calls=0,
                error_calls=0,
                error_rate_pct=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                cloud_savings_usd=0.0,
                avg_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                avg_tokens_per_sec=0.0,
                models=[],
                providers=[],
            )

        total_calls = len(rows)
        error_calls = sum(1 for r in rows if r["status"] == "error")
        success_calls = total_calls - error_calls
        error_rate_pct = round((error_calls / total_calls) * 100.0, 2)

        prompt_tokens = sum(int(r["prompt_tokens"]) for r in rows)
        completion_tokens = sum(int(r["completion_tokens"]) for r in rows)
        total_tokens = sum(int(r["total_tokens"]) for r in rows)
        estimated_cost_usd = round(sum(float(r["estimated_cost_usd"]) for r in rows), 6)

        cloud_savings_usd = round(
            sum(
                self.pricing.estimate_cloud_savings(
                    str(r["model"]),
                    int(r["prompt_tokens"]),
                    int(r["completion_tokens"]),
                    provider=str(r["provider"]),
                )
                for r in rows
            ),
            4,
        )

        latencies = sorted(float(r["latency_ms"]) for r in rows)
        total_latency_ms = sum(latencies)
        avg_latency_ms = round(total_latency_ms / total_calls, 2)
        min_latency_ms = round(latencies[0], 2)
        max_latency_ms = round(latencies[-1], 2)
        p50_latency_ms = _percentile(latencies, 50.0)
        p95_latency_ms = _percentile(latencies, 95.0)
        p99_latency_ms = _percentile(latencies, 99.0)

        avg_tokens_per_sec = (
            round((completion_tokens / (total_latency_ms / 1000.0)), 2)
            if total_latency_ms > 0
            else 0.0
        )

        # Group by (provider, model)
        model_groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
        provider_groups: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            m_key = (str(r["provider"]), str(r["model"]))
            model_groups.setdefault(m_key, []).append(r)
            provider_groups.setdefault(str(r["provider"]), []).append(r)

        models_breakdown: list[ModelBreakdown] = []
        for (prov, mod), m_rows in model_groups.items():
            m_total = len(m_rows)
            m_err = sum(1 for r in m_rows if r["status"] == "error")
            m_lat = sorted(float(r["latency_ms"]) for r in m_rows)
            models_breakdown.append(
                ModelBreakdown(
                    provider=prov,
                    model=mod,
                    total_calls=m_total,
                    success_calls=m_total - m_err,
                    error_calls=m_err,
                    error_rate_pct=round((m_err / m_total) * 100.0, 2),
                    prompt_tokens=sum(int(r["prompt_tokens"]) for r in m_rows),
                    completion_tokens=sum(int(r["completion_tokens"]) for r in m_rows),
                    total_tokens=sum(int(r["total_tokens"]) for r in m_rows),
                    estimated_cost_usd=round(
                        sum(float(r["estimated_cost_usd"]) for r in m_rows), 6
                    ),
                    avg_latency_ms=round(sum(m_lat) / m_total, 2),
                    p95_latency_ms=_percentile(m_lat, 95.0),
                )
            )
        models_breakdown.sort(key=lambda m: (m.total_calls, m.total_tokens), reverse=True)

        providers_breakdown: list[ProviderBreakdown] = []
        for prov, p_rows in provider_groups.items():
            p_total = len(p_rows)
            p_err = sum(1 for r in p_rows if r["status"] == "error")
            p_lat = [float(r["latency_ms"]) for r in p_rows]
            providers_breakdown.append(
                ProviderBreakdown(
                    provider=prov,
                    total_calls=p_total,
                    error_calls=p_err,
                    error_rate_pct=round((p_err / p_total) * 100.0, 2),
                    total_tokens=sum(int(r["total_tokens"]) for r in p_rows),
                    estimated_cost_usd=round(
                        sum(float(r["estimated_cost_usd"]) for r in p_rows), 6
                    ),
                    avg_latency_ms=round(sum(p_lat) / p_total, 2),
                )
            )
        providers_breakdown.sort(key=lambda p: p.total_calls, reverse=True)

        return DashboardStats(
            total_calls=total_calls,
            success_calls=success_calls,
            error_calls=error_calls,
            error_rate_pct=error_rate_pct,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            cloud_savings_usd=cloud_savings_usd,
            avg_latency_ms=avg_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            p50_latency_ms=p50_latency_ms,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            avg_tokens_per_sec=avg_tokens_per_sec,
            models=models_breakdown,
            providers=providers_breakdown,
        )

    def get_timeseries(
        self,
        *,
        granularity: str = "hour",
        limit: int = 24,
    ) -> list[TimeSeriesBucket]:
        safe_limit = max(1, min(int(limit), 168))
        slice_len = 10 if granularity == "day" else 13  # YYYY-MM-DD or YYYY-MM-DDTHH

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                        SUBSTR(created_at, 1, {slice_len}) AS bucket,
                        COUNT(*) AS total_calls,
                        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_calls,
                        SUM(prompt_tokens) AS prompt_tokens,
                        SUM(completion_tokens) AS completion_tokens,
                        SUM(total_tokens) AS total_tokens,
                        SUM(estimated_cost_usd) AS estimated_cost_usd,
                        AVG(latency_ms) AS avg_latency_ms
                    FROM llm_calls
                    GROUP BY bucket
                    ORDER BY bucket DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
            finally:
                if self._persistent_conn is None:
                    conn.close()

        buckets = [
            TimeSeriesBucket(
                bucket=str(r["bucket"]),
                total_calls=int(r["total_calls"]),
                error_calls=int(r["error_calls"] or 0),
                prompt_tokens=int(r["prompt_tokens"] or 0),
                completion_tokens=int(r["completion_tokens"] or 0),
                total_tokens=int(r["total_tokens"] or 0),
                estimated_cost_usd=round(float(r["estimated_cost_usd"] or 0.0), 6),
                avg_latency_ms=round(float(r["avg_latency_ms"] or 0.0), 2),
            )
            for r in rows
        ]
        buckets.reverse()
        return buckets

    def clear_calls(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM llm_calls")
                conn.commit()
                return int(cur.rowcount or 0)
            finally:
                if self._persistent_conn is None:
                    conn.close()

    def seed_demo_data(self, count: int = 60, *, seed: int = 42, reset: bool = False) -> int:
        """Populate realistic Groq and local Ollama telemetry across the past 24 hours."""
        if reset:
            self.clear_calls()

        rng = random.Random(seed)
        now = datetime.now(UTC)
        demo_specs = [
            ("groq", "openai/gpt-oss-20b", 160, 420, 210.0, ["rag", "production"]),
            ("groq", "llama-3.3-70b-versatile", 320, 680, 380.0, ["agent", "mcp"]),
            ("groq", "llama-3.1-8b-instant", 110, 190, 95.0, ["router", "fast"]),
            ("ollama", "llama3.2", 240, 390, 640.0, ["local", "summarizer"]),
            ("ollama", "qwen2.5-coder:7b", 410, 720, 820.0, ["local", "codegen"]),
            ("groq", "deepseek-r1-distill-llama-70b", 290, 810, 520.0, ["reasoning", "eval"]),
        ]
        sample_prompts = [
            (
                "Summarize the Q3 architecture migration notes and list rollback risks.",
                "Key migration steps: 1) Dual-write to SQLite WAL store, 2) Verify p95 latency...",
            ),
            (
                "Write a Python FastAPI dependency that validates X-Trace-Id headers.",
                "Here is a clean FastAPI dependency using Header(default=None)...",
            ),
            (
                "Extract structured invoice totals from OCR chunk #14.",
                '{"vendor": "Acme Cloud", "subtotal": 0.00, "currency": "USD"}',
            ),
            (
                "Classify customer ticket intent and urgency level.",
                '{"intent": "billing_inquiry", "urgency": "low", "confidence": 0.96}',
            ),
            (
                "Explain why local sentence-transformers embeddings save cloud spend.",
                "Running all-MiniLM-L6-v2 on CPU embeds chunks locally with zero API cost...",
            ),
        ]
        sample_errors = [
            (
                "RateLimitError",
                "429 Too Many Requests: tokens per minute limit reached on free tier",
            ),
            ("TimeoutError", "Upstream inference timed out after 10000ms"),
            ("ConnectionError", "Local Ollama server at http://localhost:11434 is unreachable"),
        ]

        inserted = 0
        safe_count = max(1, min(int(count), 1000))
        for i in range(safe_count):
            prov, model, base_p, base_c, base_lat, base_tags = rng.choice(demo_specs)
            hours_ago = rng.uniform(0.05, 23.5)
            ts = (now - timedelta(hours=hours_ago)).replace(microsecond=0).isoformat()
            prompt_text, resp_text = rng.choice(sample_prompts)

            is_error = rng.random() < 0.08 or (
                i == safe_count - 1 and inserted > 5 and rng.random() < 0.5
            )
            if is_error:
                err_type, err_msg = rng.choice(sample_errors)
                p_tokens = max(20, int(rng.gauss(base_p, base_p * 0.25)))
                c_tokens = 0
                latency = round(max(45.0, rng.gauss(base_lat * 1.6, 180.0)), 2)
                event = CallEventCreate(
                    provider=prov,
                    model=model,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    latency_ms=latency,
                    status="error",
                    error_type=err_type,
                    error_message=err_msg,
                    prompt_preview=prompt_text,
                    response_preview=None,
                    trace_id=uuid.UUID(int=rng.getrandbits(128)).hex,
                    span_id=uuid.UUID(int=rng.getrandbits(128)).hex[:16],
                    tags=[*base_tags, "error"],
                    metadata={"environment": "demo", "attempt": rng.randint(1, 3)},
                    created_at=ts,
                )
            else:
                p_tokens = max(24, int(rng.gauss(base_p, base_p * 0.3)))
                c_tokens = max(16, int(rng.gauss(base_c, base_c * 0.35)))
                latency = round(max(35.0, rng.gauss(base_lat, base_lat * 0.28)), 2)
                event = CallEventCreate(
                    provider=prov,
                    model=model,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    latency_ms=latency,
                    status="ok",
                    prompt_preview=prompt_text,
                    response_preview=resp_text,
                    trace_id=uuid.UUID(int=rng.getrandbits(128)).hex,
                    span_id=uuid.UUID(int=rng.getrandbits(128)).hex[:16],
                    tags=list(base_tags),
                    metadata={"environment": "demo", "temperature": 0.2},
                    created_at=ts,
                )

            self.record_call(event)
            inserted += 1

        return inserted
