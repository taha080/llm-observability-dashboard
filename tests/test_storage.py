"""Unit tests for SQLiteStore CRUD, filtering, percentile stats, and demo seeding."""

from __future__ import annotations

from pathlib import Path

from llm_obs.models import CallEventCreate
from llm_obs.storage import SQLiteStore, _percentile


def test_percentile_helper_edge_cases() -> None:
    assert _percentile([], 50.0) == 0.0
    assert _percentile([42.5], 95.0) == 42.5
    assert _percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50.0) == 30.0
    assert _percentile([10.0, 20.0, 30.0, 40.0, 50.0], 95.0) == 48.0


def test_sqlite_store_record_and_get_call(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=tmp_path / "test.db")
    record = store.record_call(
        CallEventCreate(
            provider="groq",
            model="openai/gpt-oss-20b",
            prompt_tokens=120,
            completion_tokens=280,
            latency_ms=185.4,
            status="ok",
            prompt_preview="Hello world",
            response_preview="Hi there!",
            tags=["rag", "rag", "prod"],
            metadata={"user": "taha"},
        )
    )
    assert record.id == 1
    assert record.total_tokens == 400
    assert record.estimated_cost_usd > 0.0
    assert record.tags == ["rag", "prod"]
    assert record.metadata == {"user": "taha"}

    fetched = store.get_call(record.id)
    assert fetched is not None
    assert fetched.trace_id == record.trace_id
    assert store.get_call(9999) is None


def test_sqlite_store_filters_and_search(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=tmp_path / "filters.db")
    store.record_call(
        CallEventCreate(
            provider="groq",
            model="llama-3.3-70b-versatile",
            prompt_tokens=100,
            completion_tokens=200,
            latency_ms=310.0,
            status="ok",
            prompt_preview="Analyze SQLite WAL indexing",
            tags=["backend", "fast"],
            trace_id="trace-alpha",
        )
    )
    store.record_call(
        CallEventCreate(
            provider="ollama",
            model="llama3.2",
            prompt_tokens=80,
            completion_tokens=0,
            latency_ms=950.0,
            status="error",
            error_type="TimeoutError",
            error_message="Local Ollama timed out",
            prompt_preview="Generate Python unit tests",
            tags=["local", "error"],
            trace_id="trace-beta",
        )
    )

    assert store.count_calls() == 2
    assert store.count_calls(provider="groq") == 1
    assert store.count_calls(status="error") == 1
    assert store.count_calls(tag="backend") == 1
    assert store.count_calls(min_latency_ms=500.0) == 1
    assert store.count_calls(trace_id="trace-alpha") == 1

    results = store.list_calls(q="SQLite WAL")
    assert len(results) == 1
    assert results[0].model == "llama-3.3-70b-versatile"


def test_sqlite_store_empty_stats_and_populated_stats() -> None:
    store = SQLiteStore(db_path=":memory:")
    empty = store.get_stats()
    assert empty.total_calls == 0
    assert empty.p95_latency_ms == 0.0

    for lat in [100.0, 200.0, 300.0, 400.0, 1000.0]:
        store.record_call(
            CallEventCreate(
                provider="groq",
                model="openai/gpt-oss-20b",
                prompt_tokens=100,
                completion_tokens=100,
                latency_ms=lat,
                status="error" if lat == 1000.0 else "ok",
            )
        )

    stats = store.get_stats()
    assert stats.total_calls == 5
    assert stats.success_calls == 4
    assert stats.error_calls == 1
    assert stats.error_rate_pct == 20.0
    assert stats.total_tokens == 1000
    assert stats.p50_latency_ms == 300.0
    assert stats.p95_latency_ms > 800.0
    assert len(stats.models) == 1
    assert len(stats.providers) == 1
    store.close()


def test_sqlite_store_timeseries_and_seed_and_clear(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=tmp_path / "seed.db")
    inserted = store.seed_demo_data(count=25, seed=7)
    assert inserted == 25
    assert store.count_calls() == 25

    hourly = store.get_timeseries(granularity="hour", limit=12)
    assert len(hourly) >= 1
    daily = store.get_timeseries(granularity="day", limit=7)
    assert len(daily) >= 1

    # Seed with reset=True replaces records
    store.seed_demo_data(count=10, seed=9, reset=True)
    assert store.count_calls() == 10

    deleted = store.clear_calls()
    assert deleted == 10
    assert store.count_calls() == 0
