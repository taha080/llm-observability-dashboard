"""Integration tests for FastAPI dashboard, proxy endpoints, OTel endpoint, and CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from llm_obs.cli import main as cli_main
from llm_obs.server import create_app


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_fastapi_dashboard_and_rest_endpoints(tmp_path: Path) -> None:
    db_file = tmp_path / "server.db"
    app = create_app(db_path=db_file)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            html = await client.get("/")
            assert html.status_code == 200
            assert "LLM Observability Dashboard" in html.text

            # Ingest an event
            ev_resp = await client.post(
                "/api/events",
                json={
                    "provider": "groq",
                    "model": "openai/gpt-oss-20b",
                    "prompt_tokens": 50,
                    "completion_tokens": 150,
                    "latency_ms": 210.5,
                    "status": "ok",
                    "prompt_preview": "Explain FastAPI",
                    "response_preview": "FastAPI is a modern Python web framework.",
                    "tags": ["api", "test"],
                },
            )
            assert ev_resp.status_code == 201
            call_id = ev_resp.json()["id"]

            # Inspect call detail
            detail = await client.get(f"/api/calls/{call_id}")
            assert detail.status_code == 200
            assert detail.json()["total_tokens"] == 200

            missing = await client.get("/api/calls/99999")
            assert missing.status_code == 404

            # Stats & timeseries & pricing
            stats = await client.get("/api/stats")
            assert stats.status_code == 200
            assert stats.json()["total_calls"] == 1

            ts = await client.get("/api/timeseries?granularity=hour&limit=12")
            assert ts.status_code == 200
            assert len(ts.json()["buckets"]) == 1

            pricing = await client.get("/api/pricing")
            assert pricing.status_code == 200
            assert "openai/gpt-oss-20b" in pricing.json()

            # OTel traces endpoint
            otel_resp = await client.post(
                "/v1/traces",
                json={
                    "resourceSpans": [
                        {
                            "scopeSpans": [
                                {
                                    "spans": [
                                        {
                                            "traceId": "trace123",
                                            "spanId": "span456",
                                            "name": "otel-call",
                                            "startTimeUnixNano": "1000000000",
                                            "endTimeUnixNano": "1150000000",
                                            "attributes": [
                                                {
                                                    "key": "gen_ai.system",
                                                    "value": {"stringValue": "ollama"},
                                                },
                                                {
                                                    "key": "gen_ai.request.model",
                                                    "value": {"stringValue": "llama3.2"},
                                                },
                                                {
                                                    "key": "gen_ai.usage.input_tokens",
                                                    "value": {"intValue": 30},
                                                },
                                                {
                                                    "key": "gen_ai.usage.output_tokens",
                                                    "value": {"intValue": 70},
                                                },
                                            ],
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
            )
            assert otel_resp.status_code == 200
            assert otel_resp.json()["accepted"] == 1

            # Seed & delete
            seed_resp = await client.post("/api/seed?count=12&reset=true")
            assert seed_resp.status_code == 200
            assert seed_resp.json()["total_calls"] == 12

            del_resp = await client.delete("/api/calls")
            assert del_resp.status_code == 200
            assert del_resp.json()["deleted"] == 12

    _run(scenario())


def test_proxy_chat_completions_groq_and_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def mock_upstream(request: httpx.Request) -> httpx.Response:
        if "11434" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "usage": {"prompt_tokens": 18, "completion_tokens": 42, "total_tokens": 60},
                    "choices": [{"message": {"content": "Local Ollama proxied response"}}],
                },
            )
        if request.headers.get("Authorization") == "Bearer gsk_fake_for_test":
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-oss-20b",
                    "usage": {"prompt_tokens": 25, "completion_tokens": 75, "total_tokens": 100},
                    "choices": [{"message": {"content": "Groq proxied response"}}],
                },
            )
        return httpx.Response(429, text="Rate limit exceeded on upstream")

    app = create_app(
        db_path=tmp_path / "proxy.db",
        http_transport=httpx.MockTransport(mock_upstream),
    )

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Missing Groq key returns 401 and records error call
            r_unauth = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert r_unauth.status_code == 401

            # 2. Valid Groq Authorization header proxies and logs call
            r_groq = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer gsk_fake_for_test",
                    "X-LLM-Tags": "proxy-test,groq",
                },
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert r_groq.status_code == 200
            assert "X-LLM-Obs-Call-Id" in r_groq.headers

            # 3. Ollama local model requires no API key
            r_ollama = await client.post(
                "/v1/chat/completions",
                json={"model": "llama3.2", "messages": [{"role": "user", "content": "Local"}]},
            )
            assert r_ollama.status_code == 200
            assert (
                r_ollama.json()["choices"][0]["message"]["content"]
                == "Local Ollama proxied response"
            )

            # 4. Upstream 429 error is recorded as error call
            r_429 = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer trigger_429"},
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert r_429.status_code == 429

            calls = await client.get("/api/calls")
            assert calls.json()["total"] == 4

    _run(scenario())


def test_cli_seed_stats_and_tail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = str(tmp_path / "cli.db")

    rc_seed = cli_main(["seed", "--count", "15", "--db", db_path, "--reset"])
    assert rc_seed == 0
    out_seed = capsys.readouterr().out
    assert "Seeded 15 demo LLM calls" in out_seed

    rc_stats = cli_main(["stats", "--db", db_path])
    assert rc_stats == 0
    out_stats = capsys.readouterr().out
    assert "LLM Observability Summary" in out_stats

    rc_json = cli_main(["stats", "--db", db_path, "--json"])
    assert rc_json == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["total_calls"] == 15

    rc_tail = cli_main(["tail", "--limit", "5", "--db", db_path])
    assert rc_tail == 0
    out_tail = capsys.readouterr().out
    assert "#" in out_tail


def test_github_pages_static_bundle_sync() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    static_html = (repo_root / "src" / "llm_obs" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    docs_html = (repo_root / "docs" / "index.html").read_text(encoding="utf-8")
    root_html = (repo_root / "index.html").read_text(encoding="utf-8")

    assert static_html == docs_html
    assert static_html == root_html
    assert "Interactive Browser Mode (GitHub Pages)" in static_html
    assert "runLiveGroqCall" in static_html
    assert "recordSimulatedCall" in static_html
