"""Command-line interface for llm-observability-dashboard (`llm-obs`)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import uvicorn

from llm_obs.server import create_app
from llm_obs.storage import SQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-obs",
        description="100% free local LLM observability dashboard, SQLite store, and proxy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Run the FastAPI observability dashboard and proxy")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    serve_p.add_argument("--db", default="llm_observability.db", help="SQLite database path")
    serve_p.add_argument(
        "--seed", action="store_true", help="Seed demo data on startup if database is empty"
    )

    seed_p = sub.add_parser("seed", help="Populate realistic Groq & Ollama demo call logs")
    seed_p.add_argument("--count", type=int, default=60, help="Number of calls to generate")
    seed_p.add_argument("--db", default="llm_observability.db", help="SQLite database path")
    seed_p.add_argument(
        "--reset", action="store_true", help="Clear existing records before seeding"
    )

    stats_p = sub.add_parser("stats", help="Print aggregated telemetry statistics")
    stats_p.add_argument("--db", default="llm_observability.db", help="SQLite database path")
    stats_p.add_argument("--provider", default=None, help="Filter by provider (groq, ollama)")
    stats_p.add_argument("--json", action="store_true", help="Output JSON format")

    tail_p = sub.add_parser("tail", help="Inspect the most recent recorded LLM calls")
    tail_p.add_argument("--limit", type=int, default=10, help="Number of calls to display")
    tail_p.add_argument("--provider", default=None, help="Filter by provider")
    tail_p.add_argument("--status", choices=["ok", "error"], default=None, help="Filter by status")
    tail_p.add_argument("--db", default="llm_observability.db", help="SQLite database path")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "seed":
        store = SQLiteStore(db_path=args.db)
        inserted = store.seed_demo_data(count=args.count, reset=args.reset)
        total = store.count_calls()
        print(f"Seeded {inserted} demo LLM calls into {args.db} (total records: {total}).")
        return 0

    if args.command == "stats":
        store = SQLiteStore(db_path=args.db)
        stats = store.get_stats(provider=args.provider)
        if args.json:
            print(json.dumps(stats.model_dump(), indent=2))
            return 0
        print("=== LLM Observability Summary ===")
        print(
            f"Total Calls      : {stats.total_calls} "
            f"({stats.success_calls} ok, {stats.error_calls} errors, "
            f"{stats.error_rate_pct}% error rate)"
        )
        print(
            f"Total Tokens     : {stats.total_tokens:,} "
            f"(prompt: {stats.prompt_tokens:,}, completion: {stats.completion_tokens:,})"
        )
        print(
            f"Estimated Cost   : ${stats.estimated_cost_usd:.6f} "
            f"(Cloud benchmark saved: ${stats.cloud_savings_usd:.4f})"
        )
        print(
            f"Latency (ms)     : avg={stats.avg_latency_ms} | "
            f"p50={stats.p50_latency_ms} | p95={stats.p95_latency_ms} | p99={stats.p99_latency_ms}"
        )
        if stats.models:
            print("\nModel Breakdown:")
            for m in stats.models:
                print(
                    f"  - [{m.provider}] {m.model}: {m.total_calls} calls, "
                    f"{m.total_tokens:,} tokens, p95={m.p95_latency_ms}ms, "
                    f"${m.estimated_cost_usd:.5f}"
                )
        return 0

    if args.command == "tail":
        store = SQLiteStore(db_path=args.db)
        calls = store.list_calls(provider=args.provider, status=args.status, limit=args.limit)
        if not calls:
            print("No LLM calls recorded.")
            return 0
        for c in calls:
            print(
                f"#{c.id} [{c.created_at[:19]}] {c.provider}/{c.model} | "
                f"{c.total_tokens} tok | {c.latency_ms}ms | "
                f"${c.estimated_cost_usd:.6f} | {c.status.upper()}"
            )
        return 0

    if args.command == "serve":
        store = SQLiteStore(db_path=args.db)
        if args.seed and store.count_calls() == 0:
            store.seed_demo_data(count=60)
        app = create_app(store=store)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
