# llm-observability-dashboard

**Live Interactive Demo:** [**https://taha080.github.io/llm-observability-dashboard/**](https://taha080.github.io/llm-observability-dashboard/)

A **100% free, local-first LLM observability toolkit, OpenAI/Groq/Ollama proxy, OpenTelemetry collector, and FastAPI analytics dashboard** backed by zero-config SQLite—plus a **standalone browser mode** that runs live on GitHub Pages.

Track every AI call across **Groq's free API** and **local Ollama models**—including prompt/completion token usage, estimated API cost, frontier cloud spend saved, p50/p95/p99 latency percentiles, throughput (`tokens/sec`), trace/span IDs, and error rates—without any paid SaaS accounts or external cloud databases.

---

## Live GitHub Pages Demo & Playground

Open [**https://taha080.github.io/llm-observability-dashboard/**](https://taha080.github.io/llm-observability-dashboard/) in your browser:

1. **Dual-Engine Architecture**: When served from FastAPI (`llm-obs serve`), the UI queries `/api/stats`, `/api/timeseries`, and `/api/calls` backed by local SQLite WAL storage. When opened statically on **GitHub Pages**, it automatically activates its **in-browser telemetry engine (`localStorage`)** pre-seeded with 60 realistic Groq and Ollama traces.
2. **⚡ Live Groq & Event Playground**:
   - **Live Groq API Call (Browser BYOK)**: Paste a free `gsk_...` Groq API key (stored only in your browser's `sessionStorage`, never sent to any intermediate server) to fire a real request to `https://api.groq.com/openai/v1/chat/completions` and watch its real latency, tokens, cost, and response trace appear on the dashboard.
   - **Zero-Key Event Simulator**: Log custom Groq or local Ollama spans (with custom token counts, latency, and `ok` / `error` status) and watch p50/p95/p99 percentiles and hourly charts recalculate in real time.

---

## Key Features

- **Lightweight Python SDK (`llm_obs`)**:
  - Context manager (`observer.trace_call(...)`) and sync/async `@observer.observe(...)` decorator.
  - Drop-in `GroqObservedClient` and `OllamaObservedClient` HTTP wrappers that automatically measure latency, extract token counts, calculate cost, and log errors.
- **Transparent FastAPI Proxy (`/v1/chat/completions`)**:
  - Point any OpenAI-compatible client at `http://localhost:8080/v1/chat/completions` to route requests to Groq or local Ollama while recording full telemetry into SQLite and returning `X-LLM-Obs-Call-Id` & `X-LLM-Obs-Trace-Id` headers.
- **OpenTelemetry GenAI Receiver (`/v1/traces`)**:
  - Accepts OTLP/HTTP JSON trace spans following `gen_ai.*` semantic conventions (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`).
- **Local SQLite WAL Storage (`SQLiteStore`)**:
  - Fast, single-file storage with indexed filtering by provider, model, status (`ok` / `error`), tag, trace ID, minimum latency, and full-text search across prompts and responses.
- **Interactive Dark-Mode Web Dashboard (`/`)**:
  - Real-time KPI cards (Total Calls, Total Tokens, Estimated Cost & Cloud Spend Saved, p50/p95/p99 Latency, Error Rate & Throughput), Hourly Call Volume chart, Model Breakdown table, filterable Call Log table, click-to-inspect prompt/response viewer, JSON export, and one-click **Seed Demo Data**.
- **Developer CLI (`llm-obs`)**:
  - `llm-obs serve`, `llm-obs seed`, `llm-obs stats`, and `llm-obs tail`.

---

## Architecture

```text
+-------------------------------------------------------------------------+
| Your AI App / Agents / Browser Playground                               |
|  - @observer.observe() / observer.trace_call() (Python SDK)             |
|  - GroqObservedClient / OllamaObservedClient                            |
|  - OpenAI SDK pointed at http://127.0.0.1:8080/v1/chat/completions      |
|  - OpenTelemetry exporter pointed at http://127.0.0.1:8080/v1/traces    |
|  - GitHub Pages Standalone Browser Engine (localStorage + Groq BYOK)    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| FastAPI Server & Collector (llm_obs.server)                             |
|  - POST /v1/chat/completions (Groq & Ollama proxy + latency timer)      |
|  - POST /api/events          (Direct SDK event ingest)                  |
|  - POST /v1/traces           (OTLP/HTTP JSON GenAI span parser)         |
|  - GET  /api/stats, /api/timeseries, /api/calls, /api/pricing           |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Local SQLite Store (WAL Mode) + Pricing Catalog                         |
|  - Exact p50 / p95 / p99 latency & token cost estimation                |
|  - Zero external services, 100% offline-capable                         |
+-------------------------------------------------------------------------+
```

---

## Quickstart (Local Python SDK & FastAPI Server)

### 1. Install

```bash
git clone https://github.com/taha080/llm-observability-dashboard.git
cd llm-observability-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure Environment (Optional for Live Groq Proxy)

Copy `.env.example` to `.env` (which is gitignored so your API key is never committed):

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_... if using live Groq forwarding
```

### 3. Seed Demo Data & Launch the Dashboard

```bash
# Seed 60 realistic Groq & local Ollama call logs and start the server on port 8080
llm-obs serve --seed --port 8080
```

Open **`http://127.0.0.1:8080`** in your browser to explore the local SQLite-backed dashboard, or visit **[https://taha080.github.io/llm-observability-dashboard/](https://taha080.github.io/llm-observability-dashboard/)** for the zero-install browser version.

---

## Usage Examples

### Python SDK: Context Manager & Decorator

```python
from llm_obs import LLMObserver

observer = LLMObserver(db_path="llm_observability.db", default_tags=["rag-app"])

# 1. Context manager
with observer.trace_call(
    provider="groq",
    model="openai/gpt-oss-20b",
    prompt_preview="Explain SQLite WAL mode in two sentences.",
    tags=["docs"],
) as span:
    # Call your model or pass an OpenAI/Groq/Ollama response dict
    fake_response = {
        "model": "openai/gpt-oss-20b",
        "usage": {"prompt_tokens": 28, "completion_tokens": 64, "total_tokens": 92},
        "choices": [{"message": {"content": "WAL appends changes to a write-ahead log..."}}],
    }
    span.set_response(fake_response)

# 2. Function decorator (works with both sync and async functions)
@observer.observe(provider="ollama", model="llama3.2", tags=["local-agent"])
def run_local_summary(prompt: str) -> dict:
    return {
        "model": "llama3.2",
        "prompt_eval_count": 45,
        "eval_count": 110,
        "message": {"content": "Summary of the document..."},
    }

run_local_summary("Summarize chapter 1")
```

### CLI Commands

```bash
# Seed 60 realistic calls (pass --reset to clear existing data first)
llm-obs seed --count 60 --reset

# Print terminal KPI summary and per-model breakdown
llm-obs stats

# Export stats as JSON
llm-obs stats --json

# Tail the 10 most recent LLM calls
llm-obs tail --limit 10
```

---

## Running Tests & Linting

All tests run **100% offline** using temporary SQLite databases and mocked HTTP transports:

```bash
ruff check src tests
ruff format --check src tests
pytest -v
```

---

## License

MIT License — Copyright (c) 2026 Taha Ahsan.
