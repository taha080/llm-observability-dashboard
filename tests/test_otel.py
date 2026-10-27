"""Unit tests for OpenTelemetry OTLP/HTTP JSON trace span ingestion."""

from __future__ import annotations

from llm_obs.otel import parse_otlp_traces


def test_parse_otlp_traces_gen_ai_attributes() -> None:
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "rag-pipeline"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
                                "spanId": "00f067aa0ba902b7",
                                "name": "chat openai/gpt-oss-20b",
                                "startTimeUnixNano": "1700000000000000000",
                                "endTimeUnixNano": "1700000000245000000",
                                "status": {"code": 1},
                                "attributes": [
                                    {"key": "gen_ai.system", "value": {"stringValue": "groq"}},
                                    {
                                        "key": "gen_ai.request.model",
                                        "value": {"stringValue": "openai/gpt-oss-20b"},
                                    },
                                    {
                                        "key": "gen_ai.usage.input_tokens",
                                        "value": {"intValue": "140"},
                                    },
                                    {
                                        "key": "gen_ai.usage.output_tokens",
                                        "value": {"intValue": 310},
                                    },
                                    {
                                        "key": "gen_ai.prompt",
                                        "value": {"stringValue": "What is OTel?"},
                                    },
                                    {
                                        "key": "gen_ai.completion",
                                        "value": {"stringValue": "OTel framework."},
                                    },
                                ],
                            },
                            {
                                "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
                                "spanId": "11f067aa0ba902b8",
                                "name": "chat llama3.2",
                                "startTimeUnixNano": "1700000001000000000",
                                "endTimeUnixNano": "1700000001500000000",
                                "status": {"code": 2, "message": "Connection reset"},
                                "attributes": [
                                    {"key": "gen_ai.system", "value": {"stringValue": "ollama"}},
                                    {
                                        "key": "gen_ai.request.model",
                                        "value": {"stringValue": "llama3.2"},
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }

    events = parse_otlp_traces(payload)
    assert len(events) == 2

    first = events[0]
    assert first.provider == "groq"
    assert first.model == "openai/gpt-oss-20b"
    assert first.prompt_tokens == 140
    assert first.completion_tokens == 310
    assert first.latency_ms == 245.0
    assert first.status == "ok"
    assert "rag-pipeline" in first.tags

    second = events[1]
    assert second.provider == "ollama"
    assert second.status == "error"
    assert second.error_message == "Connection reset"
