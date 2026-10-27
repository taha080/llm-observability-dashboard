"""OpenTelemetry (OTLP/HTTP JSON) trace span parser for GenAI semantic conventions."""

from __future__ import annotations

from typing import Any

from llm_obs.models import CallEventCreate


def _extract_attr_value(value_obj: dict[str, Any]) -> Any:
    """Unwrap an OTLP attribute AnyValue dictionary."""
    if not isinstance(value_obj, dict):
        return value_obj
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value_obj:
            val = value_obj[key]
            if key == "intValue":
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return 0
            if key == "doubleValue":
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0
            return val
    if "arrayValue" in value_obj and isinstance(value_obj["arrayValue"], dict):
        values = value_obj["arrayValue"].get("values", [])
        return [_extract_attr_value(v) for v in values]
    return None


def _attributes_to_dict(attributes: list[dict[str, Any]] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not attributes:
        return result
    for item in attributes:
        if isinstance(item, dict) and "key" in item:
            result[str(item["key"])] = _extract_attr_value(item.get("value", {}))
    return result


def parse_otlp_traces(payload: dict[str, Any]) -> list[CallEventCreate]:
    """Convert an OTLP/HTTP JSON `/v1/traces` request body into CallEventCreate items."""
    events: list[CallEventCreate] = []
    resource_spans = payload.get("resourceSpans", [])
    if not isinstance(resource_spans, list):
        return events

    for res_span in resource_spans:
        if not isinstance(res_span, dict):
            continue
        res_attrs = _attributes_to_dict(res_span.get("resource", {}).get("attributes", []))
        service_name = str(res_attrs.get("service.name", "otel-service"))

        scope_spans = res_span.get("scopeSpans", [])
        if not isinstance(scope_spans, list):
            continue

        for scope_span in scope_spans:
            if not isinstance(scope_span, dict):
                continue
            spans = scope_span.get("spans", [])
            if not isinstance(spans, list):
                continue

            for span in spans:
                if not isinstance(span, dict):
                    continue
                attrs = _attributes_to_dict(span.get("attributes", []))
                provider = str(
                    attrs.get("gen_ai.system")
                    or attrs.get("llm.provider")
                    or res_attrs.get("gen_ai.system")
                    or "groq"
                ).lower()
                model = str(
                    attrs.get("gen_ai.request.model")
                    or attrs.get("gen_ai.response.model")
                    or attrs.get("llm.model")
                    or span.get("name")
                    or "unknown-model"
                )
                prompt_tokens = int(
                    attrs.get("gen_ai.usage.input_tokens")
                    or attrs.get("gen_ai.usage.prompt_tokens")
                    or attrs.get("llm.usage.prompt_tokens")
                    or 0
                )
                completion_tokens = int(
                    attrs.get("gen_ai.usage.output_tokens")
                    or attrs.get("gen_ai.usage.completion_tokens")
                    or attrs.get("llm.usage.completion_tokens")
                    or 0
                )

                start_ns = int(span.get("startTimeUnixNano") or 0)
                end_ns = int(span.get("endTimeUnixNano") or 0)
                latency_ms = (
                    round((end_ns - start_ns) / 1_000_000.0, 2)
                    if end_ns > start_ns > 0
                    else float(attrs.get("llm.latency_ms") or 0.0)
                )

                status_obj = span.get("status") or {}
                status_code = status_obj.get("code")
                is_err = status_code in (2, "STATUS_CODE_ERROR", "ERROR") or bool(
                    attrs.get("error.type")
                )
                error_type = str(attrs.get("error.type") or "OTelSpanError") if is_err else None
                error_message = (
                    str(status_obj.get("message") or attrs.get("error.message") or "Span error")
                    if is_err
                    else None
                )

                prompt_preview = attrs.get("gen_ai.prompt") or attrs.get("llm.prompt")
                response_preview = attrs.get("gen_ai.completion") or attrs.get("llm.completion")

                events.append(
                    CallEventCreate(
                        provider=provider,
                        model=model,
                        prompt_tokens=max(0, prompt_tokens),
                        completion_tokens=max(0, completion_tokens),
                        latency_ms=max(0.0, latency_ms),
                        status="error" if is_err else "ok",
                        error_type=error_type,
                        error_message=error_message,
                        prompt_preview=str(prompt_preview)[:4000] if prompt_preview else None,
                        response_preview=str(response_preview)[:4000] if response_preview else None,
                        trace_id=str(span.get("traceId") or "") or None,
                        span_id=str(span.get("spanId") or "") or None,
                        tags=["otel", service_name],
                        metadata={"span_name": span.get("name", ""), "service": service_name},
                    )
                )

    return events
