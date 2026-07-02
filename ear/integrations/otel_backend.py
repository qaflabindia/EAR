"""OpenTelemetryExporter -- ship the ReasoningLog to any OTLP backend.

Langfuse and Phoenix both ingest OTLP natively, so this one exporter
covers the plan's Observability targets without binding to either SDK:
point the standard `OTEL_EXPORTER_OTLP_*` environment variables at the
platform's OTLP endpoint (for Langfuse, its documented OTLP path with the
basic-auth header built from the key env vars; for Phoenix, its collector
endpoint) and every trail record arrives as a span. Configuration is
environment-only -- never a key or endpoint written in a file.

The mapping: one trace-root span per cycle, one child span per
ReasoningRecord (name = stage), with the record's model, output, rationale
and inputs as attributes. Spans carry the record's own timestamp, so the
platform's timeline matches the markdown trail exactly -- the file on disk
remains the canonical record; this is an exporter, never a second
instrumentation path.

Requires `pip install 'ear[observability]'`. For tests or custom
pipelines, inject a configured `tracer_provider` (e.g. one backed by an
in-memory span exporter) instead of the default OTLP one.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class OpenTelemetryExporter:
    """Exports ReasoningRecords as OpenTelemetry spans, one cycle per
    trace."""

    def __init__(self, tracer_provider: Optional[Any] = None, service_name: str = "ear-runtime") -> None:
        from opentelemetry import trace

        if tracer_provider is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            tracer_provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        self._trace = trace
        self._provider = tracer_provider
        self._tracer = tracer_provider.get_tracer("ear")
        self._cycle: Optional[int] = None
        self._root: Optional[Any] = None

    def export(self, record: Any) -> None:
        nanoseconds = int(record.timestamp.timestamp() * 1_000_000_000)
        if record.cycle != self._cycle:
            self._end_root(nanoseconds)
            self._cycle = record.cycle
            self._root = self._tracer.start_span(f"cycle {record.cycle}", start_time=nanoseconds)
        context = self._trace.set_span_in_context(self._root) if self._root is not None else None
        span = self._tracer.start_span(record.stage, context=context, start_time=nanoseconds)
        span.set_attribute("ear.cycle", record.cycle)
        span.set_attribute("ear.stage", record.stage)
        if record.model:
            span.set_attribute("ear.model", record.model)
        span.set_attribute("ear.output", record.output)
        if record.rationale:
            span.set_attribute("ear.rationale", record.rationale)
        span.set_attribute("ear.inputs", json.dumps(record.inputs, default=str))
        span.end(end_time=nanoseconds)

    def flush(self) -> None:
        """Close the open cycle trace and push batched spans out. The
        ReasoningLog calls this after each fan-out, which is each cycle's
        end -- so cycle traces close when cycles do."""
        self._end_root(None)
        force = getattr(self._provider, "force_flush", None)
        if callable(force):
            force()

    def _end_root(self, end_time: Optional[int]) -> None:
        if self._root is not None:
            if end_time is not None:
                self._root.end(end_time=end_time)
            else:
                self._root.end()
            self._root = None