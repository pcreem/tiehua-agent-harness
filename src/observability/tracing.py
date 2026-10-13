"""OpenTelemetry tracing for Agent Harness."""

import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Global tracer provider
_tracer_provider: Optional[TracerProvider] = None


def init_tracing(
    service_name: str = "taiwan-tourism-agent-harness",
    otlp_endpoint: Optional[str] = None,
    enable_console: bool = False,
    auto_instrument: bool = True,
) -> TracerProvider:
    """Initialize OpenTelemetry tracing."""
    global _tracer_provider

    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Configure exporters
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if enable_console:
        console_exporter = ConsoleSpanExporter()
        _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Auto-instrument common libraries
    if auto_instrument:
        RequestsInstrumentor().instrument()
        LoggingInstrumentor().instrument()

    return _tracer_provider


def get_tracer(name: str = "taiwan-tourism-agent-harness") -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)


def get_current_span() -> trace.Span:
    """Get current active span."""
    return trace.get_current_span()


@contextmanager
def trace_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    tracer: Optional[trace.Tracer] = None,
):
    """Context manager for creating a span."""
    tracer = tracer or get_tracer()
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def trace_function(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator for tracing a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with trace_span(span_name, kind=kind, attributes=attributes) as span:
                # Add function arguments as attributes (be careful with sensitive data)
                span.set_attribute("function.name", func.__name__)
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Common span creation helpers
def trace_agent_run(query: str, run_id: str) -> trace.Span:
    """Create a span for agent run."""
    return trace_span(
        "agent.run",
        kind=SpanKind.SERVER,
        attributes={
            "agent.query": query[:200],  # Truncate long queries
            "agent.run_id": run_id,
        },
    )


def trace_llm_call(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    tool_calls: int = 0,
) -> trace.Span:
    """Create a span for LLM call."""
    return trace_span(
        "llm.call",
        kind=SpanKind.CLIENT,
        attributes={
            "llm.model": model,
            "llm.prompt_tokens": prompt_tokens,
            "llm.completion_tokens": completion_tokens,
            "llm.tool_calls": tool_calls,
        },
    )


def trace_tool_call(tool: str, args: Dict[str, Any]) -> trace.Span:
    """Create a span for tool execution."""
    return trace_span(
        "tool.call",
        kind=SpanKind.CLIENT,
        attributes={
            "tool.name": tool,
            "tool.args": str(args)[:500],  # Truncate long args
        },
    )


def trace_evaluation(query: str, status: str, score: float) -> trace.Span:
    """Create a span for evaluation."""
    return trace_span(
        "evaluation.run",
        kind=SpanKind.INTERNAL,
        attributes={
            "evaluation.query": query[:200],
            "evaluation.status": status,
            "evaluation.score": score,
        },
    )


def trace_retry(attempt: int, strategy: str, run_id: str) -> trace.Span:
    """Create a span for retry."""
    return trace_span(
        "harness.retry",
        kind=SpanKind.INTERNAL,
        attributes={
            "retry.attempt": attempt,
            "retry.strategy": strategy,
            "retry.run_id": run_id,
        },
    )


def set_span_attributes(span: trace.Span, attributes: Dict[str, Any]) -> None:
    """Set multiple attributes on a span."""
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def add_event(span: trace.Span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add an event to the current span."""
    span.add_event(name, attributes or {})


# Initialize tracing on import (can be called explicitly)
def ensure_tracing_initialized() -> TracerProvider:
    """Ensure tracing is initialized, return provider."""
    global _tracer_provider
    if _tracer_provider is None:
        return init_tracing()
    return _tracer_provider


# Cleanup
def shutdown_tracing() -> None:
    """Shutdown tracer provider."""
    global _tracer_provider
    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None