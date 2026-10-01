"""Observability module for Agent Harness."""

from .logging import (
    setup_logging,
    get_logger,
    set_run_context,
    set_tool_context,
    clear_context,
    agent_logger,
    tool_logger,
    evaluator_logger,
    harness_logger,
    llm_logger,
)

from .metrics import (
    Metrics,
    get_metrics,
    init_metrics,
)

from .tracing import (
    init_tracing,
    get_tracer,
    get_current_span,
    trace_span,
    trace_function,
    trace_agent_run,
    trace_llm_call,
    trace_tool_call,
    trace_evaluation,
    trace_retry,
    ensure_tracing_initialized,
    shutdown_tracing,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "set_run_context",
    "set_tool_context",
    "clear_context",
    "agent_logger",
    "tool_logger",
    "evaluator_logger",
    "harness_logger",
    "llm_logger",
    # Metrics
    "Metrics",
    "get_metrics",
    "init_metrics",
    # Tracing
    "init_tracing",
    "get_tracer",
    "get_current_span",
    "trace_span",
    "trace_function",
    "trace_agent_run",
    "trace_llm_call",
    "trace_tool_call",
    "trace_evaluation",
    "trace_retry",
    "ensure_tracing_initialized",
    "shutdown_tracing",
]