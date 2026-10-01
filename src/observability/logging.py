"""Structured logging for Agent Harness."""

import os
import sys
import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime
from contextvars import ContextVar

import structlog
from structlog.types import EventDict, WrappedLogger


# Context variables for request-scoped logging
run_id_var: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
attempt_var: ContextVar[int] = ContextVar("attempt", default=0)
tool_name_var: ContextVar[Optional[str]] = ContextVar("tool_name", default=None)


def add_run_context(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add run context to log entries."""
    run_id = run_id_var.get()
    attempt = attempt_var.get()
    tool_name = tool_name_var.get()

    if run_id:
        event_dict["run_id"] = run_id
    if attempt > 0:
        event_dict["attempt"] = attempt
    if tool_name:
        event_dict["tool"] = tool_name

    return event_dict


def add_timestamp(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to log entries."""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_level(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Ensure level is present."""
    if "level" not in event_dict:
        event_dict["level"] = method_name.upper()
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    log_file: Optional[str] = None,
) -> structlog.BoundLogger:
    """Configure structured logging for the application."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_run_context,
        add_timestamp,
        add_level,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def set_run_context(run_id: str, attempt: int = 0) -> None:
    """Set run context for current execution."""
    run_id_var.set(run_id)
    attempt_var.set(attempt)


def set_tool_context(tool_name: str) -> None:
    """Set tool context for current tool execution."""
    tool_name_var.set(tool_name)


def clear_context() -> None:
    """Clear all context variables."""
    run_id_var.set(None)
    attempt_var.set(0)
    tool_name_var.set(None)


# Pre-configured loggers for common use cases
agent_logger = get_logger("agent")
tool_logger = get_logger("tool")
evaluator_logger = get_logger("evaluator")
harness_logger = get_logger("harness")
llm_logger = get_logger("llm")