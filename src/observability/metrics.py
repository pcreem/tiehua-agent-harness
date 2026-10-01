"""Prometheus metrics for Agent Harness."""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from typing import Optional
import time


class Metrics:
    """Centralized metrics registry for Agent Harness."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Agent execution metrics
        self.agent_runs_total = Counter(
            "agent_runs_total",
            "Total number of agent runs",
            ["status"],  # passed, failed, fallback
            registry=self.registry,
        )

        self.agent_run_duration = Histogram(
            "agent_run_duration_seconds",
            "Agent run duration in seconds",
            ["status"],
            buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
            registry=self.registry,
        )

        # Tool execution metrics
        self.tool_calls_total = Counter(
            "tool_calls_total",
            "Total number of tool calls",
            ["tool", "status"],  # tool name, success/failed
            registry=self.registry,
        )

        self.tool_duration = Histogram(
            "tool_duration_seconds",
            "Tool execution duration in seconds",
            ["tool"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
            registry=self.registry,
        )

        # LLM metrics
        self.llm_calls_total = Counter(
            "llm_calls_total",
            "Total number of LLM calls",
            ["model", "status"],  # model name, success/failed/timeout
            registry=self.registry,
        )

        self.llm_duration = Histogram(
            "llm_duration_seconds",
            "LLM call duration in seconds",
            ["model"],
            buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
            registry=self.registry,
        )

        self.llm_tokens = Counter(
            "llm_tokens_total",
            "Total LLM tokens used",
            ["model", "type"],  # prompt, completion, total
            registry=self.registry,
        )

        self.llm_token_usage = Histogram(
            "llm_token_usage",
            "LLM tokens per call",
            ["model", "type"],
            buckets=[100, 500, 1000, 2000, 5000, 10000, 20000],
            registry=self.registry,
        )

        # Retry metrics
        self.retry_attempts = Counter(
            "retry_attempts_total",
            "Total number of retry attempts",
            ["strategy"],  # replan, retool, retrieve_again, regenerate, retry_tool
            registry=self.registry,
        )

        self.max_retries_exceeded = Counter(
            "max_retries_exceeded_total",
            "Number of times max retries exceeded",
            registry=self.registry,
        )

        # Evaluation metrics
        self.evaluations_total = Counter(
            "evaluations_total",
            "Total number of evaluations",
            ["result"],  # pass, fail
            registry=self.registry,
        )

        self.evaluation_score = Histogram(
            "evaluation_score",
            "Evaluation score distribution",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "errors_total",
            "Total errors by type",
            ["error_type", "source"],  # error type, agent/tool/llm/evaluator
            registry=self.registry,
        )

        # State metrics (gauges)
        self.active_runs = Gauge(
            "active_runs",
            "Currently active agent runs",
            registry=self.registry,
        )

        self.queue_depth = Gauge(
            "harness_queue_depth",
            "Number of pending evaluations in queue",
            registry=self.registry,
        )

    def record_agent_run(self, status: str, duration: float) -> None:
        """Record agent run completion."""
        self.agent_runs_total.labels(status=status).inc()
        self.agent_run_duration.labels(status=status).observe(duration)

    def record_tool_call(self, tool: str, success: bool, duration: float) -> None:
        """Record tool call."""
        status = "success" if success else "failed"
        self.tool_calls_total.labels(tool=tool, status=status).inc()
        self.tool_duration.labels(tool=tool).observe(duration)

    def record_llm_call(
        self,
        model: str,
        success: bool,
        duration: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record LLM call."""
        status = "success" if success else "failed"
        self.llm_calls_total.labels(model=model, status=status).inc()
        self.llm_duration.labels(model=model).observe(duration)

        if prompt_tokens:
            self.llm_tokens.labels(model=model, type="prompt").inc(prompt_tokens)
            self.llm_token_usage.labels(model=model, type="prompt").observe(prompt_tokens)

        if completion_tokens:
            self.llm_tokens.labels(model=model, type="completion").inc(completion_tokens)
            self.llm_token_usage.labels(model=model, type="completion").observe(completion_tokens)

        if prompt_tokens or completion_tokens:
            total = prompt_tokens + completion_tokens
            self.llm_tokens.labels(model=model, type="total").inc(total)
            self.llm_token_usage.labels(model=model, type="total").observe(total)

    def record_retry(self, strategy: str) -> None:
        """Record retry attempt."""
        self.retry_attempts.labels(strategy=strategy).inc()

    def record_max_retries_exceeded(self) -> None:
        """Record max retries exceeded."""
        self.max_retries_exceeded.inc()

    def record_evaluation(self, passed: bool, score: float) -> None:
        """Record evaluation result."""
        result = "pass" if passed else "fail"
        self.evaluations_total.labels(result=result).inc()
        self.evaluation_score.observe(score)

    def record_error(self, error_type: str, source: str) -> None:
        """Record error."""
        self.errors_total.labels(error_type=error_type, source=source).inc()

    def set_active_runs(self, count: int) -> None:
        """Set active runs gauge."""
        self.active_runs.set(count)

    def set_queue_depth(self, count: int) -> None:
        """Set queue depth gauge."""
        self.queue_depth.set(count)

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)


# Global metrics instance
_metrics: Optional["Metrics"] = None


def get_metrics() -> "Metrics":
    """Get global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


def init_metrics(registry: Optional[CollectorRegistry] = None) -> "Metrics":
    """Initialize global metrics."""
    global _metrics
    _metrics = Metrics(registry)
    return _metrics