"""Harness runner for executing agent queries."""

from typing import Optional, Callable
import time

from src.state import AgentState, initialize_state
from src.harness.loop import HarnessLoop
from src.harness.result import HarnessResult
from src.agent import create_agent
from src.observability import (
    get_logger,
    get_metrics,
    get_tracer,
    trace_retry,
    harness_logger,
)

logger = logging.getLogger(__name__)


class HarnessRunner:
    """Runs the agent harness with full retry loop."""

    def __init__(
        self,
        agent_factory: Optional[Callable] = None,
        max_retries: int = 2,
        evaluator_fn: Optional[Callable] = None,
    ):
        self.agent_factory = agent_factory or create_agent
        self.max_retries = max_retries
        self.evaluator_fn = evaluator_fn
        self.logger = logging.getLogger(__name__)

    def run(self, query: str) -> HarnessResult:
        """Run the harness with a user query."""
        # Create agent
        agent = self.agent_factory()

        # Initialize state
        state = initialize_state(query, max_retries=self.max_retries)

        # Observability
        metrics = get_metrics()
        tracer = get_tracer()
        start_time = time.time()

        with trace_agent_run(query, state.run_id) as span:
            try:
                # Create harness loop - use run_with_state which accepts AgentState
                harness_loop = HarnessLoop(
                    agent_fn=lambda s: agent.run_with_state(s),
                    evaluator_fn=self.evaluator_fn,
                    max_retries=self.max_retries,
                )

                # Execute with retry loop
                final_state = harness_loop.execute(state)

                duration = time.time() - start_time

                # Build result
                result = HarnessResult(
                    query=final_state.query,
                    final_answer=final_state.answer or "",
                    state=final_state,
                    success=final_state.status == "passed",
                    status=final_state.status,
                    retry_count=final_state.attempt,
                    tool_calls=final_state.tool_calls,
                    tool_results=final_state.tool_results,
                    evaluation=final_state.evaluation,
                    feedback=final_state.feedback,
                )

                # Record metrics
                metrics.record_agent_run(result.status, duration)

                harness_logger.info("harness_completed",
                    query=query[:100],
                    status=result.status,
                    retries=result.retry_count,
                    duration=duration
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                metrics.record_error(type(e).__name__, "harness")
                metrics.record_agent_run("failed", duration)
                harness_logger.error("harness_failed", query=query[:100], error=str(e), duration=duration)
                raise

    def run_streaming(self, query: str):
        """Run harness with streaming results (not yet implemented)."""
        raise NotImplementedError("Streaming not yet implemented")