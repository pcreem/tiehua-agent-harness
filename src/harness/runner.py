"""Harness runner for executing agent queries."""

from typing import Optional, Callable, Literal
import time
import logging

from src.state import AgentState, initialize_state
from src.harness.loop import HarnessLoop
from src.harness.graph import HarnessGraphRunner
from src.harness.result import HarnessResult
from src.agent import create_agent
from src.observability import (
    get_metrics,
    get_tracer,
    harness_logger,
)
from src.observability.tracing import trace_agent_run

logger = logging.getLogger(__name__)


class HarnessRunner:
    """Runs the agent harness with full retry loop.

    Supports both legacy (HarnessLoop) and LangGraph-based execution.
    """

    def __init__(
        self,
        agent_factory: Optional[Callable] = None,
        max_retries: int = 2,
        evaluator_fn: Optional[Callable] = None,
        use_langgraph: bool = True,
    ):
        self.agent_factory = agent_factory or create_agent
        self.max_retries = max_retries
        self.evaluator_fn = evaluator_fn
        self.use_langgraph = use_langgraph
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
                if self.use_langgraph:
                    final_state = self._run_with_langgraph(agent, state)
                else:
                    final_state = self._run_with_loop(agent, state)

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

    def _run_with_langgraph(self, agent, state: AgentState) -> AgentState:
        """Execute using LangGraph-based harness."""
        graph_runner = HarnessGraphRunner(
            agent_fn=lambda s: agent.run_with_state(s),
            evaluator_fn=self.evaluator_fn,
            max_retries=self.max_retries,
            thread_id=state.run_id,
        )
        return graph_runner.run(state.query, max_retries=self.max_retries)

    def _run_with_loop(self, agent, state: AgentState) -> AgentState:
        """Execute using legacy HarnessLoop."""
        harness_loop = HarnessLoop(
            agent_fn=lambda s: agent.run_with_state(s),
            evaluator_fn=self.evaluator_fn,
            max_retries=self.max_retries,
        )
        return harness_loop.execute(state)

    def run_streaming(self, query: str, max_retries: Optional[int] = None):
        """Run harness with streaming results for real-time monitoring.

        Args:
            query: The user query to execute.
            max_retries: Override for max retries (defaults to instance setting).

        Yields:
            dict: State snapshot after each LangGraph node execution.
        """
        if not self.use_langgraph:
            raise NotImplementedError("Streaming only available with LangGraph")

        effective_max_retries = max_retries if max_retries is not None else self.max_retries

        agent = self.agent_factory()
        state = initialize_state(query, max_retries=effective_max_retries)

        graph_runner = HarnessGraphRunner(
            agent_fn=lambda s: agent.run_with_state(s),
            evaluator_fn=self.evaluator_fn,
            max_retries=effective_max_retries,
            thread_id=state.run_id,
        )

        for step in graph_runner.run_streaming(query, max_retries=effective_max_retries):
            yield step