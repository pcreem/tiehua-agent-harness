"""LangGraph-based Harness for Agent Execution.

Replaces manual while-loop with LangGraph state graph for:
- Better observability (graph visualization, step-by-step tracing)
- Built-in checkpointing (state persistence across retries)
- Conditional edges (evaluation -> retry/stop decisions)
- Human-in-the-loop support (safety review nodes)
- Industry-standard framework (addresses job requirement)
"""

from typing import Optional, Callable, Literal
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.state import AgentState
from src.errors.classifier import ErrorClassifier
from src.errors.models import ClassifiedError, ErrorContext
from src.errors.types import ErrorType
from src.harness.retry import RetryPolicy, RetryDecision
from src.harness.feedback import FeedbackBuilder
from src.harness.graph_types import HarnessGraphState
from src.harness.language_detector import create_language_detector_node
from src.harness.intent_classifier import create_intent_classifier_node
from src.evaluator import evaluate

logger = logging.getLogger(__name__)


def create_harness_graph(
    agent_fn: Callable[[AgentState], AgentState],
    evaluator_fn: Optional[Callable] = None,
    error_classifier: Optional[ErrorClassifier] = None,
    retry_policy: Optional[RetryPolicy] = None,
    max_retries: int = 2,
) -> StateGraph:
    """Create the LangGraph state graph for harness execution."""

    evaluator_fn = evaluator_fn or evaluate
    error_classifier = error_classifier or ErrorClassifier(max_retries=max_retries)
    retry_policy = retry_policy or RetryPolicy(max_retries=max_retries)

    workflow = StateGraph(HarnessGraphState)

    # Multilingual & Intent nodes (run once at start)
    workflow.add_node("language_detection", create_language_detector_node())
    workflow.add_node("intent_classification", create_intent_classifier_node())

    # Core execution nodes
    workflow.add_node("agent_execution", _create_agent_node(agent_fn))
    workflow.add_node("evaluation", _create_evaluation_node(evaluator_fn))
    workflow.add_node("error_classification", _create_classification_node(error_classifier))
    workflow.add_node("feedback_builder", _create_feedback_node())
    workflow.add_node("retry_decision", _create_retry_decision_node(retry_policy))
    workflow.add_node("finalize", _create_finalize_node())

    workflow.set_entry_point("language_detection")

    # Linear flow: language -> intent -> agent -> evaluation -> classification -> feedback -> retry_decision
    workflow.add_edge("language_detection", "intent_classification")
    workflow.add_edge("intent_classification", "agent_execution")
    workflow.add_edge("agent_execution", "evaluation")
    workflow.add_edge("evaluation", "error_classification")
    workflow.add_edge("error_classification", "feedback_builder")
    workflow.add_edge("feedback_builder", "retry_decision")

    workflow.add_conditional_edges(
        "retry_decision",
        _route_retry_decision,
        {
            "retry": "agent_execution",  # Retry goes back to agent execution (skip language/intent)
            "stop": "finalize",
            "fallback": "finalize",
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=MemorySaver())


def _create_agent_node(agent_fn: Callable[[AgentState], AgentState]):
    """Node: Execute agent with tool calling."""
    def node(state: HarnessGraphState) -> HarnessGraphState:
        agent_state = state["agent_state"]
        agent_state.set_status("running")
        logger.info(f"Agent execution: attempt {agent_state.attempt + 1}")

        try:
            agent_state = agent_fn(agent_state)
        except Exception as e:
            logger.exception("Agent execution failed")
            agent_state.set_error({"type": "agent_execution_failed", "message": str(e)})

        new_state = state.copy()
        new_state["agent_state"] = agent_state
        new_state["current_node"] = "agent_execution"
        new_state["execution_path"] = state.get("execution_path", []) + ["agent_execution"]
        return new_state
    return node


def _create_evaluation_node(evaluator_fn: Callable):
    """Node: Run deterministic evaluator + LLM Judge."""
    def node(state: HarnessGraphState) -> HarnessGraphState:
        agent_state = state["agent_state"]
        logger.info("Running evaluation")

        try:
            evaluation = evaluator_fn(agent_state.query, agent_state)
            agent_state.set_evaluation(evaluation)
        except Exception as e:
            logger.exception("Evaluation failed")
            evaluation = {
                "status": "fail",
                "error_type": "evaluation_failure",
                "feedback": [f"Evaluation failed: {str(e)}"],
                "retry": True,
            }
            agent_state.set_evaluation(evaluation)

        new_state = state.copy()
        new_state["agent_state"] = agent_state
        new_state["evaluation_result"] = evaluation
        new_state["current_node"] = "evaluation"
        new_state["execution_path"] = state.get("execution_path", []) + ["evaluation"]
        return new_state
    return node


def _create_classification_node(error_classifier: ErrorClassifier):
    """Node: Classify errors from evaluation and tool results."""
    def node(state: HarnessGraphState) -> HarnessGraphState:
        agent_state = state["agent_state"]
        evaluation = state["evaluation_result"]

        context = _build_error_context(agent_state, evaluation)
        errors = error_classifier.classify(context)

        logger.warning(f"Classified {len(errors)} errors: {[e.error_type.value for e in errors]}")

        new_state = state.copy()
        new_state["agent_state"] = agent_state
        new_state["errors"] = errors
        new_state["current_node"] = "error_classification"
        new_state["execution_path"] = state.get("execution_path", []) + ["error_classification"]
        return new_state
    return node


def _create_feedback_node():
    """Node: Build structured feedback for retry."""
    def node(state: HarnessGraphState) -> HarnessGraphState:
        agent_state = state["agent_state"]
        evaluation = state["evaluation_result"]
        errors = state["errors"]

        feedback_list = []

        if evaluation and evaluation.get("status") != "pass":
            feedback_list.extend(FeedbackBuilder.from_evaluation(evaluation, agent_state))

        for error in errors:
            feedback_list.append(FeedbackBuilder.from_classified_error(error))

        for fb in feedback_list:
            agent_state.add_feedback(fb.to_dict())

        logger.info(f"Added {len(feedback_list)} feedback items")

        new_state = state.copy()
        new_state["agent_state"] = agent_state
        new_state["current_node"] = "feedback_builder"
        new_state["execution_path"] = state.get("execution_path", []) + ["feedback_builder"]
        return new_state
    return node


def _create_retry_decision_node(retry_policy: RetryPolicy):
    """Node: Decide retry/stop/fallback based on errors and policy."""
    def node(state: HarnessGraphState) -> HarnessGraphState:
        agent_state = state["agent_state"]
        errors = state["errors"]

        if evaluation := state.get("evaluation_result"):
            if evaluation.get("status") == "pass":
                agent_state.set_status("passed")
                decision = RetryDecision.STOP
                logger.info("Evaluation passed - stopping")
                return _update_state_with_decision(state, decision)

        decision = retry_policy.decide(agent_state, errors)

        if decision == RetryDecision.RETRY:
            if agent_state.attempt >= agent_state.max_retries:
                decision = RetryDecision.FALLBACK
                logger.warning("Max retries reached - fallback")
                agent_state.add_feedback(
                    FeedbackBuilder.from_retry_exhaustion(agent_state.max_retries).to_dict()
                )
            else:
                logger.info(f"Retrying... (attempt {agent_state.attempt + 1})")
                agent_state.increment_attempt()
                agent_state.set_status("running")

        elif decision == RetryDecision.FALLBACK:
            logger.warning("Fallback triggered")
            agent_state.set_status("fallback")
            agent_state.add_feedback(
                FeedbackBuilder.from_retry_exhaustion(agent_state.max_retries).to_dict()
            )

        else:
            logger.warning("Stop decision")
            agent_state.set_status("failed")

        return _update_state_with_decision(state, decision)
    return node


def _create_finalize_node():
    """Node: Finalize execution."""
    def node(state: HarnessGraphState) -> HarnessGraphState:
        agent_state = state["agent_state"]
        logger.info(f"Harness finalized with status: {agent_state.status}")
        new_state = state.copy()
        new_state["agent_state"] = agent_state
        new_state["current_node"] = "finalize"
        new_state["execution_path"] = state.get("execution_path", []) + ["finalize"]
        return new_state
    return node


def _route_retry_decision(state: HarnessGraphState) -> Literal["retry", "stop", "fallback"]:
    """Conditional edge: route based on retry decision."""
    decision = state.get("retry_decision", RetryDecision.STOP)
    return decision.value


def _update_state_with_decision(state: HarnessGraphState, decision: RetryDecision) -> HarnessGraphState:
    new_state = state.copy()
    new_state["retry_decision"] = decision
    new_state["current_node"] = "retry_decision"
    new_state["execution_path"] = state.get("execution_path", []) + ["retry_decision"]
    return new_state


def _build_error_context(agent_state: AgentState, evaluation: Optional[dict]) -> ErrorContext:
    """Build error context from agent state."""
    tool_result = None
    tool_name = None
    if agent_state.tool_results:
        tool_result = agent_state.tool_results[-1].get("result", {})
        tool_name = agent_state.tool_results[-1].get("name")

    return ErrorContext(
        state=agent_state,
        tool_name=tool_name,
        tool_result=tool_result,
        evaluation_result=evaluation,
        attempt=agent_state.attempt,
        max_retries=agent_state.max_retries,
        exception=None,
    )


class HarnessGraphRunner:
    """Runner for LangGraph-based harness with checkpointing."""

    def __init__(
        self,
        agent_fn: Callable[[AgentState], AgentState],
        evaluator_fn: Optional[Callable] = None,
        max_retries: int = 2,
        thread_id: Optional[str] = None,
    ):
        self.graph = create_harness_graph(
            agent_fn=agent_fn,
            evaluator_fn=evaluator_fn,
            max_retries=max_retries,
        )
        self.thread_id = thread_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run(self, query: str, config: Optional[dict] = None) -> AgentState:
        """Execute the harness graph with a query."""
        from src.state import initialize_state

        initial_state = initialize_state(query, max_retries=2)

        graph_config = {"configurable": {"thread_id": self.thread_id}}
        if config:
            graph_config.update(config)

        result = self.graph.invoke(
            {
                "agent_state": initial_state,
                "evaluation_result": None,
                "errors": [],
                "retry_decision": None,
                "current_node": "start",
                "execution_path": [],
            },
            config=graph_config,
        )

        return result["agent_state"]

    def run_streaming(self, query: str):
        """Stream execution steps for real-time monitoring."""
        from src.state import initialize_state

        initial_state = initialize_state(query, max_retries=2)
        graph_config = {"configurable": {"thread_id": self.thread_id}}

        for step in self.graph.stream(
            {
                "agent_state": initial_state,
                "evaluation_result": None,
                "errors": [],
                "retry_decision": None,
                "current_node": "start",
                "execution_path": [],
            },
            config=graph_config,
            stream_mode="values",
        ):
            yield step

    def get_state_history(self) -> list[dict]:
        """Get execution history from checkpoints."""
        history = []
        try:
            for checkpoint in self.graph.get_state_history(
                {"configurable": {"thread_id": self.thread_id}}
            ):
                history.append({
                    "step": checkpoint.metadata.get("step", 0),
                    "node": checkpoint.values.get("current_node"),
                    "status": checkpoint.values.get("agent_state", {}).get("status"),
                    "attempt": checkpoint.values.get("agent_state", {}).get("attempt"),
                    "timestamp": checkpoint.metadata.get("timestamp"),
                })
        except Exception:
            pass
        return history

    def visualize(self) -> str:
        """Generate Mermaid diagram of the graph."""
        return self.graph.draw_mermaid()


def create_harness_graph_runner(
    agent_fn: Callable[[AgentState], AgentState],
    evaluator_fn: Optional[Callable] = None,
    max_retries: int = 2,
) -> HarnessGraphRunner:
    """Factory function to create harness graph runner."""
    return HarnessGraphRunner(
        agent_fn=agent_fn,
        evaluator_fn=evaluator_fn,
        max_retries=max_retries,
    )