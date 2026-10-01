"""Harness loop for executing agent with retry logic."""

from typing import Optional, Callable
from src.errors.models import ClassifiedError
from src.errors.classifier import ErrorClassifier
from src.errors.types import ErrorType
from src.state import AgentState
from src.harness.feedback import FeedbackBuilder
from src.harness.retry import RetryPolicy, RetryDecision
from src.evaluator import evaluate
import logging

logger = logging.getLogger(__name__)


class HarnessLoop:
    """Executes the agent with retry logic and evaluation."""

    def __init__(
        self,
        agent_fn: Callable[[AgentState], AgentState],
        evaluator_fn: Optional[Callable[[AgentState], dict]] = None,
        error_classifier: Optional[ErrorClassifier] = None,
        retry_policy: Optional[RetryPolicy] = None,
        max_retries: int = 2,
    ):
        self.agent_fn = agent_fn
        self.evaluator_fn = evaluator_fn or evaluate
        self.error_classifier = error_classifier or ErrorClassifier(max_retries=max_retries)
        self.retry_policy = retry_policy or RetryPolicy(max_retries=max_retries)
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def execute(self, state: AgentState) -> AgentState:
        """Execute the agent with retry loop."""
        state.set_status("running")

        while True:
            self.logger.info(f"Harness loop: attempt {state.attempt + 1}/{self.max_retries + 1}")

            # Execute agent
            try:
                state = self.agent_fn(state)
            except Exception as e:
                self.logger.exception("Agent execution failed")
                state.set_error({
                    "type": "agent_execution_failed",
                    "message": str(e),
                })
                # Create error context and classify
                context = self._build_error_context(state, exception=e)
                errors = self.error_classifier.classify(context)
                decision = self.retry_policy.decide(state, errors)

                if decision == RetryDecision.RETRY and state.attempt < self.max_retries:
                    state.increment_attempt()
                    self._apply_feedback(state, errors)
                    continue
                else:
                    state.set_status("failed" if decision == RetryDecision.STOP else "fallback")
                    break

            # Evaluate result
            try:
                evaluation = self.evaluator_fn(state.query, state)
                state.set_evaluation(evaluation)
            except Exception as e:
                self.logger.exception("Evaluation failed")
                evaluation = {
                    "status": "fail",
                    "error_type": "evaluation_failure",
                    "feedback": [f"Evaluation failed: {str(e)}"],
                    "retry": True,
                }
                state.set_evaluation(evaluation)

            # Check evaluation result
            if evaluation.get("status") == "pass":
                state.set_status("passed")
                self.logger.info("Evaluation passed")
                break

            # Evaluation failed - classify errors and decide
            self.logger.warning(f"Evaluation failed: {evaluation.get('error_type')}")

            # Build error context
            context = self._build_error_context(state)
            context.evaluation_result = evaluation
            errors = self.error_classifier.classify(context)

            # Build feedback
            feedback_list = FeedbackBuilder.from_evaluation(evaluation, state)
            for fb in feedback_list:
                state.add_feedback(fb.to_dict() if hasattr(fb, 'to_dict') else fb)

            # Also add classified errors as feedback
            for error in errors:
                fb = FeedbackBuilder.from_classified_error(error)
                state.add_feedback(fb.to_dict() if hasattr(fb, 'to_dict') else fb)

            # Decide retry
            decision = self.retry_policy.decide(state, errors)

            if decision == RetryDecision.RETRY:
                if state.attempt >= self.max_retries:
                    self.logger.warning("Max retries reached, falling back")
                    state.set_status("fallback")
                    state.add_feedback(FeedbackBuilder.from_retry_exhaustion(self.max_retries).to_dict())
                    break

                self.logger.info(f"Retrying... (attempt {state.attempt + 1})")
                state.increment_attempt()
                state.set_status("running")
                continue

            elif decision == RetryDecision.FALLBACK:
                self.logger.warning("Fallback triggered")
                state.set_status("fallback")
                state.add_feedback(FeedbackBuilder.from_retry_exhaustion(self.max_retries).to_dict())
                break

            else:  # STOP
                self.logger.warning("Stop decision")
                state.set_status("failed")
                break

        return state

    def _apply_feedback(self, state: AgentState, errors: List[ClassifiedError]):
        """Apply feedback from errors to state for retry."""
        for error in errors:
            fb = FeedbackBuilder.from_classified_error(error)
            state.add_feedback(fb.to_dict() if hasattr(fb, 'to_dict') else fb)

    def _build_error_context(self, state: AgentState, exception: Optional[Exception] = None):
        """Build error context from state."""
        from src.errors.models import ErrorContext

        # Get latest tool result
        tool_result = None
        tool_name = None
        if state.tool_results:
            tool_result = state.tool_results[-1].get("result", {})
            tool_name = state.tool_results[-1].get("name")

        return ErrorContext(
            state=state,
            tool_name=tool_name,
            tool_result=tool_result,
            evaluation_result=state.evaluation,
            attempt=state.attempt,
            max_retries=state.max_retries,
            exception=exception,
        )


# Import here to avoid circular import
from src.harness.feedback import FeedbackBuilder