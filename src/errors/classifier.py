"""Error classifier for Harness."""

import logging
from typing import Optional, List

from .types import ErrorType, ErrorSeverity, RecoveryStrategy
from .models import ClassifiedError, ErrorContext

logger = logging.getLogger(__name__)


class ErrorClassifier:
    """Classifies errors and determines recovery strategy."""

    # Default recovery strategies by error type
    DEFAULT_STRATEGIES = {
        ErrorType.TOOL_TIMEOUT: RecoveryStrategy.RETRY_TOOL,
        ErrorType.TOOL_UNAVAILABLE: RecoveryStrategy.RETRY_TOOL,
        ErrorType.TOOL_INVALID_INPUT: RecoveryStrategy.RETOOL,
        ErrorType.TOOL_INVALID_OUTPUT: RecoveryStrategy.RETRY_TOOL,
        ErrorType.TOOL_EXECUTION_FAILED: RecoveryStrategy.RETRY_TOOL,
        ErrorType.LLM_TIMEOUT: RecoveryStrategy.RETRY_LLM,
        ErrorType.LLM_ERROR: RecoveryStrategy.RETRY_LLM,
        ErrorType.LLM_INVALID_RESPONSE: RecoveryStrategy.RETRY_LLM,
        ErrorType.LLM_RATE_LIMITED: RecoveryStrategy.RETRY_LLM,
        ErrorType.EVALUATION_FAILURE: RecoveryStrategy.RETRY_EVALUATION,
        ErrorType.GROUNDEDNESS_FAILURE: RecoveryStrategy.REGENERATE,
        ErrorType.TOOL_CORRECTNESS_FAILURE: RecoveryStrategy.RETOOL,
        ErrorType.COMPLETENESS_FAILURE: RecoveryStrategy.REGENERATE,
        ErrorType.POLICY_FAILURE: RecoveryStrategy.REPLAN,
        ErrorType.MAX_RETRIES_EXCEEDED: RecoveryStrategy.FALLBACK,
        ErrorType.RETRY_BUDGET_EXHAUSTED: RecoveryStrategy.FALLBACK,
        ErrorType.SAFETY_INPUT_VIOLATION: RecoveryStrategy.STOP,
        ErrorType.SAFETY_OUTPUT_VIOLATION: RecoveryStrategy.STOP,
        ErrorType.SAFETY_TOOL_VIOLATION: RecoveryStrategy.STOP,
        ErrorType.PROMPT_INJECTION: RecoveryStrategy.STOP,
        ErrorType.UNKNOWN: RecoveryStrategy.STOP,
    }

    # Recoverable error types
    RECOVERABLE_TYPES = {
        ErrorType.TOOL_TIMEOUT,
        ErrorType.TOOL_UNAVAILABLE,
        ErrorType.TOOL_INVALID_INPUT,
        ErrorType.TOOL_INVALID_OUTPUT,
        ErrorType.TOOL_EXECUTION_FAILED,
        ErrorType.LLM_TIMEOUT,
        ErrorType.LLM_ERROR,
        ErrorType.LLM_INVALID_RESPONSE,
        ErrorType.LLM_RATE_LIMITED,
        ErrorType.EVALUATION_FAILURE,
        ErrorType.GROUNDEDNESS_FAILURE,
        ErrorType.TOOL_CORRECTNESS_FAILURE,
        ErrorType.COMPLETENESS_FAILURE,
        ErrorType.POLICY_FAILURE,
    }

    # Retryable error types (subset of recoverable)
    RETRYABLE_TYPES = {
        ErrorType.TOOL_TIMEOUT,
        ErrorType.TOOL_UNAVAILABLE,
        ErrorType.TOOL_INVALID_OUTPUT,
        ErrorType.TOOL_EXECUTION_FAILED,
        ErrorType.LLM_TIMEOUT,
        ErrorType.LLM_ERROR,
        ErrorType.LLM_INVALID_RESPONSE,
        ErrorType.LLM_RATE_LIMITED,
        ErrorType.GROUNDEDNESS_FAILURE,
        ErrorType.COMPLETENESS_FAILURE,
    }

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def classify(self, context: ErrorContext) -> List[ClassifiedError]:
        """Classify errors based on context. Returns list of classified errors."""
        errors = []

        # Check for explicit exception
        if context.exception:
            errors.append(self._classify_exception(context.exception, context))

        # Check tool result errors
        if context.tool_result:
            errors.extend(self._classify_tool_result(context.tool_result, context))

        # Check evaluation result
        if context.evaluation_result:
            errors.extend(self._classify_evaluation(context.evaluation_result, context))

        # Check state-based errors (retry exhaustion, etc.)
        errors.extend(self._classify_state(context))

        return errors

    def _classify_exception(self, exc: Exception, context: ErrorContext) -> ClassifiedError:
        """Classify a raw exception."""
        exc_type = type(exc).__name__
        exc_msg = str(exc)

        if "timeout" in exc_msg.lower() or "Timeout" in exc_type:
            return ClassifiedError(
                error_type=ErrorType.TOOL_TIMEOUT if context.tool_name else ErrorType.LLM_TIMEOUT,
                message=f"Timeout: {exc_msg}",
                recoverable=True,
                retryable=True,
                source=context.tool_name or "llm",
                severity=ErrorSeverity.ERROR,
                suggested_strategy=RecoveryStrategy.RETRY_TOOL if context.tool_name else RecoveryStrategy.RETRY_LLM,
            )

        if "rate limit" in exc_msg.lower() or "429" in exc_msg:
            return ClassifiedError(
                error_type=ErrorType.LLM_RATE_LIMITED,
                message=f"Rate limited: {exc_msg}",
                recoverable=True,
                retryable=True,
                source="llm",
                severity=ErrorSeverity.WARNING,
                suggested_strategy=RecoveryStrategy.RETRY_LLM,
            )

        return ClassifiedError(
            error_type=ErrorType.UNKNOWN,
            message=f"{exc_type}: {exc_msg}",
            recoverable=False,
            retryable=False,
            source=context.tool_name or "unknown",
            severity=ErrorSeverity.CRITICAL,
            suggested_strategy=RecoveryStrategy.STOP,
        )

    def _classify_tool_result(self, tool_result: dict, context: ErrorContext) -> List[ClassifiedError]:
        """Classify tool execution result."""
        errors = []

        if not tool_result.get("success", True):
            error_type_str = tool_result.get("error_type", "unknown")
            error_msg = tool_result.get("message", "Tool execution failed")

            # Map tool error types to our error types
            error_type_map = {
                "timeout": ErrorType.TOOL_TIMEOUT,
                "api_error": ErrorType.TOOL_UNAVAILABLE,
                "invalid_input": ErrorType.TOOL_INVALID_INPUT,
                "city_not_found": ErrorType.TOOL_INVALID_INPUT,
                "data_not_found": ErrorType.TOOL_UNAVAILABLE,
                "invalid_dataset": ErrorType.TOOL_INVALID_OUTPUT,
                "unknown_tool": ErrorType.TOOL_EXECUTION_FAILED,
            }

            error_type = error_type_map.get(error_type_str, ErrorType.TOOL_EXECUTION_FAILED)

            errors.append(ClassifiedError(
                error_type=error_type,
                message=error_msg,
                recoverable=error_type in self.RECOVERABLE_TYPES,
                retryable=error_type in self.RETRYABLE_TYPES,
                source=context.tool_name or "tool",
                severity=ErrorSeverity.ERROR,
                suggested_strategy=self.DEFAULT_STRATEGIES.get(error_type, RecoveryStrategy.STOP),
            ))

        return errors

    def _classify_evaluation(self, evaluation_result: dict, context: ErrorContext) -> List[ClassifiedError]:
        """Classify evaluation failure."""
        errors = []

        if evaluation_result.get("status") == "fail":
            error_type_str = evaluation_result.get("error_type", "evaluation_failure")

            error_type_map = {
                "missing_evidence": ErrorType.GROUNDEDNESS_FAILURE,
                "incomplete_answer": ErrorType.COMPLETENESS_FAILURE,
                "wrong_tool": ErrorType.TOOL_CORRECTNESS_FAILURE,
                "tool_error": ErrorType.TOOL_EXECUTION_FAILED,
            }

            error_type = error_type_map.get(error_type_str, ErrorType.EVALUATION_FAILURE)

            feedback = evaluation_result.get("feedback", [])
            message = "; ".join(feedback) if feedback else "Evaluation failed"

            errors.append(ClassifiedError(
                error_type=error_type,
                message=message,
                recoverable=error_type in self.RECOVERABLE_TYPES,
                retryable=error_type in self.RETRYABLE_TYPES,
                source="evaluator",
                severity=ErrorSeverity.ERROR,
                suggested_strategy=self.DEFAULT_STRATEGIES.get(error_type, RecoveryStrategy.STOP),
            ))

        return errors

    def _classify_state(self, context: ErrorContext) -> List[ClassifiedError]:
        """Classify state-based errors (retry exhaustion, etc.)."""
        errors = []

        # Check retry exhaustion
        if context.attempt >= self.max_retries:
            errors.append(ClassifiedError(
                error_type=ErrorType.MAX_RETRIES_EXCEEDED,
                message=f"Maximum retries ({self.max_retries}) exceeded",
                recoverable=False,
                retryable=False,
                source="harness",
                severity=ErrorSeverity.CRITICAL,
                suggested_strategy=RecoveryStrategy.FALLBACK,
            ))

        return errors

    def is_recoverable(self, error_type: ErrorType) -> bool:
        """Check if error type is recoverable."""
        return error_type in self.RECOVERABLE_TYPES

    def is_retryable(self, error_type: ErrorType) -> bool:
        """Check if error type is retryable."""
        return error_type in self.RETRYABLE_TYPES

    def get_default_strategy(self, error_type: ErrorType) -> RecoveryStrategy:
        """Get default recovery strategy for error type."""
        return self.DEFAULT_STRATEGIES.get(error_type, RecoveryStrategy.STOP)