"""Feedback builder for Harness."""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Feedback:
    """Structured feedback for retry loop."""

    source: str                    # evaluator, classifier, tool, etc.
    code: str                      # e.g., MISSING_TOOL_EVIDENCE, GROUNDEDNESS_FAILURE
    message: str                   # Human-readable message
    severity: str = "error"        # info, warning, error, critical
    actionable: bool = True        # Can agent act on this?
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "actionable": self.actionable,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Feedback":
        return cls(
            source=data["source"],
            code=data["code"],
            message=data["message"],
            severity=data.get("severity", "error"),
            actionable=data.get("actionable", True),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


class FeedbackBuilder:
    """Builds structured feedback from evaluation results."""

    # Error code mappings
    EVAL_ERROR_CODES = {
        "missing_evidence": "MISSING_TOOL_EVIDENCE",
        "incomplete_answer": "INCOMPLETE_ANSWER",
        "wrong_tool": "WRONG_TOOL_SELECTED",
        "tool_error": "TOOL_EXECUTION_FAILED",
        "groundedness_failure": "GROUNDING_FAILURE",
    }

    @classmethod
    def from_evaluation(cls, evaluation: dict, state=None) -> List[Feedback]:
        """Build feedback list from evaluation result."""
        feedback_list = []

        if evaluation.get("status") == "pass":
            return feedback_list

        # Extract error type and feedback
        error_type = evaluation.get("error_type", "evaluation_failure")
        feedback_messages = evaluation.get("feedback", [])

        for msg in feedback_messages:
            code = cls.EVAL_ERROR_CODES.get(error_type, "EVALUATION_FAILURE")
            feedback_list.append(Feedback(
                source="evaluator",
                code=code,
                message=msg,
                severity="error",
                actionable=True,
                metadata={"error_type": error_type, "attempt": state.attempt if state else 0}
            ))

        return feedback_list

    @classmethod
    def from_tool_error(cls, tool_name: str, tool_result: dict, attempt: int) -> List[Feedback]:
        """Build feedback from tool execution error."""
        feedback_list = []

        if not tool_result.get("success", True):
            error_type = tool_result.get("error_type", "tool_execution_failed")
            message = tool_result.get("message", "Tool execution failed")

            error_codes = {
                "timeout": "TOOL_TIMEOUT",
                "api_error": "TOOL_API_ERROR",
                "invalid_input": "TOOL_INVALID_INPUT",
                "city_not_found": "TOOL_CITY_NOT_FOUND",
                "data_not_found": "TOOL_DATA_NOT_FOUND",
                "invalid_dataset": "TOOL_INVALID_DATASET",
                "unknown_tool": "TOOL_UNKNOWN",
            }

            code = error_codes.get(error_type, "TOOL_EXECUTION_FAILED")

            feedback_list.append(Feedback(
                source="tool",
                code=code,
                message=f"{tool_name}: {message}",
                severity="error",
                actionable=error_type in ["timeout", "api_error", "invalid_input"],
                metadata={"tool": tool_name, "error_type": error_type, "attempt": attempt}
            ))

        return feedback_list

    @classmethod
    def from_classified_error(cls, classified_error) -> Feedback:
        """Build feedback from classified error."""
        return Feedback(
            source="classifier",
            code=classified_error.error_type.value.upper(),
            message=classified_error.message,
            severity=classified_error.severity.value if hasattr(classified_error.severity, 'value') else str(classified_error.severity),
            actionable=classified_error.retryable,
            metadata={
                "error_type": classified_error.error_type.value if hasattr(classified_error.error_type, 'value') else str(classified_error.error_type),
                "recoverable": classified_error.recoverable,
                "retryable": classified_error.retryable,
                "source": classified_error.source,
                **classified_error.metadata
            }
        )

    @classmethod
    def from_retry_exhaustion(cls, max_retries: int) -> Feedback:
        """Build feedback when max retries exceeded."""
        return Feedback(
            source="harness",
            code="MAX_RETRIES_EXCEEDED",
            message=f"Maximum retries ({max_retries}) exceeded, falling back",
            severity="critical",
            actionable=False,
            metadata={"max_retries": max_retries}
        )

    @classmethod
    def from_safety_violation(cls, violation_type: str, message: str, attempt: int) -> Feedback:
        """Build feedback from safety violation."""
        return Feedback(
            source="safety",
            code=f"SAFETY_{violation_type.upper()}",
            message=message,
            severity="critical",
            actionable=False,
            metadata={"violation_type": violation_type, "attempt": attempt}
        )