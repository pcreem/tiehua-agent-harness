"""Error models for Harness."""

from dataclasses import dataclass, field
from typing import Any, Optional
from .types import ErrorType, ErrorSeverity, RecoveryStrategy


@dataclass
class ClassifiedError:
    """Classified error with recovery information."""

    error_type: ErrorType
    message: str
    recoverable: bool
    retryable: bool
    source: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    suggested_strategy: RecoveryStrategy = RecoveryStrategy.STOP
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "retryable": self.retryable,
            "source": self.source,
            "severity": self.severity.value,
            "suggested_strategy": self.suggested_strategy.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClassifiedError":
        return cls(
            error_type=ErrorType(data["error_type"]),
            message=data["message"],
            recoverable=data["recoverable"],
            retryable=data["retryable"],
            source=data["source"],
            severity=ErrorSeverity(data.get("severity", "error")),
            suggested_strategy=RecoveryStrategy(data.get("suggested_strategy", "stop")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ErrorContext:
    """Context for error classification."""

    state: Optional["AgentState"] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[dict] = None
    evaluation_result: Optional[dict] = None
    attempt: int = 0
    max_retries: int = 2
    exception: Optional[Exception] = None


# Import here to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.harness.state import AgentState