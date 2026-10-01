"""Error classification module for Harness."""

from .types import (
    ErrorType,
    ErrorSeverity,
    RecoveryStrategy,
)
from .models import (
    ClassifiedError,
    ErrorContext,
)
from .classifier import ErrorClassifier

__all__ = [
    "ErrorType",
    "ErrorSeverity",
    "RecoveryStrategy",
    "ClassifiedError",
    "ErrorContext",
    "ErrorClassifier",
]