"""Retry policy for Harness."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from src.errors.types import RecoveryStrategy
from src.errors.models import ClassifiedError
from src.state import AgentState


class RetryDecision(str, Enum):
    """Retry decision."""

    RETRY = "retry"
    STOP = "stop"
    FALLBACK = "fallback"


@dataclass
class RetryPolicy:
    """Policy for deciding retry behavior."""

    max_retries: int = 2

    def decide(
        self,
        state: AgentState,
        errors: List["ClassifiedError"],
    ) -> RetryDecision:
        """Decide whether to retry, stop, or fallback."""
        # Check for critical errors that should stop immediately
        for error in errors:
            if error.error_type.value in [
                "safety_input_violation",
                "safety_output_violation",
                "safety_tool_violation",
                "prompt_injection",
            ]:
                return RetryDecision.STOP

        # Check retry exhaustion
        if state.attempt >= self.max_retries:
            return RetryDecision.FALLBACK

        # Check if any error is retryable
        retryable = any(error.retryable for error in errors)
        if retryable:
            return RetryDecision.RETRY

        # Check if any error is recoverable (but not retryable - e.g., wrong tool)
        recoverable = any(error.recoverable for error in errors)
        if recoverable:
            return RetryDecision.RETRY

        return RetryDecision.STOP

    def get_suggested_strategy(self, errors: List["ClassifiedError"]) -> List[str]:
        """Get suggested recovery strategies from errors."""
        strategies = []
        for error in errors:
            strategy = error.suggested_strategy
            if hasattr(strategy, 'value'):
                strategies.append(strategy.value)
            else:
                strategies.append(str(strategy))
        return list(set(strategies))  # deduplicate

    def should_retry(self, state: AgentState, errors: List["ClassifiedError"]) -> bool:
        """Quick check if should retry."""
        decision = self.decide(state, errors)
        return decision == RetryDecision.RETRY