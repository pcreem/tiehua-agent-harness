"""Error type definitions for Harness."""

from enum import Enum


class ErrorType(str, Enum):
    """Standardized error types for Harness error classification."""

    # Tool errors
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_INVALID_INPUT = "tool_invalid_input"
    TOOL_INVALID_OUTPUT = "tool_invalid_output"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"

    # LLM errors
    LLM_TIMEOUT = "llm_timeout"
    LLM_ERROR = "llm_error"
    LLM_INVALID_RESPONSE = "llm_invalid_response"
    LLM_RATE_LIMITED = "llm_rate_limited"

    # Evaluation errors
    EVALUATION_FAILURE = "evaluation_failure"
    GROUNDEDNESS_FAILURE = "groundedness_failure"
    TOOL_CORRECTNESS_FAILURE = "tool_correctness_failure"
    COMPLETENESS_FAILURE = "completeness_failure"

    # Policy errors
    POLICY_FAILURE = "policy_failure"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"

    # Safety errors
    SAFETY_INPUT_VIOLATION = "safety_input_violation"
    SAFETY_OUTPUT_VIOLATION = "safety_output_violation"
    SAFETY_TOOL_VIOLATION = "safety_tool_violation"
    PROMPT_INJECTION = "prompt_injection"

    # Unknown
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for retry policy."""

    RETRY_TOOL = "retry_tool"           # Retry the same tool
    RETRY_LLM = "retry_llm"             # Retry LLM call
    RETRY_EVALUATION = "retry_evaluation"  # Retry evaluation
    RETOOL = "retool"                   # Use different tool
    REPLAN = "replan"                   # Re-plan tool sequence
    REGENERATE = "regenerate"           # Regenerate response
    RETRIEVE_AGAIN = "retrieve_again"   # Re-retrieve data
    FALLBACK = "fallback"               # Fallback response
    STOP = "stop"                       # Stop execution