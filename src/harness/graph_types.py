"""Shared types for Harness Graph."""

from typing import Optional, List, TypedDict
from src.state import AgentState
from src.errors.models import ClassifiedError
from src.harness.retry import RetryDecision


class HarnessGraphState(TypedDict):
    """LangGraph state schema - extends AgentState with graph-specific fields."""
    agent_state: AgentState
    evaluation_result: Optional[dict]
    errors: List[ClassifiedError]
    retry_decision: Optional[RetryDecision]
    current_node: str
    execution_path: List[str]