"""Harness result module."""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from src.state import AgentState


@dataclass
class HarnessResult:
    """Result of a harness execution."""

    query: str
    final_answer: str
    state: AgentState
    success: bool
    status: str  # passed, failed, fallback
    retry_count: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    evaluation: Optional[Dict[str, Any]] = None
    feedback: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "success": self.success,
            "status": self.status,
            "retry_count": self.retry_count,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "evaluation": self.evaluation,
            "feedback": self.feedback,
            "state": self.state.to_dict(),
        }