"""Agent State Management

Centralized state container for the Agent Harness.
All execution flow revolves around this state object.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import copy


@dataclass
class AgentState:
    """Centralized state for Agent Harness execution.

    All Harness execution flow revolves around this state object.
    """

    # Original user query
    query: str = ""

    # Conversation history (system, user, assistant, tool messages)
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # Tool execution history
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # Retrieved documents (for future RAG integration)
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)

    # Final answer from LLM
    answer: Optional[str] = None

    # Evaluation result from Evaluator
    evaluation: Optional[Dict[str, Any]] = None

    # Structured feedback from Evaluator
    feedback: List[str] = field(default_factory=list)

    # Retry tracking
    retry_count: int = 0

    # Error tracking
    last_error: Optional[Dict[str, Any]] = None

    # Formal version fields (P0)
    run_id: str = field(default_factory=lambda: "")
    attempt: int = 0
    max_retries: int = 2
    status: str = "running"  # running, passed, failed, fallback
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to conversation history."""
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)
        self.update_timestamp()

    def add_tool_call(self, name: str, arguments: Dict[str, Any], call_id: str):
        """Record a tool call."""
        self.tool_calls.append({
            "name": name,
            "arguments": arguments,
            "call_id": call_id,
            "timestamp": datetime.now().isoformat()
        })
        self.update_timestamp()

    def add_tool_result(self, name: str, arguments: Dict[str, Any], result: Dict[str, Any], call_id: str):
        """Record a tool execution result."""
        self.tool_results.append({
            "name": name,
            "arguments": arguments,
            "result": result,
            "call_id": call_id,
            "timestamp": datetime.now().isoformat()
        })
        self.update_timestamp()

    def set_answer(self, answer: str):
        """Set the final answer."""
        self.answer = answer
        self.update_timestamp()

    def set_evaluation(self, evaluation: Dict[str, Any]):
        """Set evaluation result."""
        self.evaluation = evaluation
        self.update_timestamp()

    def add_feedback(self, feedback: str):
        """Add structured feedback."""
        self.feedback.append(feedback)
        self.update_timestamp()

    def increment_retry(self):
        """Increment retry counter."""
        self.retry_count += 1
        self.update_timestamp()

    def increment_attempt(self):
        """Increment attempt counter for retry loop."""
        self.attempt += 1
        self.update_timestamp()

    def set_status(self, status: str):
        """Set execution status."""
        self.status = status
        self.update_timestamp()

    def set_error(self, error: Dict[str, Any]):
        """Record last error."""
        self.last_error = error
        self.update_timestamp()

    def get_tool_calls_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Get all tool calls of a specific type."""
        return [tc for tc in self.tool_calls if tc["name"] == name]

    def get_latest_tool_result(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent result for a specific tool."""
        for tr in reversed(self.tool_results):
            if tr["name"] == name:
                return tr["result"]
        return None

    def has_called_tool(self, name: str) -> bool:
        """Check if a specific tool has been called."""
        return any(tc["name"] == name for tc in self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary (for serialization)."""
        return {
            "query": self.query,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "retrieved_docs": self.retrieved_docs,
            "answer": self.answer,
            "evaluation": self.evaluation,
            "feedback": self.feedback,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create AgentState from dictionary."""
        state = cls(
            query=data.get("query", ""),
            messages=data.get("messages", []),
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
            retrieved_docs=data.get("retrieved_docs", []),
            answer=data.get("answer"),
            evaluation=data.get("evaluation"),
            feedback=data.get("feedback", []),
            retry_count=data.get("retry_count", 0),
            last_error=data.get("last_error"),
            run_id=data.get("run_id", ""),
            attempt=data.get("attempt", 0),
            max_retries=data.get("max_retries", 2),
            status=data.get("status", "running"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
        return state

    def copy(self) -> "AgentState":
        """Create a deep copy of the state."""
        return copy.deepcopy(self)


def initialize_state(query: str, max_retries: int = 2) -> AgentState:
    """Initialize a new AgentState with the user query.

    Args:
        query: The original user query
        max_retries: Maximum number of retries allowed

    Returns:
        Initialized AgentState with system prompt in messages
    """
    from src.agent import SYSTEM_PROMPT
    import uuid

    state = AgentState(
        query=query,
        max_retries=max_retries,
        run_id=str(uuid.uuid4())[:8],
        attempt=0,
        status="running",
        metadata={},
    )
    state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    return state