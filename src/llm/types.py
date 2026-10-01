"""LLM types and data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None  # Provider-specific raw response

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def get_tool_calls(self) -> List[Dict]:
        return self.tool_calls


@dataclass
class LLMMessage:
    """Standardized message format."""

    role: str  # system, user, assistant, tool
    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        msg = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "LLMMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = None, tool_calls: List[Dict] = None) -> "LLMMessage":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str = None) -> "LLMMessage":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class LLMTool:
    """Tool definition for LLM function calling."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def complete(
        self,
        messages: List[LLMMessage],
        tools: List[LLMTool] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto",
    ) -> "LLMResponse":
        """Call LLM with messages and optional tools."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier."""
        pass

    def estimate_tokens(self, messages: List[LLMMessage]) -> int:
        """Estimate token count (approximate)."""
        total_chars = sum(len(m.content or "") for m in messages)
        return total_chars // 4  # Rough approximation


@dataclass
class LLMConfig:
    """LLM client configuration."""

    model: str
    base_url: str
    api_key: str
    timeout: float = 30.0
    max_retries: int = 2
    temperature: float = 0.3
    max_tokens: Optional[int] = None