"""Mock LLM client for testing."""

import json
from typing import List, Optional, Any, Dict
from .types import LLMClient, LLMMessage, LLMResponse, LLMTool


class MockLLMClient:
    """Mock LLM client for testing without network calls."""

    def __init__(self, responses: List[LLMResponse] = None):
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None

    def complete(
        self,
        messages: List,
        tools: List = None,
        temperature: float = 0.3,
        max_tokens: int = None,
        tool_choice: str = "auto",
    ):
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools

        if self.responses:
            if self.call_count <= len(self.responses):
                return self.responses[self.call_count - 1]

        # Default response
        from src.llm.types import LLMResponse
        return LLMResponse(
            content="Mock response",
            tool_calls=[],
            finish_reason="stop",
            model="mock-model",
        )

    def get_model_name(self) -> str:
        return "mock-model"

    def add_response(self, response: "LLMResponse") -> None:
        """Add a response to the queue."""
        self.responses.append(response)

    def clear(self) -> None:
        """Clear call history."""
        self.responses = []
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None

    def set_side_effect(self, side_effect) -> None:
        """Set side effect for complete method."""
        self.complete = side_effect


def create_mock_client(responses: List = None) -> "MockLLMClient":
    """Factory function to create mock client."""
    return MockLLMClient(responses)


def create_mock_tool_call_response(
    tool_name: str,
    arguments: Dict,
    call_id: str = "call_1",
    content: str = "I'll call the tool",
) -> "LLMResponse":
    """Create a mock response with tool call."""
    from src.llm.types import LLMResponse, LLMTool
    return LLMResponse(
        content=content,
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments),
            }
        }],
        finish_reason="tool_calls",
        model="mock-model",
    )


def create_mock_final_response(content: str) -> "LLMResponse":
    """Create a mock final response."""
    from src.llm.types import LLMResponse
    return LLMResponse(
        content=content,
        tool_calls=[],
        finish_reason="stop",
        model="mock-model",
    )