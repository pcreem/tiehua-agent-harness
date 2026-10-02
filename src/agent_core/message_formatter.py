"""Message formatting utilities for Agent."""

import json
from typing import List, Dict, Any


class MessageFormatter:
    """Handles message formatting for LLM communication."""

    @staticmethod
    def to_openai_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal message format to OpenAI API format."""
        openai_messages = []
        for msg in messages:
            if msg["role"] == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg["content"],
                    "tool_call_id": msg.get("tool_call_id", ""),
                })
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                openai_messages.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": msg.get("tool_calls", []),
                })
            else:
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg.get("content"),
                })
        return openai_messages

    @staticmethod
    def format_tool_call(tc) -> Dict:
        """Format tool call for message history."""
        return {
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        }

    @staticmethod
    def format_tool_result(result: Dict, tool_call_id: str) -> Dict:
        """Format tool result for message history."""
        return {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
            "tool_call_id": result.get("call_id", ""),
        }

    @staticmethod
    def format_tool_call_data(tc) -> Dict:
        """Format tool call for message history."""
        return {
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        }

    @staticmethod
    def assistant_message(content: str, tool_calls: List = None) -> Dict:
        """Create assistant message."""
        msg = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    @staticmethod
    def user_message(content: str) -> Dict:
        """Create user message."""
        return {"role": "user", "content": content}

    @staticmethod
    def system_message(content: str) -> Dict:
        """Create system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def tool_message(content: str, tool_call_id: str, name: str = None) -> Dict:
        """Create tool message."""
        msg = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }
        if name:
            msg["name"] = name
        return msg

    @staticmethod
    def tool_result_message(result: Dict, tool_call_id: str) -> Dict:
        """Create tool result message for LLM."""
        return {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
            "tool_call_id": result.get("call_id", ""),
        }