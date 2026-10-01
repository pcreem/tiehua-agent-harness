"""Agent with Function Calling for Tourism Tools

Uses centralized AgentState for all execution state management.
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from src.state import AgentState, initialize_state


load_dotenv()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., Taitung, Taipei, 台東, 台北)"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "Search attractions in a city by category",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., Taitung, Taipei, 台東, 台北)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category: accommodation, activity, sightseeing, food, nature, service",
                        "enum": ["accommodation", "activity", "sightseeing", "food", "nature", "service"]
                    }
                },
                "required": ["city"]
            }
        }
    }
]


SYSTEM_PROMPT = """你是鐵花觀光代理人，協助使用者規劃台東旅遊。

可用工具：
1. get_weather(city) - 查詢即時天氣
2. search_attractions(city, category) - 搜尋景點/住宿/活動

原則：
- 使用者詢問天氣 → 呼叫 get_weather
- 使用者詢問景點/住宿/活動 → 呼叫 search_attractions
- 使用者同時詢問天氣和景點 → 先查天氣，再根據天氣決定搜尋策略
- 工具回傳結構化資料，你需整理成自然語言回答
- 不要編造工具沒有的資訊
- 回答用繁體中文"""


class Agent:
    """Tourism Agent with Function Calling and Centralized State Management"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url: str = "https://openrouter.ai/api/v1",
        max_tool_calls: int = 5
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=30.0)
        self.model = model
        self.max_tool_calls = max_tool_calls

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with arguments"""
        if name == "get_weather":
            from src.tools.weather import get_weather
            return get_weather(**args)
        elif name == "search_attractions":
            from src.tools.attractions import search_attractions
            return search_attractions(**args)
        else:
            return {
                "success": False,
                "error_type": "unknown_tool",
                "message": f"Unknown tool: {name}"
            }

    def run(self, user_query: str) -> Dict[str, Any]:
        """Run agent with user query, return final answer and state"""
        # Initialize centralized state
        state = initialize_state(user_query)
        state.add_message("user", user_query)
        state = self.run_with_state(state)

        # Return dict for backward compatibility
        return {
            "query": state.query,
            "final_answer": state.answer or "",
            "messages": state.messages,
            "tool_calls": state.tool_calls,
            "tool_results": state.tool_results,
            "retry_count": state.retry_count,
            "feedback": state.feedback,
            "evaluation": state.evaluation,
            "success": True,
            "state": state.to_dict()
        }

    def run_with_state(self, state: "AgentState") -> "AgentState":
        """Run agent with pre-initialized state, mutate and return state"""
        tool_call_count = 0

        while tool_call_count < self.max_tool_calls:
            # Call LLM with current messages from state
            print(f"[DEBUG] Calling LLM (attempt {tool_call_count + 1})...", flush=True)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=state.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                )
                print(f"[DEBUG] LLM responded", flush=True)
            except Exception as e:
                print(f"[DEBUG] LLM API error: {e}", flush=True)
                raise RuntimeError(f"LLM API 呼叫失敗: {e}")

            if not response.choices or not response.choices[0].message:
                raise RuntimeError(f"LLM 回傳異常 (無 choices 或 message): {response}")

            message = response.choices[0].message

            # If no tool calls, we have final answer
            if not message.tool_calls:
                final_content = message.content or ""
                state.add_message("assistant", final_content)
                state.set_answer(final_content)
                break

            # Execute tool calls
            tool_calls_data = []
            for tc in message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })

            state.add_message(
                "assistant",
                message.content or "",
                tool_calls=tool_calls_data
            )

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"[DEBUG] Executing tool: {name} with args: {args}", flush=True)

                # Record tool call in state
                state.add_tool_call(name, args, tool_call.id)

                # Execute tool
                result = self._execute_tool(name, args)
                print(f"[DEBUG] Tool result: {result.get('success')}", flush=True)

                # Record tool result in state
                state.add_tool_result(name, args, result, tool_call.id)

                # Add tool result to messages for LLM
                state.add_message(
                    "tool",
                    json.dumps(result, ensure_ascii=False),
                    tool_call_id=tool_call.id
                )

                tool_call_count += 1

            state.increment_retry()

        return state


def create_agent(api_key: Optional[str] = None) -> Agent:
    """Factory function to create agent"""
    return Agent(api_key=api_key)


if __name__ == "__main__":
    # Quick test
    agent = create_agent()
    result = agent.run("台東今天天氣如何？")
    print(json.dumps(result, indent=2, ensure_ascii=False))