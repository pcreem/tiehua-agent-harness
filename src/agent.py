"""Agent with Function Calling for Tourism Tools

Uses centralized AgentState for all execution state management.
"""

import os
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from src.state import AgentState, initialize_state
from src.tools import WeatherTool, AttractionsTool, get_tool_registry

load_dotenv()


# Tool schemas for LLM function calling (backward compatibility)
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
        max_tool_calls: int = 5,
        llm_client = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        # Store model for backward compatibility
        self.model = model

        # Use provided LLM client or create default OpenRouter client
        self.llm_client = llm_client
        if self.llm_client is None:
            # Backward compatibility: use OpenAI client directly so tests can mock src.agent.OpenAI
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=30.0)

        self.max_tool_calls = max_tool_calls

        # Initialize tools and registry
        self.tools = [
            WeatherTool(),
            AttractionsTool(),
        ]
        self._tool_map = {tool.name: tool for tool in self.tools}

    def _get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM function calling."""
        schemas = []
        for tool in self.tools:
            schema = tool.get_schema()
            schemas.append(schema.to_dict())
        return schemas

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with arguments"""
        tool = self._tool_map.get(name)
        if not tool:
            return {
                "success": False,
                "error_type": "unknown_tool",
                "message": f"Unknown tool: {name}"
            }

        try:
            # Validate and execute
            input_obj = tool.validate_input(args)
            result = tool.execute(input_obj)
            if result.success:
                return result.data.model_dump()
            else:
                return {
                    "success": False,
                    "error_type": result.error.type,
                    "message": result.error.message,
                }
        except Exception as e:
            return {
                "success": False,
                "error_type": "tool_execution_failed",
                "message": f"Tool execution failed: {str(e)}",
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
            # Prepare messages for LLM (convert state.messages to OpenAI format)
            openai_messages = []
            for msg in state.messages:
                if msg["role"] == "tool":
                    openai_messages.append({
                        "role": "tool",
                        "content": msg["content"],
                        "tool_call_id": msg.get("tool_call_id", ""),
                    })
                elif msg["role"] == "assistant" and "tool_calls" in msg:
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

            # Prepare tools for LLM
            tools = []
            for tool in self.tools:
                schema = tool.get_schema()
                tools.append(schema.to_dict())

            # Call LLM with current messages from state
            print(f"[DEBUG] Calling LLM (attempt {tool_call_count + 1})...", flush=True)
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    tools=tools,
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


def create_agent(api_key: Optional[str] = None, llm_client=None) -> "Agent":
    """Factory function to create agent"""
    return Agent(api_key=api_key, llm_client=llm_client)


if __name__ == "__main__":
    # Quick test
    agent = create_agent()
    result = agent.run("台東今天天氣如何？")
    print(json.dumps(result, indent=2, ensure_ascii=False))