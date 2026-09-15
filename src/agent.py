"""Agent with Function Calling for Tourism Tools"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv


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
    """Tourism Agent with Function Calling"""

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

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.max_tool_calls = max_tool_calls

        # Conversation state
        self.messages: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_results: List[Dict[str, Any]] = []

    def _reset_state(self):
        """Reset conversation state for new query"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_calls = []
        self.tool_results = []

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
        self._reset_state()
        self.messages.append({"role": "user", "content": user_query})

        tool_call_count = 0

        while tool_call_count < self.max_tool_calls:
            # Call LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                )
            except Exception as e:
                raise RuntimeError(f"LLM API 呼叫失敗: {e}")

            if not response.choices or not response.choices[0].message:
                raise RuntimeError(f"LLM 回傳異常 (無 choices 或 message): {response}")

            message = response.choices[0].message

            # If no tool calls, we have final answer
            if not message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": message.content
                })
                break

            # Execute tool calls
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                # Record tool call
                self.tool_calls.append({
                    "name": name,
                    "arguments": args,
                    "call_id": tool_call.id
                })

                # Execute tool
                result = self._execute_tool(name, args)

                # Record tool result
                self.tool_results.append({
                    "name": name,
                    "arguments": args,
                    "result": result,
                    "call_id": tool_call.id
                })

                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

                tool_call_count += 1

        # Get final answer
        final_answer = self.messages[-1].get("content", "") if self.messages else ""

        return {
            "query": user_query,
            "final_answer": final_answer,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "success": True
        }


def create_agent(api_key: Optional[str] = None) -> Agent:
    """Factory function to create agent"""
    return Agent(api_key=api_key)


if __name__ == "__main__":
    # Quick test
    agent = create_agent()
    result = agent.run("台東今天天氣如何？")
    print(json.dumps(result, indent=2, ensure_ascii=False))