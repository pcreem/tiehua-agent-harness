"""Agent with Function Calling for Tourism Tools

Uses centralized AgentState for all execution state management.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from src.state import AgentState, initialize_state
from src.tools import WeatherTool, AttractionsTool, get_tool_registry
from src.observability import (
    get_logger,
    get_metrics,
    get_tracer,
    trace_agent_run,
    trace_llm_call,
    trace_tool_call,
    agent_logger,
    tool_logger,
    llm_logger,
)
from src.safety import create_guardrails_pipeline, get_guardrails_pipeline

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

        # Initialize safety pipeline
        self.safety_pipeline = create_guardrails_pipeline()

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

        # Safety check on input
        safety_result = self.safety_pipeline.check_input(user_query)
        if not safety_result.passed and safety_result.action == "block":
            return {
                "query": user_query,
                "final_answer": "您的輸入包含不安全內容，無法處理。",
                "messages": state.messages,
                "tool_calls": [],
                "tool_results": [],
                "retry_count": 0,
                "feedback": [],
                "evaluation": None,
                "success": False,
                "state": state.to_dict(),
                "safety": {
                    "action": safety_result.action,
                    "violations": [v.message for v in safety_result.violations],
                },
            }
        if safety_result.sanitized_input:
            state.messages[-1]["content"] = safety_result.sanitized_input

        # Observability: track agent run
        start_time = time.time()
        run_id = state.run_id
        metrics = get_metrics()
        tracer = get_tracer()

        with trace_agent_run(user_query, run_id) as span:
            try:
                state = self.run_with_state(state)
                duration = time.time() - start_time

                # Record metrics
                metrics.record_agent_run("passed", duration)
                span.set_attribute("agent.duration_seconds", duration)

                # Record metrics
                metrics.llm_calls_total.labels(model=self.model, status="success").inc()

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
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_agent_run("failed", duration)
                metrics.record_error(type(e).__name__, "agent")
                agent_logger.error("agent_run_failed", query=user_query, error=str(e), duration=duration)
                raise

    def run_with_state(self, state: "AgentState") -> "AgentState":
        """Run agent with pre-initialized state, mutate and return state"""
        tool_call_count = 0
        metrics = get_metrics()
        tracer = get_tracer()

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
            llm_start = time.time()
            with trace_llm_call(self.model) as llm_span:
                try:
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=openai_messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=0.3,
                    )
                    llm_span.set_attribute("llm.duration_seconds", time.time() - llm_start)
                except Exception as e:
                    llm_span.set_status(Status(StatusCode.ERROR, str(e)))
                    llm_span.record_exception(e)
                    llm_logger.error("llm_call_failed", model=self.model, error=str(e))
                    raise RuntimeError(f"LLM API 呼叫失敗: {e}")

            if not response.choices or not response.choices[0].message:
                raise RuntimeError(f"LLM 回傳異常 (無 choices 或 message): {response}")

            message = response.choices[0].message

            # Record LLM metrics
            llm_duration = time.time() - llm_start
            metrics.record_llm_call(
                model=self.model,
                success=True,
                duration=llm_duration,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            llm_span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens if response.usage else 0)
            llm_span.set_attribute("llm.completion_tokens", response.usage.completion_tokens if response.usage else 0)

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
                tool_logger.info("tool_call_start", tool=name, args=args)

                # Safety check on tool arguments
                tool_safety = self.safety_pipeline.check_tool_args(name, args)
                if not tool_safety.passed and tool_safety.action == "block":
                    result = {
                        "success": False,
                        "error_type": "tool_safety_violation",
                        "message": f"工具參數安全檢查失敗: {'; '.join(v.message for v in tool_safety.violations)}",
                    }
                    tool_logger.warning("tool_call_blocked", tool=name, violations=[v.message for v in tool_safety.violations])
                else:
                    # Record tool call in state
                    state.add_tool_call(name, args, tool_call.id)

                    # Execute tool with observability
                    tool_start = time.time()
                    with trace_tool_call(name, args) as tool_span:
                        try:
                            result = self._execute_tool(name, args)
                            tool_span.set_attribute("tool.success", result.get("success", False))
                        except Exception as e:
                            tool_span.set_status(Status(StatusCode.ERROR, str(e)))
                            tool_span.record_exception(e)
                            tool_logger.error("tool_call_failed", tool=name, error=str(e))
                            result = {
                                "success": False,
                                "error_type": "tool_execution_failed",
                                "message": f"Tool execution failed: {str(e)}",
                            }
                        finally:
                            tool_duration = time.time() - tool_start
                            metrics.record_tool_call(name, result.get("success", False), tool_duration)
                            tool_span.set_attribute("tool.duration_seconds", tool_duration)

                tool_logger.info("tool_call_end", tool=name, success=result.get("success", False), duration=tool_duration)

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