"""Agent Core - Main orchestration logic for the Tourism Agent."""

import os
import json
import time
from typing import List, Dict, Any, Optional, Literal

from opentelemetry.trace import Status, StatusCode

from src.agent_core.llm_client import LLMClient
from src.agent_core.tool_executor import ToolExecutor
from src.agent_core.message_formatter import MessageFormatter
from src.state import AgentState, initialize_state
from src.tools import get_tool_registry
from src.safety import create_guardrails_pipeline, get_guardrails_pipeline
from src.observability import (
    get_logger, get_metrics, get_tracer,
    trace_agent_run, trace_llm_call, trace_tool_call,
    agent_logger, tool_logger, llm_logger
)

from src.tools import get_tool_registry
from src.tools.base import BaseTool

from src.safety import create_guardrails_pipeline, get_guardrails_pipeline


class AgentCore:
    """Core agent orchestration logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url: str = "https://openrouter.ai/api/v1",
        max_tool_calls: int = 5,
        temperature: float = 0.3,
        llm_client=None,
    ):
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        # Use provided LLM client or create default OpenRouter client
        self.llm_client = llm_client
        if self.llm_client is None:
            # Use LLMClient wrapper for consistent interface
            self.llm_client = LLMClient(
                api_key=self.api_key,
                base_url=base_url,
                model=model,
                timeout=30.0,
            )

        # Initialize tools
        from src.tools import get_tool_registry
        registry = get_tool_registry()
        self.tools = list(registry._tools.values())
        self._tool_map = {tool.name: tool for tool in registry._tools.values()}

        # Safety pipeline
        from src.safety import create_guardrails_pipeline
        self.safety_pipeline = create_guardrails_pipeline()

        # Components
        from src.agent_core.tool_executor import ToolExecutor
        from src.tools import get_tool_registry
        registry = get_tool_registry()
        self._tool_executor = ToolExecutor(
            tools=list(registry._tools.values()),
            max_tool_calls=max_tool_calls
        )
        self.message_formatter = MessageFormatter()

        # System prompt
        self.system_prompt = """你是鐵花觀光代理人，協助使用者規劃台東旅遊。

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

    def _get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM function calling."""
        from src.tools import get_tool_registry
        return get_tool_registry().get_schemas_dict()

    def run(self, user_query: str) -> Dict[str, Any]:
        """Run agent with user query, return final answer and state."""
        from src.state import initialize_state
        state = self._initialize_state(user_query)
        state = self.run_with_state(state)
        return self._build_result(state)

    def _initialize_state(self, user_query: str):
        from src.state import initialize_state
        state = initialize_state(user_query)
        state.add_message("user", user_query)
        return state

    def _build_result(self, state) -> Dict[str, Any]:
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

    def run(self, user_query: str) -> Dict[str, Any]:
        """Run agent with user query, return final answer and state."""
        state = self._initialize_state(user_query)
        state = self.run_with_state(state)
        return self._build_result(state)

    def run_with_state(self, state) -> "AgentState":
        """Run agent with pre-initialized state, mutate and return state."""
        tool_call_count = 0

        while tool_call_count < self.max_tool_calls:
            # Prepare messages for LLM
            openai_messages = self._prepare_messages(state)

            # Prepare tools for LLM
            tools = self._get_tool_schemas()

            # Call LLM
            response = self._call_llm(openai_messages, tools)

            if not response.choices or not response.choices[0].message:
                raise RuntimeError(f"LLM 回傳異常 (無 choices 或 message): {response}")

            message = response.choices[0].message

            # If no tool calls, we have final answer
            if not message.tool_calls:
                final_content = message.content or ""
                state.add_message("assistant", final_content)
                state.answer = final_content
                break

            # Execute tool calls
            tool_calls_data = self._format_tool_calls(message.tool_calls)
            state.add_message("assistant", message.content or "", tool_calls=tool_calls_data)

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                # Safety check
                tool_safety = self._safety_pipeline.check_tool_args(name, args)
                if not tool_safety.passed and tool_safety.action == "block":
                    result = {
                        "success": False,
                        "error_type": "tool_safety_violation",
                        "message": f"工具參數安全檢查失敗: {'; '.join(v.message for v in tool_safety.violations)}"
                    }
                else:
                    # Execute tool
                    result = self._execute_tool(name, args)

                # Record tool call/result in state
                state.add_tool_call(name, args, tool_call.id)
                state.add_tool_result(name, args, result, tool_call.id)
                state.add_message("tool", json.dumps(result, ensure_ascii=False), tool_call_id=tool_call.id)

                tool_call_count += 1

            state.increment_retry()

        return state

    def _prepare_messages(self, state):
        """Convert state messages to OpenAI format."""
        from src.agent_core.message_formatter import MessageFormatter
        formatter = MessageFormatter()
        return formatter.to_openai_format(state.messages)

    def _get_tool_schemas(self) -> List[Dict]:
        from src.tools import get_tool_registry
        return get_tool_registry().get_schemas_dict()

    def _call_llm(self, messages, tools):
        from src.observability import trace_llm_call
        from src.observability import get_metrics

        with trace_llm_call(self.model) as span:
            try:
                response = self.llm_client.complete(
                    messages=messages,
                    tools=tools,
                    temperature=0.3,
                )
                return response
            except Exception as e:
                raise RuntimeError(f"LLM API 呼叫失敗: {e}")

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with arguments"""
        from src.tools.base import BaseTool
        tool = getattr(self, '_tool_map', {}).get(name)
        if not tool:
            return {
                "success": False,
                "error_type": "unknown_tool",
                "message": f"Unknown tool: {name}"
            }

        try:
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

    def _format_tool_calls(self, tool_calls) -> List[Dict]:
        from src.agent_core.message_formatter import MessageFormatter
        formatter = MessageFormatter()
        tool_calls_data = []
        for tc in tool_calls:
            tool_calls_data.append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            })
        return tool_calls_data

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        from src.agent_core.tool_executor import ToolExecutor
        tool_executor = self._tool_executor
        return tool_executor.execute(name, args)

    def _get_safety_pipeline(self):
        from src.safety import get_guardrails_pipeline
        return get_guardrails_pipeline()


# Backward compatibility
class Agent:
    """Backward compatible Agent wrapper."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.core = AgentCore(api_key=api_key, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to core."""
        return getattr(self.core, name)

    def run(self, user_query: str) -> Dict[str, Any]:
        return self.core.run(user_query)

    def run_with_state(self, state):
        return self.core.run_with_state(state)


def create_agent(api_key: Optional[str] = None, **kwargs):
    """Factory function to create agent"""
    return Agent(api_key=api_key, **kwargs)