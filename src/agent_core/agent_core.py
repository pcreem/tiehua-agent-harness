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
        fallback_models: Optional[List[str]] = None,
        prompt_manager=None,  # Lazy import to avoid circular dependency
    ):
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        # Use provided LLM client or create default OpenRouter client with fallback
        self.llm_client = llm_client
        if self.llm_client is None:
            from src.agent_core.llm_client import LLMClient
            self.llm_client = LLMClient(
                api_key=self.api_key,
                base_url=base_url,
                model=model,
                fallback_models=fallback_models,
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

        # Prompt Manager (lazy initialization to avoid circular import)
        self._prompt_manager = prompt_manager
        self._prompt_manager_class = None

        # Default role (can be overridden via state.metadata["role"])
        self.default_role = "local_guide"
        self.default_output_format = "markdown_output"

    @property
    def prompt_manager(self):
        """Lazy-load PromptManager to avoid circular imports."""
        if self._prompt_manager is None:
            if self._prompt_manager_class is None:
                from src.harness.prompt_manager import PromptManager
                self._prompt_manager_class = PromptManager
            self._prompt_manager = self._prompt_manager_class()
        return self._prompt_manager

    def _get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM function calling."""
        from src.tools import get_tool_registry
        return get_tool_registry().get_schemas_dict()

    def _build_system_prompt(self, state: AgentState) -> str:
        """Build system prompt dynamically using PromptManager."""
        # Ensure state has required metadata for prompt composition
        if "role" not in state.metadata:
            state.metadata["role"] = self.default_role
        if "output_format" not in state.metadata:
            state.metadata["output_format"] = self.default_output_format

        # Compose system prompt from modules
        system_prompt = self.prompt_manager.compose_system_prompt(state)

        # Append tool usage instructions (dynamic based on available tools)
        tool_instructions = self._build_tool_instructions()
        system_prompt += f"\n\n{tool_instructions}"

        return system_prompt

    def _build_tool_instructions(self) -> str:
        """Build tool usage instructions."""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}({self._format_tool_params(tool)}) - {tool.description}")
        tools_text = "\n".join(tool_descriptions)

        return f"""可用工具：
{tools_text}

原則：
- 使用者詢問天氣 → 呼叫 get_weather
- 使用者詢問景點/住宿/活動 → 呼叫 search_attractions
- 使用者同時詢問天氣和景點 → 先查天氣，再根據天氣決定搜尋策略
- 工具回傳結構化資料，你需整理成自然語言回答
- 不要編造工具沒有的資訊
- 回答語言跟隨使用者查詢語言（state.language）"""

    def _format_tool_params(self, tool) -> str:
        """Format tool parameters for prompt."""
        if hasattr(tool, 'input_schema'):
            schema = tool.input_schema.model_json_schema()
            props = schema.get('properties', {})
            required = schema.get('required', [])
            params = []
            for name, prop in props.items():
                param_type = prop.get('type', 'string')
                desc = prop.get('description', '')
                req = " (required)" if name in required else ""
                params.append(f"{name}: {param_type}{req}")
            return ", ".join(params)
        return ""

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
            # Prepare messages for LLM (with dynamic system prompt)
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

                # Inject language from state into tool arguments
                if "language" not in args and state.language:
                    args["language"] = state.language

                # Safety check
                tool_safety = self.safety_pipeline.check_tool_args(name, args)
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
        """Convert state messages to OpenAI format with dynamic system prompt."""
        from src.agent_core.message_formatter import MessageFormatter
        formatter = MessageFormatter()

        # Build dynamic system prompt
        system_prompt = self._build_system_prompt(state)

        # Get existing messages (excluding any old system prompt)
        messages = [msg for msg in state.messages if msg.get("role") != "system"]

        # Prepend new system prompt
        messages = [{"role": "system", "content": system_prompt}] + messages

        return formatter.to_openai_format(messages)

    def _get_tool_schemas(self) -> List[Dict]:
        from src.tools import get_tool_registry
        return get_tool_registry().get_schemas_dict()

    def _call_llm(self, messages, tools):
        from src.observability import trace_llm_call, get_logger
        from src.agent_core.llm_client import ModelExhaustedError

        logger = get_logger(__name__)

        with trace_llm_call(self.model) as span:
            try:
                response = self.llm_client.complete(
                    messages=messages,
                    tools=tools,
                    temperature=self.temperature,
                )
                return response
            except ModelExhaustedError as e:
                # All models exhausted - return a fallback response
                logger.error("all_models_exhausted", tried_models=e.tried_models)
                from types import SimpleNamespace
                class MockChoice:
                    class MockMessage:
                        content = "抱歉，目前所有 AI 模型都已達到使用限制，無法處理您的請求。請稍後再試或聯繫管理員。"
                        tool_calls = None
                    message = MockMessage()
                class MockResponse:
                    choices = [MockChoice()]
                return MockResponse()
            except Exception as e:
                raise RuntimeError(f"LLM API 呼叫失敗: {e}")

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
        return self._tool_executor.execute(name, args)

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