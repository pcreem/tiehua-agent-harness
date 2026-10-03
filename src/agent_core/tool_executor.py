"""Tool Executor with safety checks and observability."""

import time
import json
from typing import Dict, Any, List, Optional

from src.tools.base import BaseTool, ToolResult
from src.safety import create_guardrails_pipeline
from src.observability import get_metrics, trace_tool_call

from opentelemetry.trace import Status, StatusCode


class ToolExecutor:
    """Executes tools with safety checks and observability."""

    def __init__(self, tools: list, max_tool_calls: int = 5):
        self.tools = {tool.name: tool for tool in tools}
        self.max_tool_calls = max_tool_calls
        self._safety_pipeline = None  # Lazy init
        self._tool_map = {tool.name: tool for tool in tools}

    def _get_safety_pipeline(self):
        if not hasattr(self, '_safety_pipeline') or self._safety_pipeline is None:
            from src.safety import create_guardrails_pipeline
            self._safety_pipeline = create_guardrails_pipeline(mode='strict', enable_content_filter=False)
        return self._safety_pipeline

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with arguments and return a plain dict."""
        tool = self._tool_map.get(name)
        if not tool:
            return {
                "success": False,
                "error_type": "unknown_tool",
                "message": f"Unknown tool: {name}"
            }

        # Safety check on tool arguments
        safety = self._get_safety_pipeline()
        tool_safety = safety.check_tool_args(name, args)

        if not tool_safety.passed and tool_safety.action == "block":
            return {
                "success": False,
                "error_type": "tool_safety_violation",
                "message": f"工具參數安全檢查失敗: {'; '.join(v.message for v in tool_safety.violations)}",
            }

        metrics = get_metrics()
        start_time = time.time()

        with trace_tool_call(name, args):
            try:
                input_obj = tool.validate_input(args)
                tool_result = tool.execute(input_obj)
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_tool_call(name, False, duration)
                return {
                    "success": False,
                    "error_type": "tool_execution_failed",
                    "message": f"Tool execution failed: {str(e)}",
                }

        duration = time.time() - start_time

        # Normalize ToolResult into a plain dict payload
        if isinstance(tool_result, ToolResult):
            if tool_result.success and tool_result.data is not None:
                payload = tool_result.data.model_dump()
                payload["success"] = True
                metrics.record_tool_call(name, True, duration)
                return payload

            error = tool_result.error
            metrics.record_tool_call(name, False, duration)
            return {
                "success": False,
                "error_type": error.type if error else "tool_execution_failed",
                "message": error.message if error else "Tool execution failed",
                "retryable": error.retryable if error else False,
            }

        metrics.record_tool_call(name, True, duration)
        return tool_result if isinstance(tool_result, dict) else {"success": True, "result": str(tool_result)}

    def get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM function calling."""
        from src.tools import get_tool_registry
        return get_tool_registry().get_schemas_dict()