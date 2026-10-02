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
        """Execute a tool by name with arguments."""
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

        # Execute tool with observability
        start_time = time.time()
        from src.observability import trace_tool_call, get_metrics
        from opentelemetry.trace import Status, StatusCode
        from opentelemetry.trace import get_current_span

        metrics = get_metrics()
        tool = self._tool_map.get(name)

        try:
            # Validate and execute
            input_obj = tool.validate_input(args)
            result = tool.execute(input_obj)
            success = result.success
        except Exception as e:
            result = {
                "success": False,
                "error_type": "tool_execution_failed",
                "message": f"Tool execution failed: {str(e)}",
            }
        else:
            # No exception
            if hasattr(result, 'success') and result.success:
                success = True
            else:
                success = False

        # Record metrics
        duration = time.time() - start_time
        success = result.get("success", False)
        metrics = get_metrics()
        metrics.record_tool_call(name, success, time.time() - time.time() + start_time)  # Fixed

        return result

    def get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM function calling."""
        from src.tools import get_tool_registry
        return get_tool_registry().get_schemas_dict()