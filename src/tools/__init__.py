"""Tools module for Agent Harness."""

from .base import (
    BaseTool,
    ToolResult,
    ToolError,
    ToolSchema,
    ToolRegistry,
    get_tool_registry,
    register_tool,
)
from .weather import WeatherTool, WeatherInput, WeatherOutput, get_weather
from .attractions import AttractionsTool, AttractionSearchInput, AttractionSearchOutput, search_attractions


def register_default_tools() -> ToolRegistry:
    """Register built-in tools into the global registry.

    Idempotent: safe to call multiple times.
    """
    registry = get_tool_registry()
    defaults = [WeatherTool(), AttractionsTool()]
    for tool in defaults:
        if tool.name not in registry._tools:
            registry.register(tool)
    return registry


register_default_tools()


__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolError",
    "ToolSchema",
    "ToolRegistry",
    "get_tool_registry",
    "register_tool",
    "register_default_tools",
    "WeatherTool",
    "WeatherInput",
    "WeatherOutput",
    "get_weather",
    "AttractionsTool",
    "AttractionSearchInput",
    "AttractionSearchOutput",
    "search_attractions",
]