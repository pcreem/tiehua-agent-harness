"""Base tool interface and types for standardized tool contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field


# Type variables for input/output
InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolError(BaseModel):
    """Structured tool error."""

    type: str
    message: str
    retryable: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


class ToolResult(BaseModel, Generic[OutputT]):
    """Standardized tool result."""

    success: bool
    data: Optional[OutputT] = None
    error: Optional[ToolError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: OutputT, metadata: Dict = None) -> "ToolResult":
        return cls(success=True, data=data, metadata=metadata or {})

    @classmethod
    def fail(cls, error_type: str, message: str, retryable: bool = True, details: Dict = None) -> "ToolResult":
        return cls(
            success=False,
            error=ToolError(
                type=error_type,
                message=message,
                retryable=retryable,
                details=details or {},
            ),
        )


class ToolSchema(BaseModel):
    """Tool schema for LLM function calling."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    strict: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LLM function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
            }
        }


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Base interface for all tools.

    Every tool must implement:
    - name: unique identifier
    - description: what the tool does
    - input_schema: Pydantic model for input validation
    - output_schema: Pydantic model for output structure
    - execute(): main logic
    """

    name: str
    description: str
    input_schema: type[InputT]
    output_schema: type[OutputT]

    def __init__(self, **kwargs):
        """Initialize tool with optional configuration."""
        self.config = kwargs

    @abstractmethod
    def execute(self, input: InputT) -> ToolResult[OutputT]:
        """Execute the tool with validated input."""
        pass

    def validate_input(self, raw_input: Dict[str, Any]) -> InputT:
        """Validate and parse raw input."""
        return self.input_schema(**raw_input)

    def get_schema(self) -> ToolSchema:
        """Get tool schema for LLM function calling."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.input_schema.model_json_schema(),
        )

    def __call__(self, *args, **kwargs) -> ToolResult[OutputT]:
        """Convenience call with positional or keyword arguments.
        
        Supports:
        - tool(WeatherInput(city="Taipei"))  # positional input object
        - tool(city="Taipei")                 # keyword arguments
        """
        if args and not kwargs:
            # Single positional argument (input object)
            input_obj = args[0]
            if isinstance(input_obj, self.input_schema):
                return self.execute(input_obj)
            # If it's a dict, validate it
            if isinstance(input_obj, dict):
                input_obj = self.validate_input(input_obj)
                return self.execute(input_obj)
            raise TypeError(f"Expected {self.input_schema.__name__} or dict, got {type(input_obj)}")
        elif kwargs and not args:
            # Keyword arguments
            input_obj = self.validate_input(kwargs)
            return self.execute(input_obj)
        else:
            raise TypeError("Expected either single positional argument or keyword arguments")


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self._tools.get(name)

    def all(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def names(self) -> List[str]:
        """Get all tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> List[ToolSchema]:
        """Get all tool schemas for LLM."""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_schemas_dict(self) -> List[Dict[str, Any]]:
        """Get all tool schemas as dictionaries for LLM function calling."""
        return [tool.get_schema().to_dict() for tool in self._tools.values()]

    def execute(self, name: str, input: Dict[str, Any]) -> ToolResult:
        """Execute tool by name with raw input."""
        tool = self.get(name)
        if not tool:
            return ToolResult.fail(
                error_type="tool_not_found",
                message=f"Tool '{name}' not found",
                retryable=False,
            )
        try:
            validated_input = tool.validate_input(input)
            return tool.execute(validated_input)
        except Exception as e:
            return ToolResult.fail(
                error_type="validation_error",
                message=f"Invalid input for tool '{name}': {str(e)}",
                retryable=False,
            )


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: BaseTool) -> None:
    """Register tool in global registry."""
    get_tool_registry().register(tool)