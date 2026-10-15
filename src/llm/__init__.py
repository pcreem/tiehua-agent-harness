"""LLM module for Agent Harness."""

from .types import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMTool,
    LLMConfig,
)
from .client import KiloClient, create_llm_client

# Backwards compatibility
OpenRouterClient = KiloClient

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMTool",
    "LLMConfig",
    "KiloClient",
    "OpenRouterClient",
    "create_llm_client",
]