"""OpenRouter LLM client implementation."""

import os
import logging
from typing import List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from .types import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMTool,
    LLMConfig,
)

logger = logging.getLogger(__name__)


class OpenRouterClient(LLMClient):
    """OpenRouter LLM client using OpenAI SDK."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self._model = config.model

    def complete(
        self,
        messages: List[LLMMessage],
        tools: List = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto",
    ) -> "LLMResponse":
        """Call LLM via OpenRouter API."""
        from src.llm.types import LLMResponse

        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [tool.to_dict() for tool in tools]

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=openai_messages,
                tools=openai_tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens or self.config.max_tokens,
            )

            if not response.choices or not response.choices[0].message:
                raise RuntimeError(f"LLM 回傳異常 (無 choices 或 message): {response}")

            message = response.choices[0].message

            # Convert tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                raw=response,
            )

        except Exception as e:
            logger.exception(f"OpenRouter API error: {e}")
            raise RuntimeError(f"LLM API 呼叫失敗: {e}")

    def get_model_name(self) -> str:
        return self._model


def create_llm_client(
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    timeout: float = 30.0,
    max_retries: int = 2,
    temperature: float = 0.3,
    max_tokens: int = None,
) -> "OpenRouterClient":
    """Factory function to create OpenRouter client."""
    from src.llm.types import LLMConfig

    config = LLMConfig(
        model=model or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        base_url=base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not config.api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    return OpenRouterClient(config)