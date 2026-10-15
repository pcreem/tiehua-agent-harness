"""OpenRouter LLM client implementation with Mistral fallback."""

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


# Mistral 模型
MISTRAL_MODELS = [
    "mistral-small-latest",
    "mistral-medium-latest",
    "magistral-small-latest",
    "magistral-medium-latest",
]


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
        self._mistral_client = None
        self._mistral_api_key = config.mistral_api_key or os.getenv("MISTRAL_API_KEY")

    def _get_client_for_model(self, model: str):
        """Get the appropriate client for a model."""
        if model in MISTRAL_MODELS:
            if self._mistral_client is None:
                if not self._mistral_api_key:
                    raise ValueError("MISTRAL_API_KEY not configured")
                self._mistral_client = OpenAI(
                    api_key=self._mistral_api_key,
                    base_url="https://api.mistral.ai/v1",
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
            return self._mistral_client
        return self.client

    def complete(
        self,
        messages: List[LLMMessage],
        tools: List = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto",
    ) -> "LLMResponse":
        """Call LLM via OpenRouter API with Mistral fallback."""
        from src.llm.types import LLMResponse

        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [tool.to_dict() for tool in tools]

        # Try primary model first, then fallback to Mistral models
        models_to_try = [self._model]
        if self._model not in MISTRAL_MODELS and self._mistral_api_key:
            models_to_try += MISTRAL_MODELS

        last_error = None
        for model in models_to_try:
            client = self._get_client_for_model(model)
            try:
                response = client.chat.completions.create(
                    model=model,
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
                logger.warning(f"Model {model} failed: {e}, trying next...")
                last_error = e
                continue

        raise RuntimeError(f"All models failed. Last error: {last_error}")

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
    mistral_api_key: str = None,
) -> "OpenRouterClient":
    """Factory function to create OpenRouter client with Mistral fallback."""
    from src.llm.types import LLMConfig

    config = LLMConfig(
        model=model or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        base_url=base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
        mistral_api_key=mistral_api_key,
    )

    if not config.api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    return OpenRouterClient(config)