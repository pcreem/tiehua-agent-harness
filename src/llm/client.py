"""Kilo AI LLM client implementation with OpenRouter fallback."""

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


# Kilo AI 模型清單（優先序：越前面越優先使用）
KILO_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",      # 目前可用的免費模型
]

# OpenRouter 免費模型清單
OPENROUTER_FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3.5-lightning:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "thinkingmachines/inkling:free",
    "apodex/apodex-1.1-mini:free",
]


class KiloClient(LLMClient):
    """Kilo AI LLM client using OpenAI SDK with OpenRouter fallback."""

    def __init__(self, config: LLMConfig):
        self.config = config
        # Kilo client (primary)
        self.client = OpenAI(
            api_key=config.kilo_api_key or config.api_key,
            base_url=config.base_url or "https://api.kilo.ai/api/gateway",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self._model = config.model
        self._kilo_api_key = config.kilo_api_key or os.getenv("KILO_API_KEY")
        self._openrouter_api_key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        # Fallback client (lazy init)
        self._openrouter_client = None

    def _get_openrouter_client(self):
        """Lazy initialization of OpenRouter client."""
        if self._openrouter_client is None:
            if not self._openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not configured")
            self._openrouter_client = OpenAI(
                api_key=self._openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._openrouter_client

    def _get_client_for_model(self, model: str):
        """Get the appropriate client for a model."""
        if model in KILO_MODELS:
            return self.client  # Kilo client
        elif model in OPENROUTER_FREE_MODELS:
            return self._get_openrouter_client()
        # Default to Kilo for unknown models
        return self.client

    def complete(
        self,
        messages: List[LLMMessage],
        tools: List = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto",
    ) -> "LLMResponse":
        """Call LLM via Kilo API with OpenRouter fallback."""
        from src.llm.types import LLMResponse

        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [tool.to_dict() for tool in tools]

        # Try Kilo models first, then OpenRouter
        models_to_try = [self._model]
        
        # Add other Kilo models if primary is Kilo
        if self._model in KILO_MODELS:
            models_to_try += [m for m in KILO_MODELS if m != self._model]
        
        # Add OpenRouter models
        if self._openrouter_api_key:
            models_to_try += [m for m in OPENROUTER_FREE_MODELS if m not in models_to_try]

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
    kilo_api_key: str = None,
    openrouter_api_key: str = None,
) -> "KiloClient":
    """Factory function to create Kilo client with OpenRouter fallback."""
    from src.llm.types import LLMConfig

    config = LLMConfig(
        model=model or os.getenv("KILO_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        base_url=base_url or os.getenv("KILO_BASE_URL", "https://api.kilo.ai/api/gateway"),
        api_key=api_key or os.getenv("KILO_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
        kilo_api_key=kilo_api_key,
        openrouter_api_key=openrouter_api_key,
    )

    if not config.api_key:
        raise ValueError("KILO_API_KEY or OPENROUTER_API_KEY not set")

    return KiloClient(config)