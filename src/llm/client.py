"""Kilo AI LLM client implementation - using only specified Kilo models."""

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


# 用戶指定的 Kilo 模型（優先序：越前面越優先使用）
# 註：模型 ID 必須與 Kilo API 實際提供的完全一致
# 只有帶 :free 後綴或 kilo-auto/free 的模型才是真正免費的
KILO_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",      # 原預設模型（免費，但有時過載）
    "kilo-auto/free",                                # Kilo 自動選擇免費模型
    "liquid/lfm-2.5-2.6b:free",                     # 免費模型
    "nvidia/nemotron-3.5-lightning:free",           # 免費模型
    "dots-studio/dots-3-note-preview:free",         # 免費模型
    "nvidia/nemotron-3-super-120b-a12b:free",       # 免費模型
    "poolside/laguna-s-2.1:free",                   # 免費模型
    "poolside/laguna-xs-2.1:free",                  # 免費模型
    "cohere/north-mini-code:free",                  # 免費模型
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", # 免費模型
]


class KiloClient(LLMClient):
    """Kilo AI LLM client using OpenAI SDK - only uses specified Kilo models."""

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
        # OpenRouter API key is ignored - we only use Kilo models

    def _get_client_for_model(self, model: str):
        """Get the appropriate client for a model - all models use Kilo client."""
        return self.client

    def complete(
        self,
        messages: List[LLMMessage],
        tools: List = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto",
    ) -> "LLMResponse":
        """Call LLM via Kilo API with fallback to other Kilo models only.
        
        Priority:
        1. Primary model (from config)
        2. Other Kilo models (from KILO_MODELS list)
        """
        from src.llm.types import LLMResponse

        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [tool.to_dict() for tool in tools]

        # Build model fallback list with priority
        models_to_try = [self._model]
        
        # Add other Kilo models as fallback
        models_to_try += [m for m in KILO_MODELS if m != self._model]

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
    timeout: float = 60.0,
    max_retries: int = 2,
    temperature: float = 0.3,
    max_tokens: int = None,
    kilo_api_key: str = None,
) -> "KiloClient":
    """Factory function to create Kilo client with specified models only."""
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
    )

    if not config.api_key:
        raise ValueError("KILO_API_KEY not set")

    return KiloClient(config)