"""LLM Client wrapper for OpenRouter API with multi-model fallback."""

import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai import RateLimitError, APIError
from opentelemetry.trace import Status, StatusCode

from src.observability import get_logger, get_metrics, trace_llm_call

logger = get_logger(__name__)


# 預設免費模型清單（優先序：越前面越優先使用）
DEFAULT_FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",      # 原預設模型
    "poolside/laguna-s-2.1:free",                   # 程式碼生成強
    "nvidia/nemotron-3.5-lightning:free",           # 輕量快速
    "dots-studio/dots-3-note-preview:free",         # 實驗性
    "nvidia/nemotron-3-super-120b-a12b:free",       # 大模型
    "thinkingmachines/inkling:free",                # 思考型
    "apodex/apodex-1.1-mini:free",                  # 輕量
]


class ModelExhaustedError(Exception):
    """當所有模型都用盡額度時拋出"""
    def __init__(self, tried_models: List[str]):
        self.tried_models = tried_models
        super().__init__(f"All models exhausted: {tried_models}")


class LLMClient:
    """Wrapper for OpenRouter/OpenAI LLM client with multi-model fallback."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        fallback_models: Optional[List[str]] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._primary_model = model
        self._fallback_models = fallback_models or DEFAULT_FREE_MODELS
        # 確保 primary model 在 fallback 清單最前面
        self._all_models = [model] + [m for m in self._fallback_models if m != model]
        self.timeout = timeout

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        self._current_model_idx = 0
        self._model = self._all_models[0]  # 目前使用的模型

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value
        if value in self._all_models:
            self._current_model_idx = self._all_models.index(value)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.3,
        tool_choice: str = "auto",
    ):
        """Call LLM with automatic fallback on rate limit / model unavailable."""
        last_error = None
        tried_models = []

        for idx in range(self._current_model_idx, len(self._all_models)):
            model = self._all_models[idx]
            tried_models.append(model)

            try:
                return self._call_single_model(model, messages, tools, temperature, tool_choice)
            except RateLimitError as e:
                logger.warning("model_rate_limited", model=model, error=str(e))
                last_error = e
                continue
            except APIError as e:
                # 404 = model not available, 403 = key limit exceeded
                if e.status_code in (403, 404):
                    logger.warning("model_unavailable", model=model, status=e.status_code)
                    last_error = e
                    continue
                # 其他 API 錯誤直接拋出
                raise
            except Exception as e:
                logger.error("llm_call_failed", model=model, error=str(e))
                last_error = e
                # 非率限錯誤，嘗試下一個模型（或可選擇直接拋出）
                continue

        # 所有模型都失敗
        self._current_model_idx = 0  # 重置
        raise ModelExhaustedError(tried_models)

    def _call_single_model(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.3,
        tool_choice: str = "auto",
    ):
        """呼叫單一模型，成功時更新 current model"""
        start_time = time.time()

        with trace_llm_call(model) as span:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
            )

            duration = time.time() - start_time

            span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens if response.usage else 0)
            span.set_attribute("llm.completion_tokens", response.usage.completion_tokens if response.usage else 0)

            # 成功：更新當前模型為此模型（下次優先使用）
            self._model = model
            self._current_model_idx = self._all_models.index(model)

            return response

    def get_model_name(self) -> str:
        return self.model

    def get_available_models(self) -> List[str]:
        """取得所有可用模型清單"""
        return self._all_models.copy()

    def reset_to_primary(self):
        """重置回主要模型"""
        self._current_model_idx = 0
        self._model = self._all_models[0]

    @property
    def primary_model(self) -> str:
        return self._primary_model

    @property
    def fallback_models(self) -> List[str]:
        return self._fallback_models.copy()