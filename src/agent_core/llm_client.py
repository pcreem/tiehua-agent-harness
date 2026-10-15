"""LLM Client wrapper for Kilo AI API with multi-model fallback (Kilo models only)."""

import os
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai import RateLimitError, APIError
from opentelemetry.trace import Status, StatusCode

from src.observability import get_logger, get_metrics, trace_llm_call

logger = get_logger(__name__)


# Kilo 模型清單（優先序：越前面越優先使用）
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


class ModelExhaustedError(Exception):
    """當所有模型都用盡額度時拋出"""
    def __init__(self, tried_models: List[str]):
        self.tried_models = tried_models
        super().__init__(f"All models exhausted: {tried_models}")


class LLMClient:
    """Wrapper for Kilo LLM client with multi-model fallback (Kilo models only)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.kilo.ai/api/gateway",  # Kilo AI gateway as default
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        fallback_models: Optional[List[str]] = None,
        timeout: float = 60.0,
        kilo_api_key: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._primary_model = model
        self._kilo_fallback = fallback_models or KILO_MODELS
        self.kilo_api_key = kilo_api_key or os.getenv("KILO_API_KEY") or api_key
        self.timeout = timeout

        # Kilo client (only client needed)
        self.client = OpenAI(api_key=self.kilo_api_key, base_url=base_url, timeout=timeout)

        # 建立完整的模型鏈：只使用 Kilo models
        self._all_models = [model] + [m for m in self._kilo_fallback if m != model]
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

    def _get_client_for_model(self, model: str):
        """Get the client for a model - all models use Kilo client."""
        return self.client

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
        self._model = self._all_models[0]
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

        # 取得正確的 client (所有模型都用 Kilo client)
        client = self._get_client_for_model(model)

        with trace_llm_call(model) as span:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
            )

            duration = time.time() - start_time

            span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens if response.usage else 0)
            span.set_attribute("llm.completion_tokens", response.usage.completion_tokens if response.usage else 0)
            span.set_attribute("llm.provider", "kilo")

            # 成功：更新當前模型為此模型（下次優先使用）
            self._model = model
            self._current_model_idx = self._all_models.index(model)

            return response

    def get_model_name(self) -> str:
        return self.model

    def get_available_models(self) -> List[str]:
        """取得所有可用模型清單"""
        return self._all_models.copy()

    def get_kilo_models(self) -> List[str]:
        """取得 Kilo 模型清單"""
        return [m for m in self._all_models if m in KILO_MODELS]

    def reset_to_primary(self):
        """重置回主要模型"""
        self._current_model_idx = 0
        self._model = self._all_models[0]

    @property
    def primary_model(self) -> str:
        return self._primary_model

    @property
    def kilo_fallback_models(self) -> List[str]:
        return self._kilo_fallback.copy()