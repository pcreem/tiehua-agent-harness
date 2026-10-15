"""LLM Client wrapper for OpenRouter API with multi-model fallback (including Mistral)."""

import os
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai import RateLimitError, APIError
from opentelemetry.trace import Status, StatusCode

from src.observability import get_logger, get_metrics, trace_llm_call

logger = get_logger(__name__)


# OpenRouter 免費模型清單（優先序：越前面越優先使用）
OPENROUTER_FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",      # 原預設模型
    "poolside/laguna-s-2.1:free",                   # 程式碼生成強
    "nvidia/nemotron-3.5-lightning:free",           # 輕量快速
    "dots-studio/dots-3-note-preview:free",         # 實驗性
    "nvidia/nemotron-3-super-120b-a12b:free",       # 大模型
    "thinkingmachines/inkling:free",                # 思考型
    "apodex/apodex-1.1-mini:free",                  # 輕量
]

# Mistral 模型清單（需要 MISTRAL_API_KEY）
MISTRAL_MODELS = [
    "mistral-moderation-2603",   # 內容審核模型，較快
    "mistral-medium-latest",      # 中等規模模型，平衡速度與品質
    "mistral-large-latest",       # 大模型，較強推理能力
    "pixtral-large-latest",       # 多模態模型
]


class ModelExhaustedError(Exception):
    """當所有模型都用盡額度時拋出"""
    def __init__(self, tried_models: List[str]):
        self.tried_models = tried_models
        super().__init__(f"All models exhausted: {tried_models}")


class LLMClient:
    """Wrapper for OpenRouter/OpenAI LLM client with multi-model fallback across providers."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        fallback_models: Optional[List[str]] = None,
        timeout: float = 30.0,
        mistral_api_key: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._primary_model = model
        self._openrouter_fallback = fallback_models or OPENROUTER_FREE_MODELS
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        self.timeout = timeout

        # OpenRouter client
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        
        # Mistral client (lazy init)
        self._mistral_client = None
        
        # 建立完整的模型鏈：OpenRouter models -> Mistral models
        self._all_models = [model] + [m for m in self._openrouter_fallback if m != model]
        if self.mistral_api_key:
            self._all_models += MISTRAL_MODELS
            
        self._current_model_idx = 0
        self._model = self._all_models[0]  # 目前使用的模型
        self._current_provider = "openrouter"  # 追蹤目前 provider

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value
        if value in self._all_models:
            self._current_model_idx = self._all_models.index(value)
            # 判斷 provider
            if value in MISTRAL_MODELS:
                self._current_provider = "mistral"
            else:
                self._current_provider = "openrouter"

    def _get_client_for_model(self, model: str):
        """Get the appropriate client for a model."""
        if model in MISTRAL_MODELS:
            if self._mistral_client is None:
                if not self.mistral_api_key:
                    raise ValueError("MISTRAL_API_KEY not configured")
                # Mistral uses OpenAI-compatible API
                self._mistral_client = OpenAI(
                    api_key=self.mistral_api_key,
                    base_url="https://api.mistral.ai/v1",
                    timeout=self.timeout,
                )
            return self._mistral_client
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
        self._current_provider = "openrouter" if self._all_models[0] not in MISTRAL_MODELS else "mistral"
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

        # 取得正確的 client
        client = self._get_client_for_model(model)
        provider = "mistral" if model in MISTRAL_MODELS else "openrouter"

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
            span.set_attribute("llm.provider", provider)

            # 成功：更新當前模型為此模型（下次優先使用）
            self._model = model
            self._current_model_idx = self._all_models.index(model)
            self._current_provider = provider

            return response

    def get_model_name(self) -> str:
        return self.model

    def get_available_models(self) -> List[str]:
        """取得所有可用模型清單"""
        return self._all_models.copy()

    def get_openrouter_models(self) -> List[str]:
        """取得 OpenRouter 模型清單"""
        return [m for m in self._all_models if m not in MISTRAL_MODELS]

    def get_mistral_models(self) -> List[str]:
        """取得 Mistral 模型清單"""
        return [m for m in self._all_models if m in MISTRAL_MODELS]

    def reset_to_primary(self):
        """重置回主要模型"""
        self._current_model_idx = 0
        self._model = self._all_models[0]
        self._current_provider = "openrouter" if self._all_models[0] not in MISTRAL_MODELS else "mistral"

    @property
    def primary_model(self) -> str:
        return self._primary_model

    @property
    def openrouter_fallback_models(self) -> List[str]:
        return self._openrouter_fallback.copy()

    @property
    def mistral_fallback_models(self) -> List[str]:
        return MISTRAL_MODELS.copy()

    @property
    def current_provider(self) -> str:
        return self._current_provider