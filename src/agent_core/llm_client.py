"""LLM Client wrapper for OpenRouter API."""

import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from opentelemetry.trace import Status, StatusCode

from src.observability import get_logger, get_metrics, get_tracer, trace_llm_call

logger = get_logger(__name__)


class LLMClient:
    """Wrapper for OpenRouter/OpenAI LLM client with observability."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.3,
        tool_choice: str = "auto",
    ):
        """Call LLM with messages and optional tools."""
        start_time = time.time()

        with trace_llm_call(self.model) as span:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                )

                duration = time.time() - start_time

                # Record metrics
                from src.observability import get_metrics
                metrics = get_metrics()
                metrics.record_llm_call(
                    model=self.model,
                    success=True,
                    duration=duration,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                )
                span = get_tracer().start_span("llm_call")
                span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens if response.usage else 0)
                span.set_attribute("llm.completion_tokens", response.usage.completion_tokens if response.usage else 0)
                span.end()

                return response

            except Exception as e:
                logger.error("llm_call_failed", model=self.model, error=str(e))
                raise RuntimeError(f"LLM API 呼叫失敗: {e}")

    def get_model_name(self) -> str:
        return self.model