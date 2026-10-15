"""LLM-as-a-Judge integration for Agent Harness."""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.evaluation.types import EvaluationResult, ExpectedBehavior, EvaluatorStatus
from src.state import AgentState

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from LLM Judge."""

    passed: bool
    score: float
    issues: List[Dict[str, Any]]
    retry: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": self.issues,
            "retry": self.retry,
            "reason": self.reason,
        }


class LLMJudge:
    """LLM-as-a-Judge evaluator using free models."""

    # Free models available on OpenRouter
    OPENROUTER_MODELS = {
        "primary": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "fallback": "nvidia/nemotron-3.5-lightning:free",
        "fallback2": "dots-studio/dots-3-note-preview:free",
        "fallback3": "nvidia/nemotron-3-super-120b-a12b:free",
        "fallback4": "cohere/north-mini-code:free",
        "fallback5": "thinkingmachines/inkling-small:free",
    }

    # Mistral 模型（需要 MISTRAL_API_KEY）
    MISTRAL_MODELS = [
        "mistral-small-latest",
        "mistral-medium-latest",
        "magistral-small-latest",
        "magistral-medium-latest",
    ]

    OPENROUTER_FALLBACK_CHAIN = [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3.5-lightning:free",
        "dots-studio/dots-3-note-preview:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "cohere/north-mini-code:free",
        "thinkingmachines/inkling-small:free",
    ]

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        fallback_chain: List[str] = None,
        mistral_api_key: str = None,
    ):
        self.model = model or self.OPENROUTER_MODELS["primary"]
        self.fallback_model = self.OPENROUTER_MODELS["fallback"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        self._client = None
        self._mistral_client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client for OpenRouter."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _get_mistral_client(self):
        """Lazy initialization of Mistral client."""
        if self._mistral_client is None:
            if not self.mistral_api_key:
                raise ValueError("MISTRAL_API_KEY not configured")
            from openai import OpenAI
            self._mistral_client = OpenAI(
                api_key=self.mistral_api_key,
                base_url="https://api.mistral.ai/v1",
                timeout=self.timeout,
            )
        return self._mistral_client

    def _get_client_for_model(self, model: str):
        """Get the appropriate client for a model."""
        if model in self.MISTRAL_MODELS:
            return self._get_mistral_client()
        return self.client

    def _load_prompt(self) -> str:
        """Load judge prompt from file."""
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "prompts", "judge.md"
        )
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return self._default_prompt()

    def _default_prompt(self) -> str:
        """Default judge prompt."""
        return """# Role

You are the quality judge for Taiwan Tourism Agent Harness.

Your task is to evaluate whether an agent execution satisfies the
provided expected behavior and whether the final answer is grounded
in the available tool results.

# Rules

1. Evaluate only the provided execution evidence.
2. Do not invent external facts.
3. Distinguish missing evidence from incorrect evidence.
4. A response is not grounded when it makes factual claims that are
   unsupported by the supplied tool results.
5. Follow the expected behavior specification.
6. Return structured JSON only.

# Input

## User Query
{{user_query}}

## Expected Behavior
{{expected_behavior}}

## Tool Calls
{{tool_calls}}

## Tool Results
{{tool_results}}

## Agent Response
{{agent_response}}

# Output Schema

{
  "passed": true,
  "score": 0.0,
  "issues": [
    {
      "code": "string",
      "severity": "info|warning|error",
      "message": "string"
    }
  ],
  "retry": false,
  "reason": "string"
}

# Evaluation Criteria

## Task Correctness (0-1)
- Does the response answer the user's actual question?
- Is the response relevant to the query?

## Tool Accuracy (0-1)
- Were the correct tools called?
- Were tool arguments appropriate?
- Were all required tools called?
- Were no forbidden tools called?

## Groundedness (0-1)
- Are all factual claims supported by tool results?
- Does the answer reference specific tool result data?
- Are there unsupported claims?

## Completeness (0-1)
- Does the answer cover all required aspects?
- Are required facts present?
- Are required conditions addressed?

## Safety (0-1)
- No hallucinated information
- No dangerous recommendations
- No PII leakage

# Scoring Guidelines

- 1.0: Perfect - all criteria met, fully grounded
- 0.8-0.9: Good - minor gaps, mostly grounded
- 0.6-0.7: Acceptable - some gaps, mostly correct
- 0.4-0.5: Poor - significant gaps, partially grounded
- 0.0-0.3: Failed - major issues, ungrounded or wrong

# Issue Codes

- MISSING_TOOL: Required tool not called
- WRONG_TOOL: Incorrect tool called
- INVALID_ARGS: Tool called with wrong arguments
- UNGROUNDED_CLAIM: Factual claim without tool support
- MISSING_EVIDENCE: Required evidence not provided
- INCOMPLETE_ANSWER: Answer missing required information
- HALLUCINATION: Fabricated information
- SAFETY_VIOLATION: Safety concern

# Severity Levels

- info: Minor observation, no action needed
- warning: Potential issue, may affect quality
- error: Significant issue, affects correctness
- critical: Critical failure, must fix"""

    def _format_tool_calls(self, tool_calls: List[Dict]) -> str:
        """Format tool calls for prompt."""
        if not tool_calls:
            return "No tool calls made."
        lines = []
        for tc in tool_calls:
            lines.append(f"- {tc['name']}({tc.get('arguments', {})})")
        return "\n".join(lines)

    def _format_tool_results(self, tool_results: List[Dict]) -> str:
        """Format tool results for prompt."""
        if not tool_results:
            return "No tool results."
        lines = []
        for tr in tool_results:
            result = tr.get('result', {})
            success = result.get('success', False)
            lines.append(f"- {tr['name']}: {'SUCCESS' if success else 'FAILED'} - {result}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        query: str,
        state,
        expected: Optional[Any] = None,
    ) -> str:
        """Build the evaluation prompt."""
        prompt_template = self._load_prompt()
        
        # Format expected behavior
        expected_str = "None specified"
        if expected:
            parts = []
            if expected.required_tools:
                parts.append(f"Required tools: {', '.join(t['name'] for t in expected.required_tools)}")
            if expected.forbidden_tools:
                parts.append(f"Forbidden tools: {', '.join(expected.forbidden_tools)}")
            if expected.required_facts:
                parts.append(f"Required facts: {', '.join(expected.required_facts)}")
            if expected.required_conditions:
                parts.append(f"Required conditions: {', '.join(expected.required_conditions)}")
            if expected.must_be_grounded:
                parts.append("Must be grounded in tool results")
            if expected.expected_outcome:
                parts.append(f"Expected outcome: {expected.expected_outcome}")
            if expected.category:
                parts.append(f"Category: {expected.category}")
            expected_str = "\n".join(parts)
        
        # Format tool calls
        tool_calls_str = self._format_tool_calls(state.tool_calls)
        
        # Format tool results
        tool_results_str = self._format_tool_results(state.tool_results)
        
        # Agent response
        agent_response = state.answer or ""
        
        return prompt_template.format(
            user_query=query,
            expected_behavior=expected_str,
            tool_calls=tool_calls_str,
            tool_results=tool_results_str,
            agent_response=agent_response,
        )

    def _call_llm(self, prompt: str, model: str = None) -> Optional[Dict]:
        """Call LLM with prompt, with fallback chain including Mistral."""
        models_to_try = [model or self.model] if model else self.OPENROUTER_FALLBACK_CHAIN
        # Add Mistral models as additional fallback if API key is available
        if self.mistral_api_key:
            models_to_try += self.MISTRAL_MODELS
        
        last_error = None
        for m in models_to_try:
            try:
                client = self._get_client_for_model(m)
                response = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                
                if not response.choices or not response.choices[0].message:
                    raise RuntimeError(f"LLM 回傳異常 (無 choices 或 message): {response}")
                
                message = response.choices[0].message
                content = message.content or "{}"
                
                try:
                    result = json.loads(content)
                    result["_model_used"] = m
                    return result
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"LLM 回傳非 JSON 格式: {e}")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Model {m} failed: {e}, trying next...")
                continue
        
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def evaluate(
        self,
        query: str,
        state,
        expected: Optional[Any] = None,
    ) -> JudgeResult:
        """Evaluate using LLM Judge."""
        try:
            prompt = self._build_prompt(query, state, expected)
            response = self._call_llm(prompt)
            
            if not response:
                return JudgeResult(
                    passed=False,
                    score=0.0,
                    issues=[{
                        "code": "JUDGE_NO_RESPONSE",
                        "severity": "error",
                        "message": "LLM Judge 回傳空回應"
                    }],
                    retry=True,
                    reason="LLM Judge 回傳空回應"
                )
            
            # Parse response
            passed = response.get("passed", False)
            score = float(response.get("score", 0.0))
            issues = response.get("issues", [])
            retry = response.get("retry", False)
            reason = response.get("reason", "")
            
            return JudgeResult(
                passed=passed,
                score=score,
                issues=issues,
                retry=retry,
                reason=reason,
            )
            
        except Exception as e:
            return JudgeResult(
                passed=False,
                score=0.0,
                issues=[{
                    "code": "JUDGE_ERROR",
                    "severity": "error",
                    "message": f"LLM Judge 執行失敗: {str(e)}"
                }],
                retry=True,
                reason=f"LLM Judge 錯誤: {str(e)}"
            )


class LLMJudgeEvaluator:
    """LLM Judge as an evaluator plugin."""

    name = "llm_judge"
    description = "LLM-as-a-Judge for deep semantic evaluation"

    def __init__(self, judge: Optional[LLMJudge] = None):
        self.judge = judge or LLMJudge()

    def evaluate(
        self,
        query: str,
        state: AgentState,
        expected: Optional[Any] = None,
    ) -> EvaluationResult:
        """Evaluate using LLM Judge."""
        judge_result = self.judge.evaluate(query, state, expected)

        from src.evaluation.types import EvaluationResult, EvaluatorStatus, CriterionResult, EvaluationCriterion

        if judge_result.passed:
            return EvaluationResult(
                status=EvaluatorStatus.PASS,
                score=judge_result.score,
                criteria_results=[],
                feedback=[],
                retry=False,
                metadata={"evaluator": "llm_judge", "judge_model": self.judge.model},
            )
        else:
            feedback = [issue["message"] for issue in judge_result.issues]
            return EvaluationResult(
                status=EvaluatorStatus.FAIL,
                score=judge_result.score,
                criteria_results=[],
                feedback=feedback,
                retry=judge_result.retry,
                metadata={
                    "evaluator": "llm_judge",
                    "judge_model": self.judge.model,
                    "judge_reason": judge_result.reason,
                },
            )


# For backward compatibility
def create_llm_judge_evaluator() -> LLMJudgeEvaluator:
    """Create LLM Judge evaluator."""
    return LLMJudgeEvaluator()