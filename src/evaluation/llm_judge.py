"""LLM-as-a-Judge integration for Agent Harness."""

import json
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.evaluation.types import EvaluationResult, ExpectedBehavior, EvaluatorStatus
from src.harness.state import AgentState


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
    MODELS = {
        "primary": "poolside/laguna-s-2.1:free",
        "fallback": "dots-studio/dots-3-note-preview:free",
        "reuse_agent": "nvidia/nemotron-3-ultra-550b-a55b:free",
    }

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        self.model = model or self.MODELS["primary"]
        self.fallback_model = self.MODELS["fallback"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

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

You are the quality judge for Tiehua Tourism Agent Harness.

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
}"""

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

    def evaluate(
        self,
        query: str,
        state: AgentState,
        expected: Optional[ExpectedBehavior] = None,
    ) -> JudgeResult:
        """Evaluate using LLM Judge (requires LLM client)."""
        # This is a stub - requires actual LLM client integration
        # For now, return a result indicating judge not available
        return JudgeResult(
            passed=False,
            score=0.5,
            issues=[{
                "code": "JUDGE_UNAVAILABLE",
                "severity": "warning",
                "message": "LLM Judge not configured - requires OpenRouter API key and client setup"
            }],
            retry=False,
            reason="LLM Judge not configured"
        )

    def _call_llm(self, prompt: str, model: str = None) -> Optional[Dict]:
        """Call LLM with prompt (stub)."""
        # This would integrate with OpenRouter client
        # For now, return None to indicate not implemented
        return None


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
        expected: Optional[ExpectedBehavior] = None,
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