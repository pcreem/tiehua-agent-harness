"""Safety guardrails pipeline - orchestrates all safety checks."""

import time
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .types import (
    SafetyCheckResult,
    SafetyViolation,
    SafetyViolationType,
    SafetyAction,
)
from .prompt_injection import PromptInjectionDetector, create_prompt_injection_detector
from .pii import PIIDetector, create_pii_detector
from .content_filter import ContentFilter, create_content_filter
from .tool_guard import ToolGuard, create_tool_guard, DEFAULT_TOOL_RULES, ToolParameterRule


class PipelineMode(str, Enum):
    """Pipeline execution mode."""
    STRICT = "strict"      # Block on any violation
    PERMISSIVE = "permissive"  # Warn but allow
    CUSTOM = "custom"      # Use custom decision function


@dataclass
class PipelineConfig:
    """Configuration for the safety pipeline."""
    mode: PipelineMode = PipelineMode.STRICT
    enable_prompt_injection: bool = True
    enable_pii_detection: bool = True
    enable_content_filter: bool = True
    enable_tool_guard: bool = True
    pii_redact: bool = True
    custom_decision_fn: Optional[Callable[['PipelineResult'], SafetyAction]] = None
    max_latency_ms: int = 1000  # Max latency budget for pipeline


@dataclass
class PipelineResult:
    """Result of the entire safety pipeline."""
    passed: bool
    action: SafetyAction
    violations: List = field(default_factory=list)
    sanitized_input: Optional[str] = None
    sanitized_tool_args: Optional[Dict] = None
    latency_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_violations(self, violations: List) -> None:
        self.violations.extend(violations)
        if violations:
            self.passed = False

    def has_blocking_violations(self) -> bool:
        return any(v.suggested_action == SafetyAction.BLOCK for v in self.violations)


class GuardrailsPipeline:
    """Main safety pipeline orchestrating all safety checks."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        prompt_injection_detector=None,
        pii_detector=None,
        content_filter=None,
        tool_guard=None,
    ):
        self.config = config or PipelineConfig()
        self.prompt_injection_detector = prompt_injection_detector or create_prompt_injection_detector()
        self.pii_detector = pii_detector or create_pii_detector()
        self.content_filter = content_filter or create_content_filter(use_detoxify=False)
        self.tool_guard = tool_guard or create_tool_guard()

        # Metrics
        self.total_checks = 0
        self.total_blocked = 0
        self.total_latency_ms = 0.0

    def check_input(self, text: str) -> 'PipelineResult':
        """Check user input for safety violations."""
        start_time = time.time()
        result = PipelineResult(passed=True, action=SafetyAction.ALLOW)

        # 1. Prompt Injection Detection
        if self.config.enable_prompt_injection:
            pi_result = self.prompt_injection_detector.check(text, location="input")
            result.add_violations(pi_result.violations)
            if pi_result.violations:
                result.metadata["prompt_injection"] = {
                    "violations": len(pi_result.violations),
                    "blocked": len(pi_result.get_blocking_violations()) > 0,
                }

        # 2. PII Detection
        if self.config.enable_pii_detection:
            pii_result = self.pii_detector.check(text, location="input")
            result.add_violations(pii_result.violations)
            result.metadata["pii"] = {
                "violations": len(pii_result.violations),
                "blocked": len(pii_result.get_blocking_violations()) > 0,
            }

            # Redact PII if enabled
            if self.config.pii_redact and pii_result.violations:
                result.sanitized_input = self.pii_detector.redact(text)

        # 3. Content Filter
        if self.config.enable_content_filter:
            cf_result = self.content_filter.check(text, location="input")
            result.add_violations(cf_result.violations)
            result.metadata["content_filter"] = {
                "violations": len(cf_result.violations),
                "blocked": len(cf_result.get_blocking_violations()) > 0,
            }

        # Determine final action
        result.latency_ms = (time.time() - start_time) * 1000
        result.action = self._decide_action(result)
        self._update_metrics(result)

        return result

    def check_output(self, text: str) -> 'PipelineResult':
        """Check LLM output for safety violations."""
        start_time = time.time()
        result = PipelineResult(passed=True, action=SafetyAction.ALLOW)

        # Content filter on output
        if self.config.enable_content_filter:
            cf_result = self.content_filter.check(text, location="output")
            result.add_violations(cf_result.violations)

        # PII in output
        if self.config.enable_pii_detection:
            pii_result = self.pii_detector.check(text, location="output")
            result.add_violations(pii_result.violations)

        result.latency_ms = (time.time() - start_time) * 1000
        result.action = self._decide_action(result)
        self._update_metrics(result)

        return result

    def check_tool_args(self, tool_name: str, arguments: Dict[str, Any]) -> 'PipelineResult':
        """Check tool arguments for security violations."""
        start_time = time.time()
        result = PipelineResult(passed=True, action=SafetyAction.ALLOW)

        if self.config.enable_tool_guard:
            tg_result = self.tool_guard.check(tool_name, arguments, location="tool_args")
            result.add_violations(tg_result.violations)
            result.metadata["tool_guard"] = {
                "tool": tool_name,
                "violations": len(tg_result.violations),
                "blocked": len(tg_result.get_blocking_violations()) > 0,
            }

            # Sanitize if violations
            if tg_result.violations:
                result.sanitized_tool_args = self.tool_guard.sanitize(tool_name, arguments)

        result.latency_ms = (time.time() - start_time) * 1000
        result.action = self._decide_action(result)
        self._update_metrics(result)

        return result

    def check_tool_result(self, tool_name: str, result_data: Dict[str, Any]) -> 'PipelineResult':
        """Check tool execution result for safety issues."""
        start_time = time.time()
        result = PipelineResult(passed=True, action=SafetyAction.ALLOW)

        # Convert result to text for checking
        import json
        result_text = json.dumps(result_data, ensure_ascii=False)

        # Content filter on tool result
        if self.config.enable_content_filter:
            cf_result = self.content_filter.check(result_text, location=f"tool_result:{tool_name}")
            result.add_violations(cf_result.violations)

        # PII in tool result
        if self.config.enable_pii_detection:
            pii_result = self.pii_detector.check(result_text, location=f"tool_result:{tool_name}")
            result.add_violations(pii_result.violations)

        result.latency_ms = (time.time() - start_time) * 1000
        result.action = self._decide_action(result)
        self._update_metrics(result)

        return result

    def _decide_action(self, result: PipelineResult) -> SafetyAction:
        """Decide final action based on violations and mode."""
        if result.passed:
            return SafetyAction.ALLOW

        if self.config.mode == PipelineMode.STRICT:
            if result.has_blocking_violations():
                return SafetyAction.BLOCK
            return SafetyAction.WARN

        elif self.config.mode == PipelineMode.PERMISSIVE:
            return SafetyAction.WARN

        elif self.config.mode == PipelineMode.CUSTOM and self.config.custom_decision_fn:
            return self.config.custom_decision_fn(result)

        return SafetyAction.WARN

    def _update_metrics(self, result: PipelineResult) -> None:
        """Update pipeline metrics."""
        self.total_checks += 1
        self.total_latency_ms += result.latency_ms
        if result.action == SafetyAction.BLOCK:
            self.total_blocked += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_checks": self.total_checks,
            "total_blocked": self.total_blocked,
            "block_rate": self.total_blocked / max(self.total_checks, 1),
            "avg_latency_ms": self.total_latency_ms / max(self.total_checks, 1),
        }

    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self.total_checks = 0
        self.total_blocked = 0
        self.total_latency_ms = 0.0


def create_guardrails_pipeline(
    mode: str = "strict",
    enable_prompt_injection: bool = True,
    enable_pii_detection: bool = True,
    enable_content_filter: bool = True,
    enable_tool_guard: bool = True,
    pii_redact: bool = True,
    **kwargs,
) -> GuardrailsPipeline:
    """Factory function to create guardrails pipeline."""
    config = PipelineConfig(
        mode=PipelineMode(mode),
        enable_prompt_injection=enable_prompt_injection,
        enable_pii_detection=enable_pii_detection,
        enable_content_filter=enable_content_filter,
        enable_tool_guard=enable_tool_guard,
        pii_redact=pii_redact,
        **kwargs,
    )
    return GuardrailsPipeline(config=config)


# Default pipeline instance
_default_pipeline: Optional['GuardrailsPipeline'] = None


def get_guardrails_pipeline() -> GuardrailsPipeline:
    """Get default guardrails pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = create_guardrails_pipeline()
    return _default_pipeline