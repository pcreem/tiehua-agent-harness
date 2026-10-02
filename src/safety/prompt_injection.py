"""Prompt injection detection using llm-guard and custom rules."""

import re
from typing import List, Optional

from .types import SafetyCheckResult, SafetyViolation, SafetyViolationType, SafetyAction


class PromptInjectionDetector:
    """Detect prompt injection attacks using llm-guard and custom patterns."""

    # Common injection patterns
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(previous|prior|above)\s+instructions?",
        r"(?i)forget\s+(previous|prior|above)\s+instructions?",
        r"(?i)disregard\s+(previous|prior|above)\s+instructions?",
        r"(?i)you\s+are\s+now\s+(a|an)\s+\w+",
        r"(?i)act\s+as\s+(a|an)\s+\w+",
        r"(?i)pretend\s+(to\s+be\s+)?(a|an)\s+\w+",
        r"(?i)roleplay\s+(as\s+)?(a|an)\s+\w+",
        r"(?i)system\s*:\s*",
        r"(?i)assistant\s*:\s*",
        r"(?i)user\s*:\s*",
        r"(?i)###\s*(instruction|system|user|assistant)",
        r"(?i)<\|im_start\|>",
        r"(?i)<\|im_end\|>",
        r"(?i)ignore\s+all\s+(previous|prior)\s+(rules|instructions|constraints)",
        r"(?i)new\s+(rules|instructions|constraints)\s*:",
        r"(?i)from\s+now\s+on\s*,?\s*you\s+(will|must|should)",
        r"(?i)override\s+(previous|prior)\s+(rules|instructions)",
        r"(?i)bypass\s+(safety|security|filter|guard)",
        r"(?i)disable\s+(safety|security|filter|guard)",
        r"(?i)jailbreak",
        r"(?i)dan\s+mode",
        r"(?i)developer\s+mode",
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        use_llm_guard: bool = True,
        custom_patterns: Optional[List[str]] = None,
    ):
        self.threshold = threshold
        self.use_llm_guard = use_llm_guard and self._check_llm_guard_available()
        self.patterns = self.INJECTION_PATTERNS + (custom_patterns or [])
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]

        # Initialize llm-guard scanner
        if self.use_llm_guard:
            try:
                from llm_guard.input_scanners import PromptInjection
                from llm_guard.input_scanners.prompt_injection import MatchType
                self.scanner = PromptInjection(threshold=threshold, match_type=MatchType.FULL)
            except ImportError:
                self.use_llm_guard = False
                self.scanner = None
        else:
            self.scanner = None

    def _check_llm_guard_available(self) -> bool:
        try:
            import llm_guard
            return True
        except ImportError:
            return False

    def check(self, text: str, location: str = "input") -> SafetyCheckResult:
        """Check text for prompt injection."""
        result = SafetyCheckResult(passed=True)

        # Regex-based detection
        for pattern in self.compiled_patterns:
            matches = list(pattern.finditer(text))
            if matches:
                for match in matches:
                    violation = SafetyViolation(
                        violation_type=SafetyViolationType.PROMPT_INJECTION,
                        severity="high",
                        message=f"Prompt injection pattern detected: {match.group()}",
                        location=location,
                        details={
                            "pattern": match.re.pattern,
                            "matched_text": match.group(),
                            "position": match.span(),
                        },
                        suggested_action=SafetyAction.BLOCK,
                        confidence=0.9,
                    )
                    result.add_violation(violation)

        # llm-guard detection
        if self.scanner:
            sanitized, is_valid, risk_score = self.scanner.scan(text)
            if not is_valid:
                violation = SafetyViolation(
                    violation_type=SafetyViolationType.PROMPT_INJECTION,
                    severity="high" if risk_score > 0.7 else "medium",
                    message=f"LLM-Guard detected prompt injection (risk: {risk_score:.2f})",
                    location=location,
                    details={
                        "risk_score": risk_score,
                        "sanitized": sanitized,
                    },
                    suggested_action=SafetyAction.BLOCK,
                    confidence=risk_score,
                )
                result.add_violation(violation)

        return result

    def sanitize(self, text: str) -> str:
        """Sanitize text by removing injection patterns."""
        if self.scanner:
            sanitized, _, _ = self.scanner.scan(text)
            return sanitized
        return text


def create_prompt_injection_detector(**kwargs) -> PromptInjectionDetector:
    """Factory function to create prompt injection detector."""
    return PromptInjectionDetector(**kwargs)