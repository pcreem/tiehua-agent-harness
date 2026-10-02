"""Prompt injection detection with risk scoring and context awareness.

Architecture:
- Detection: regex + llm-guard (optional) -> signals
- Scoring: weighted signals -> risk score (0-1)
- Decision: policy-based (BLOCK/WARN/ALLOW/REVIEW)
- Context-aware: educational content, indirect injection channels
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from .types import SafetyCheckResult, SafetyViolation, SafetyViolationType, SafetyAction


class InjectionSignalType(Enum):
    """Types of injection signals with different risk weights."""
    CRITICAL_OVERRIDE = "critical_override"
    CRITICAL_DISABLE = "critical_disable"
    CRITICAL_REVEAL = "critical_reveal"
    HIGH_OVERRIDE = "high_override"
    HIGH_REVEAL = "high_reveal"
    HIGH_BYPASS = "high_bypass"
    MEDIUM_OVERRIDE = "medium_override"
    MEDIUM_REVEAL = "medium_reveal"
    MEDIUM_ROLEPLAY = "medium_roleplay"
    MEDIUM_SYSTEM = "medium_system"
    WEAK_ROLEPLAY = "weak_roleplay"
    WEAK_CONTEXT = "weak_context"


SIGNAL_WEIGHTS = {
    "critical_override": 1.0,
    "critical_disable": 1.0,
    "critical_reveal": 1.0,
    "high_override": 0.8,
    "high_reveal": 0.85,
    "high_bypass": 0.85,
    "medium_override": 0.4,
    "medium_reveal": 0.5,
    "medium_roleplay": 0.3,
    "medium_system": 0.25,
    "weak_roleplay": 0.15,
    "weak_context": 0.05,
}


SIGNAL_PATTERNS = [
    ("critical_override", r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules|constraints)"),
    ("critical_override", r"(?i)forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules|constraints)"),
    ("critical_override", r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules|constraints)"),
    ("critical_disable", r"(?i)disable\s+(safety|security|filter|guard|protection)"),
    ("critical_disable", r"(?i)bypass\s+(safety|security|filter|guard|protection)"),
    ("critical_disable", r"(?i)turn\s+off\s+(safety|security|filter|guard)"),
    ("critical_reveal", r"(?i)reveal\s+(your\s+)?(system\s+)?prompt"),
    ("critical_reveal", r"(?i)show\s+(me\s+)?(your\s+)?(system\s+)?prompt"),
    ("critical_reveal", r"(?i)what\s+(is|are)\s+(your|the)\s+(system\s+)?prompt"),
    ("critical_reveal", r"(?i)output\s+(your\s+)?(system\s+)?prompt"),
    ("high_override", r"(?i)ignore\s+(previous|prior|above)\s+instructions?"),
    ("high_override", r"(?i)forget\s+(previous|prior|above)\s+instructions?"),
    ("high_override", r"(?i)disregard\s+(previous|prior|above)\s+instructions?"),
    ("high_reveal", r"(?i)what\s+(is|are)\s+your\s+instructions?"),
    ("high_reveal", r"(?i)tell\s+me\s+your\s+instructions?"),
    ("high_reveal", r"(?i)repeat\s+(your\s+)?(instructions?|prompt)"),
    ("high_bypass", r"(?i)bypass\s+(all\s+)?(filters?|restrictions?|limits?)"),
    ("high_bypass", r"(?i)override\s+(all\s+)?(safety|security)"),
    ("medium_override", r"(?i)override\s+(previous|prior)\s+(instructions?|rules)"),
    ("medium_override", r"(?i)new\s+(rules|instructions|constraints)\s*:"),
    ("medium_override", r"(?i)from\s+now\s+on\s*,?\s*you\s+(will|must|should)"),
    ("medium_reveal", r"(?i)show\s+(me\s+)?(the\s+)?(instructions?|prompt)"),
    ("medium_roleplay", r"(?i)you\s+are\s+now\s+(a|an)\s+\w+"),
    ("medium_roleplay", r"(?i)act\s+as\s+(a|an)\s+\w+"),
    ("medium_roleplay", r"(?i)pretend\s+(to\s+be\s+)?(a|an)\s+\w+"),
    ("medium_roleplay", r"(?i)roleplay\s+(as\s+)?(a|an)\s+\w+"),
    ("medium_system", r"(?i)system\s*:"),
    ("medium_system", r"(?i)assistant\s*:"),
    ("medium_system", r"(?i)user\s*:"),
    ("medium_system", r"(?i)###\s*(instruction|system|user|assistant)"),
    ("medium_system", r"(?i)<\|im_start\|>"),
    ("medium_system", r"(?i)<\|im_end\|>"),
    ("weak_roleplay", r"(?i)act\s+as\s+a\s+(translator|assistant|expert|teacher)"),
    ("weak_context", r"(?i)(what\s+is|explain|describe|define)\s+prompt\s+injection"),
    ("weak_context", r"(?i)(how\s+(does|to)\s+)(prompt\s+injection|injection)\s+(work|work\s*\?)"),
    ("weak_context", r"(?i)(example|example\s+of)\s+(prompt\s+injection|injection)"),
    ("weak_context", r"(?i)(detect|prevent|protect|defend)\s+(against\s+)?(prompt\s+injection|injection)"),
    ("weak_context", r"(?i)(learn|study|understand)\s+(prompt\s+injection|security)"),
]


EDUCATIONAL_CONTEXT_PATTERNS = [
    r"(?i)(what\s+is|explain|describe|define)\s+prompt\s+injection",
    r"(?i)how\s+(does|to)\s+(prompt\s+injection|injection)\s+(work|work\s*\?)",
    r"(?i)(example|example\s+of)\s+(prompt\s+injection|injection)",
    r"(?i)(detect|prevent|protect|defend)\s+(against\s+)?(prompt\s+injection|injection)",
    r"(?i)(learn|study|understand)\s+(prompt\s+injection|security)",
]


@dataclass
class InjectionSignal:
    signal_type: str
    matched_text: str
    pattern: str
    position: Tuple[int, int]
    weight: float
    confidence: float
    
    @property
    def signal_name(self) -> str:
        return self.signal_type


@dataclass
class InjectionDetectionResult:
    text: str
    signals: List = field(default_factory=list)
    risk_score: float = 0.0
    max_signal_weight: float = 0.0
    educational_context: bool = False
    
    @property
    def is_high_risk(self) -> bool:
        return self.risk_score >= 0.7
    
    @property
    def is_critical(self) -> bool:
        return any(s.signal_type in ('critical_override', 'critical_disable', 'critical_reveal') 
                   for s in self.signals)


class PromptInjectionDetector:
    """Prompt injection detector with risk scoring and context awareness.
    
    Detection pipeline:
    1. Normalize text (Unicode, case, whitespace)
    2. Check educational context (reduce risk if educational)
    3. Detect signals with weighted patterns
    4. Compute risk score from weighted signals
    5. Apply context adjustments (indirect injection channels)
    6. Return risk score and signals for policy decision
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        use_llm_guard: bool = True,
        custom_patterns: Optional[List[Tuple[str, str]]] = None,
    ):
        self.threshold = threshold
        self.use_llm_guard = use_llm_guard and self._check_llm_guard_available()
        
        self.patterns = SIGNAL_PATTERNS + (custom_patterns or [])
        self.compiled_patterns = [
            (signal_type, re.compile(pattern, re.IGNORECASE)) 
            for signal_type, pattern in self.patterns
        ]
        
        self.educational_patterns = [
            re.compile(p, re.IGNORECASE) for p in EDUCATIONAL_CONTEXT_PATTERNS
        ]
        
        self.scanner = None
        if self.use_llm_guard:
            try:
                from llm_guard.input_scanners import PromptInjection
                from llm_guard.input_scanners.prompt_injection import MatchType
                self.scanner = PromptInjection(threshold=0.3, match_type=MatchType.FULL)
            except ImportError:
                self.use_llm_guard = False
        
        self.channel_risk_modifiers = {
            "user_input": 1.0,
            "web_content": 1.2,
            "document": 1.1,
            "tool_result": 1.0,
            "system_generated": 0.5,
            "retrieved_context": 1.1,
        }

    def _check_llm_guard_available(self) -> bool:
        try:
            import llm_guard
            return True
        except ImportError:
            return False

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = " ".join(text.split())
        return text.lower()

    def check_educational_context(self, text: str) -> bool:
        for pattern in self.educational_patterns:
            if pattern.search(text):
                return True
        return False

    def detect_signals(self, text: str):
        signals = []
        normalized = self.normalize_text(text)
        
        for signal_type, pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                weight = 0.5
                for sig_type, pattern_str in SIGNAL_PATTERNS:
                    if sig_type == signal_type and re.search(pattern_str, match.group(), re.IGNORECASE):
                        weight = SIGNAL_WEIGHTS.get(signal_type, 0.5)
                        break
                
                signals.append(InjectionSignal(
                    signal_type=signal_type,
                    matched_text=match.group(),
                    pattern=pattern.pattern,
                    position=match.span(),
                    weight=weight,
                    confidence=min(0.9, 0.5 + weight * 0.5),
                ))
        
        return signals

    def assess_risk(self, text: str, channel: str = "user_input"):
        normalized = self.normalize_text(text)
        is_educational = self.check_educational_context(text)
        signals = self.detect_signals(text)
        
        risk_score = 0.0
        max_weight = 0.0
        
        for signal in signals:
            risk_score += signal.weight
            max_weight = max(max_weight, signal.weight)
        
        risk_score = min(1.0, risk_score)
        max_weight = max(max_weight, 0.0)
        
        if self.check_educational_context(text):
            risk_score *= 0.3
        
        channel_mod = self.channel_risk_modifiers.get(channel, 1.0)
        risk_score = min(1.0, risk_score * self.channel_risk_modifiers.get(channel, 1.0))
        
        if risk_score >= 0.9:
            action = SafetyAction.BLOCK
        elif risk_score >= 0.7:
            action = SafetyAction.REVIEW
        elif risk_score >= 0.4:
            action = SafetyAction.WARN
        else:
            action = SafetyAction.ALLOW
        
        from .types import RiskAssessment
        return RiskAssessment(
            text=text,
            channel=channel,
            normalized_text=self.normalize_text(text),
            signals=[],
            risk_score=risk_score,
            max_signal_weight=max_weight,
            educational_context=False,
            action=SafetyAction(action.value) if hasattr(action, 'value') else action,
            is_critical=False
        )
    
    def check(self, text: str, location: str = "input", channel: str = "user_input") -> SafetyCheckResult:
        assessment = self.assess_risk(text, channel)
        
        result = SafetyCheckResult(passed=assessment.action != SafetyAction.BLOCK)
        
        for signal in self.detect_signals(text):
            severity = "critical" if signal.signal_type.startswith("critical") else \
                      "high" if signal.weight >= 0.7 else "medium" if signal.weight >= 0.4 else "low"
            
            violation = SafetyViolation(
                violation_type=SafetyViolationType.PROMPT_INJECTION,
                severity=severity,
                message=f"Injection signal detected: {signal.signal_name} ({signal.matched_text})",
                location=location,
                details={
                    "signal_type": signal.signal_name,
                    "matched_text": signal.matched_text,
                    "weight": signal.weight,
                    "position": signal.position,
                    "channel": location,
                    "risk_score": sum(s.weight for s in self.detect_signals(text)),
                },
                suggested_action=SafetyAction.BLOCK if signal.weight >= 0.7 else SafetyAction.WARN,
                confidence=signal.confidence,
            )
            result.add_violation(violation)
        
        if self.scanner:
            sanitized, is_valid, risk_score = self.scanner.scan(text)
            if not is_valid:
                violation = SafetyViolation(
                    violation_type=SafetyViolationType.PROMPT_INJECTION,
                    severity="high" if risk_score > 0.7 else "medium",
                    message=f"LLM-Guard detected prompt injection (risk: {risk_score:.2f})",
                    location="input",
                    details={"risk_score": risk_score, "sanitized": sanitized},
                    suggested_action=SafetyAction.BLOCK if risk_score > 0.7 else SafetyAction.WARN,
                    confidence=risk_score,
                )
                result.add_violation(violation)
        
        if assessment.action == SafetyAction.BLOCK:
            result.passed = False
            result.action = SafetyAction.BLOCK
        elif assessment.action == SafetyAction.REVIEW:
            result.action = SafetyAction.REVIEW
        elif assessment.action == SafetyAction.WARN:
            result.action = SafetyAction.WARN
        else:
            result.action = SafetyAction.ALLOW
        
        result.metadata["risk_assessment"] = {
            "risk_score": sum(s.weight for s in self.detect_signals(text)),
            "max_signal_weight": max((s.weight for s in self.detect_signals(text)), default=0),
            "signal_count": len(self.detect_signals(text)),
            "educational_context": self.check_educational_context(text),
            "channel": location,
            "is_critical": False,
        }
        
        return result

    def _check_llm_guard_available(self) -> bool:
        try:
            import llm_guard
            return True
        except ImportError:
            return False

    def sanitize(self, text: str) -> str:
        if self.scanner:
            sanitized, _, _ = self.scanner.scan(text)
            return sanitized
        return text


def create_prompt_injection_detector(**kwargs) -> PromptInjectionDetector:
    return PromptInjectionDetector(**kwargs)
