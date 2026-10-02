"""Safety types and data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class SafetyViolationType(str, Enum):
    """Types of safety violations."""
    PROMPT_INJECTION = "prompt_injection"
    PII_DETECTED = "pii_detected"
    TOXIC_CONTENT = "toxic_content"
    TOOL_VIOLATION = "tool_violation"
    SENSITIVE_TOPIC = "sensitive_topic"


class SafetyAction(str, Enum):
    """Actions to take when violation detected."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    SANITIZE = "sanitize"
    REDACT = "redact"
    REVIEW = "review"


class PipelineMode(str, Enum):
    """Pipeline execution mode."""
    STRICT = "strict"
    PERMISSIVE = "permissive"
    CUSTOM = "custom"


@dataclass
class PipelineConfig:
    """Configuration for the safety pipeline."""
    mode: str = "strict"  # strict, permissive, custom
    enable_prompt_injection: bool = True
    enable_pii_detection: bool = True
    enable_content_filter: bool = True
    enable_tool_guard: bool = True
    pii_redact: bool = True
    custom_decision_fn: Optional[Callable] = None
    max_latency_ms: int = 1000


@dataclass
class PipelineResult:
    """Result of the entire safety pipeline."""
    passed: bool
    action: str
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
        return any(v.suggested_action == "block" for v in self.violations)


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    text: str
    channel: str
    normalized_text: str
    signals: List = field(default_factory=list)
    risk_score: float = 0.0
    max_signal_weight: float = 0.0
    educational_context: bool = False
    action: SafetyAction = SafetyAction.ALLOW
    is_critical: bool = False


@dataclass
class SafetyViolation:
    """A detected safety violation."""
    violation_type: str
    severity: str  # low, medium, high, critical
    message: str
    location: str  # input, output, tool_args, tool_result
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = "block"
    confidence: float = 1.0


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    passed: bool
    violations: List = field(default_factory=list)
    sanitized_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_violation(self, violation) -> None:
        self.violations.append(violation)
        self.passed = False

    def has_critical(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)

    def get_blocking_violations(self) -> List:
        return [v for v in self.violations if v.suggested_action == "block"]


class PIIEntityType(str, Enum):
    """Types of PII entities."""
    PERSON_NAME = "person_name"
    EMAIL = "email"
    PHONE = "phone"
    ID_CARD = "id_card"  # 台灣身分證
    PHONE_TW = "phone_tw"  # 台灣電話
    ADDRESS = "address"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"


@dataclass
class PIIEntity:
    """Detected PII entity."""
    entity_type: str
    value: str
    start: int
    end: int
    confidence: float
    redacted_value: str = ""