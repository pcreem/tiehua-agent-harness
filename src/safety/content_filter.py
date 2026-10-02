"""Content filtering for toxic/harmful content using detoxify and custom rules."""

import re
from typing import List, Optional, Dict, Any

# Lazy import - only load when actually needed
DETOXIFY_AVAILABLE = True
_DETOXIFY_MODEL = None


def _get_detoxify_model():
    """Lazy load the detoxify model."""
    global _DETOXIFY_MODEL
    if _DETOXIFY_MODEL is None:
        from detoxify import Detoxify
        _DETOXIFY_MODEL = Detoxify('multilingual')
    return _DETOXIFY_MODEL


from .types import (
    SafetyCheckResult,
    SafetyViolation,
    SafetyViolationType,
    SafetyAction,
)


class ContentFilter:
    """Content filtering for toxic, harmful, or inappropriate content."""

    # Sensitive topics that should be flagged
    SENSITIVE_TOPICS = {
        "violence": [
            r"(?i)(kill|murder|assassinate|slaughter|massacre)",
            r"(?i)(bomb|explosive|weapon|gun|knife|attack)",
            r"(?i)(torture|abuse|assault|harass)",
        ],
        "self_harm": [
            r"(?i)(suicide|kill\s+myself|end\s+my\s+life)",
            r"(?i)(self.harm|cutting|overdose)",
        ],
        "hate_speech": [
            r"(?i)(hate|despise|loathe)\s+(all|every)\s+\w+",
            r"(?i)(inferior|superior)\s+race",
            r"(?i)(exterminate|eliminate)\s+\w+\s+people",
        ],
        "sexual_content": [
            r"(?i)(explicit|pornographic|sexual)\s+(content|material)",
            r"(?i)(nude|naked|sexual)\s+(photo|image|video)",
        ],
        "illegal_activities": [
            r"(?i)(how\s+to\s+)(make|create|build)\s+(bomb|drug|weapon)",
            r"(?i)(steal|rob|hack|fraud|scam)",
            r"(?i)(buy|sell)\s+(drugs|weapons|stolen)",
        ],
        "misinformation": [
            r"(?i)(covid|vaccine|election)\s+(is|was)\s+(fake|hoax|lie)",
            r"(?i)(conspiracy|cover.up)\s+(theory|exposed)",
        ],
    }

    def __init__(
        self,
        use_detoxify: bool = True,
        toxicity_threshold: float = 0.7,
        custom_rules: Optional[Dict[str, List[str]]] = None,
    ):
        self.toxicity_threshold = toxicity_threshold
        self.use_detoxify = use_detoxify and DETOXIFY_AVAILABLE
        self._detoxify_model = None

        # Custom rules
        self.custom_rules = custom_rules or {}
        self.sensitive_patterns = {}
        for category, patterns in self.SENSITIVE_TOPICS.items():
            self.sensitive_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        # Add custom rules
        for category, patterns in self.custom_rules.items():
            if category not in self.sensitive_patterns:
                self.sensitive_patterns[category] = []
            self.sensitive_patterns[category].extend(
                [re.compile(p, re.IGNORECASE) for p in patterns]
            )

    def _get_detoxify_model(self):
        """Lazy load the detoxify model."""
        global _DETOXIFY_MODEL
        if _DETOXIFY_MODEL is None:
            from detoxify import Detoxify
            _DETOXIFY_MODEL = Detoxify('multilingual')
        return _DETOXIFY_MODEL

    def check(self, text: str, location: str = "input") -> 'SafetyCheckResult':
        """Check text for toxic/harmful content."""
        from .types import SafetyCheckResult, SafetyViolation, SafetyViolationType, SafetyAction

        result = SafetyCheckResult(passed=True)

        # Detoxify analysis (lazy load)
        if self.use_detoxify:
            detoxify = self._get_detoxify_model()
            scores = detoxify.predict(text)
            for category, score in scores.items():
                if score >= self.toxicity_threshold:
                    violation = SafetyViolation(
                        violation_type=SafetyViolationType.TOXIC_CONTENT,
                        severity="high" if score > 0.9 else "medium",
                        message=f"Toxic content detected: {category} (score: {score:.2f})",
                        location=location,
                        details={
                            "category": category,
                            "score": float(score),
                            "threshold": self.toxicity_threshold,
                        },
                        suggested_action=SafetyAction.BLOCK if score > 0.9 else SafetyAction.WARN,
                        confidence=score,
                    )
                    result.add_violation(violation)

        # Sensitive topic patterns
        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                matches = list(pattern.finditer(text))
                if matches:
                    for match in matches:
                        violation = SafetyViolation(
                            violation_type=SafetyViolationType.SENSITIVE_TOPIC,
                            severity="high" if category in ("violence", "self_harm", "illegal_activities") else "medium",
                            message=f"Sensitive topic detected: {category}",
                            location=location,
                            details={
                                "category": category,
                                "matched_text": match.group(),
                                "position": match.span(),
                            },
                            suggested_action=SafetyAction.WARN,
                            confidence=0.8,
                        )
                        result.add_violation(violation)

        return result


# Default instances
_default_filter = None


def get_content_filter() -> 'ContentFilter':
    """Get default content filter instance."""
    global _default_filter
    if _default_filter is None:
        _default_filter = ContentFilter()
    return _default_filter


def create_content_filter(**kwargs) -> 'ContentFilter':
    """Factory function to create content filter."""
    return ContentFilter(**kwargs)