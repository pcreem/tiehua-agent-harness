"""PII (Personally Identifiable Information) detection and redaction."""

import re
from typing import List, Optional, Dict, Any, Match

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

from .types import (
    SafetyCheckResult,
    SafetyViolation,
    SafetyViolationType,
    SafetyAction,
    PIIEntity,
    PIIEntityType,
)


class PIIDetector:
    """PII detection and redaction using Presidio and custom regex patterns."""

    # Custom regex patterns for Taiwan-specific PII
    TW_ID_PATTERN = re.compile(r'[A-Z][12]\d{8}')  # 台灣身分證
    TW_PHONE_PATTERN = re.compile(r'(09\d{8}|0[2-8]\d{7,8})')  # 台灣電話
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
    IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    URL_PATTERN = re.compile(r'https?://[^\s]+')

    # Chinese name pattern (簡單版本)
    CN_NAME_PATTERN = re.compile(r'[\u4e00-\u9fff]{2,4}')

    def __init__(
        self,
        language: str = "zh",
        use_presidio: bool = True,
        custom_recognizers: Optional[List] = None,
    ):
        self.language = language
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE

        if self.use_presidio:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None

        # Custom regex patterns for Taiwan-specific PII
        self.custom_patterns = [
            (PIIEntityType.ID_CARD, self.TW_ID_PATTERN, 0.95),
            (PIIEntityType.PHONE_TW, self.TW_PHONE_PATTERN, 0.9),
            (PIIEntityType.EMAIL, self.EMAIL_PATTERN, 0.95),
            (PIIEntityType.CREDIT_CARD, self.CREDIT_CARD_PATTERN, 0.9),
            (PIIEntityType.IP_ADDRESS, self.IP_PATTERN, 0.8),
            (PIIEntityType.URL, self.URL_PATTERN, 0.85),
        ]

    def check(self, text: str, location: str = "input") -> 'SafetyCheckResult':
        """Check text for PII."""
        from .types import SafetyCheckResult, SafetyViolation, SafetyViolationType, SafetyAction, PIIEntity, PIIEntityType

        result = SafetyCheckResult(passed=True)

        # Custom regex patterns
        for entity_type, pattern, confidence in self.custom_patterns:
            for match in pattern.finditer(text):
                pii = PIIEntity(
                    entity_type=entity_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    redacted_value=self._redact_value(entity_type, match.group()),
                )

                violation = SafetyViolation(
                    violation_type=SafetyViolationType.PII_DETECTED,
                    severity="medium",
                    message=f"PII detected: {entity_type.value} = {match.group()}",
                    location=location,
                    details={
                        "entity_type": entity_type.value,
                        "value": match.group(),
                        "position": match.span(),
                        "redacted": pii.redacted_value,
                    },
                    suggested_action=SafetyAction.REDACT,
                    confidence=confidence,
                )
                result.add_violation(violation)

        # Presidio analysis
        if self.use_presidio and self.analyzer:
            results = self.analyzer.analyze(text=text, language=self.language)
            for res in results:
                # Map Presidio entity types to our types
                entity_type = self._map_presidio_type(res.entity_type)
                if entity_type:
                    pii = PIIEntity(
                        entity_type=entity_type,
                        value=text[res.start:res.end],
                        start=res.start,
                        end=res.end,
                        confidence=res.score,
                        redacted_value="*" * (res.end - res.start),
                    )

                    violation = SafetyViolation(
                        violation_type=SafetyViolationType.PII_DETECTED,
                        severity="medium",
                        message=f"PII detected: {res.entity_type} = {text[res.start:res.end]}",
                        location=location,
                        details={
                            "entity_type": res.entity_type,
                            "value": text[res.start:res.end],
                            "position": (res.start, res.end),
                            "presidio_score": res.score,
                            "redacted": pii.redacted_value,
                        },
                        suggested_action=SafetyAction.REDACT,
                        confidence=res.score,
                    )
                    result.add_violation(violation)

        return result

    def _map_presidio_type(self, presidio_type: str) -> Optional['PIIEntityType']:
        """Map Presidio entity types to our PIIEntityType."""
        from .types import PIIEntityType
        mapping = {
            "PERSON": PIIEntityType.PERSON_NAME,
            "EMAIL_ADDRESS": PIIEntityType.EMAIL,
            "PHONE_NUMBER": PIIEntityType.PHONE,
            "CREDIT_CARD": PIIEntityType.CREDIT_CARD,
            "IP_ADDRESS": PIIEntityType.IP_ADDRESS,
            "URL": PIIEntityType.URL,
            "LOCATION": PIIEntityType.ADDRESS,
        }
        return mapping.get(presidio_type)

    def _redact_value(self, entity_type: 'PIIEntityType', value: str) -> str:
        """Generate redacted version of PII value."""
        if entity_type == PIIEntityType.EMAIL:
            parts = value.split('@')
            return parts[0][:2] + '***@' + parts[1]
        elif entity_type in (PIIEntityType.PHONE, PIIEntityType.PHONE_TW):
            return value[:3] + '****' + value[-2:]
        elif entity_type == PIIEntityType.ID_CARD:
            return value[0] + '********'
        elif entity_type == PIIEntityType.CREDIT_CARD:
            return '****-****-****-' + value[-4:]
        else:
            return '*' * len(value)

    def redact(self, text: str) -> str:
        """Redact all PII in text."""
        result = self.check(text)
        redacted = text
        # Replace from end to start to maintain positions
        for violation in sorted(result.violations, key=lambda v: v.details.get("position", (0,0))[0], reverse=True):
            pos = violation.details.get("position", (0, 0))
            if pos:
                start, end = pos
                redacted = violation.details.get("redacted", "*" * (end - start))
                text = text[:start] + redacted + text[end:]
        return text


def create_pii_detector(**kwargs) -> 'PIIDetector':
    """Factory function to create PII detector."""
    return PIIDetector(**kwargs)