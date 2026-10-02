"""Tool parameter validation and security guard."""

import re
import json
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass

from .types import (
    SafetyCheckResult,
    SafetyViolation,
    SafetyViolationType,
    SafetyAction,
)


@dataclass
class ToolParameterRule:
    """Validation rule for a tool parameter."""
    name: str
    type: str  # string, number, integer, boolean, array, object
    required: bool = False
    pattern: Optional[str] = None  # regex pattern for strings
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    enum: Optional[List[Any]] = None
    allowed_values: Optional[Set[Any]] = None
    custom_validator: Optional[callable] = None


class ToolGuard:
    """Validate and sanitize tool parameters for security."""

    # Dangerous patterns that should never appear in tool arguments
    DANGEROUS_PATTERNS = [
        # SQL Injection
        (r"(?i)(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)", "sql_injection"),
        (r"(?i)(\b(or|and)\s+\d+\s*=\s*\d+)", "sql_injection"),
        (r"(?i)('|--|;)\s*(\b(or|and)\b)", "sql_injection"),

        # Command Injection
        (r"(?i)(;|\||&|\$\(|\`)\s*(cat|ls|rm|wget|curl|nc|bash|sh|python)", "command_injection"),
        (r"(?i)(\$\(|\`).*(\$\(|\`)", "command_injection"),

        # Path Traversal
        (r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)", "path_traversal"),
        (r"(/etc/passwd|/etc/shadow|/proc/|/sys/|C:\\Windows\\)", "path_traversal"),

        # Code Injection
        (r"(?i)(eval|exec|compile|__import__)\s*\(", "code_injection"),
        (r"(?i)(<script|javascript:|onerror=|onload=)", "xss"),

        # LDAP Injection
        (r"(?i)(\*|\)\(|&|\|)", "ldap_injection"),

        # NoSQL Injection
        (r"(?i)(\$where|\$ne|\$gt|\$lt|\$regex)", "nosql_injection"),
    ]

    def __init__(
        self,
        tool_schemas: Optional[Dict[str, List[ToolParameterRule]]] = None,
        strict_mode: bool = True,
    ):
        self.strict_mode = strict_mode
        self.tool_rules = tool_schemas or {}
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), threat_type)
            for pattern, threat_type in self.DANGEROUS_PATTERNS
        ]

    def register_tool(self, tool_name: str, rules: List[ToolParameterRule]) -> None:
        """Register validation rules for a tool."""
        self.tool_rules[tool_name] = rules

    def check(self, tool_name: str, arguments: Dict[str, Any], location: str = "tool_args") -> 'SafetyCheckResult':
        """Validate tool arguments."""
        from .types import SafetyCheckResult, SafetyViolation, SafetyViolationType, SafetyAction

        result = SafetyCheckResult(passed=True)

        # Check against dangerous patterns
        for key, value in arguments.items():
            if isinstance(value, str):
                for pattern, threat_type in self.compiled_patterns:
                    if pattern.search(value):
                        from .types import SafetyViolation, SafetyViolationType, SafetyAction
                        violation = SafetyViolation(
                            violation_type=SafetyViolationType.TOOL_VIOLATION,
                            severity="critical",
                            message=f"Dangerous pattern detected in {tool_name}.{key}: {threat_type}",
                            location=location,
                            details={
                                "tool": tool_name,
                                "parameter": key,
                                "threat_type": threat_type,
                                "matched_text": value[:200],
                            },
                            suggested_action="block",
                            confidence=0.95,
                        )
                        result.add_violation(violation)

        # Schema validation if rules exist
        if tool_name in self.tool_rules:
            for rule in self.tool_rules[tool_name]:
                value = arguments.get(rule.name)
                violation = self._validate_parameter(rule, value, tool_name, location)
                if violation:
                    result.add_violation(violation)

        return result

    def _validate_parameter(
        self,
        rule: 'ToolParameterRule',
        value: Any,
        tool_name: str,
        location: str,
    ) -> Optional['SafetyViolation']:
        """Validate a single parameter against its rule."""
        from .types import SafetyViolation, SafetyViolationType, SafetyAction

        # Required check
        if value is None:
            if rule.required:
                return SafetyViolation(
                    violation_type=SafetyViolationType.TOOL_VIOLATION,
                    severity="high",
                    message=f"Required parameter missing: {tool_name}.{rule.name}",
                    location=location,
                    details={"tool": tool_name, "parameter": rule.name, "issue": "required_missing"},
                    suggested_action="block",
                    confidence=1.0,
                )
            return None

        # Type check
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_type = type_map.get(rule.type)
        if expected_type and not isinstance(value, expected_type):
            return SafetyViolation(
                violation_type=SafetyViolationType.TOOL_VIOLATION,
                severity="high",
                message=f"Type mismatch: {tool_name}.{rule.name} expected {rule.type}, got {type(value).__name__}",
                location=location,
                details={"tool": tool_name, "parameter": rule.name, "expected": rule.type, "got": type(value).__name__},
                suggested_action="block",
                confidence=1.0,
            )

        # String validations
        if isinstance(value, str):
            if rule.min_length and len(value) < rule.min_length:
                return self._make_violation(rule, tool_name, location,
                    f"String too short: {len(value)} < {rule.min_length}", "min_length")
            if rule.max_length and len(value) > rule.max_length:
                return self._make_violation(rule, tool_name, location,
                    f"String too long: {len(value)} > {rule.max_length}", "max_length")
            if rule.pattern and not re.match(rule.pattern, value):
                return self._make_violation(rule, tool_name, location,
                    f"Pattern mismatch: {rule.pattern}", "pattern_mismatch")

        # Numeric validations
        if isinstance(value, (int, float)):
            if rule.minimum is not None and value < rule.minimum:
                return self._make_violation(rule, tool_name, location,
                    f"Value too small: {value} < {rule.minimum}", "below_minimum")
            if rule.maximum is not None and value > rule.maximum:
                return self._make_violation(rule, tool_name, location,
                    f"Value too large: {value} > {rule.maximum}", "above_maximum")

        # Enum check
        if rule.enum and value not in rule.enum:
            return self._make_violation(rule, tool_name, location,
                f"Value not in allowed enum: {rule.enum}", "invalid_enum")

        # Allowed values check
        if rule.allowed_values and value not in rule.allowed_values:
            return self._make_violation(rule, tool_name, location,
                f"Value not in allowed set", "invalid_value")

        # Custom validator
        if rule.custom_validator:
            try:
                if not rule.custom_validator(value):
                    return self._make_violation(rule, tool_name, location,
                        "Custom validation failed", "custom_validator")
            except Exception as e:
                return self._make_violation(rule, tool_name, location,
                    f"Custom validator error: {e}", "custom_validator_error")

        return None

    def _make_violation(
        self,
        rule: 'ToolParameterRule',
        tool_name: str,
        location: str,
        message: str,
        issue_type: str,
    ) -> 'SafetyViolation':
        from .types import SafetyViolation, SafetyViolationType, SafetyAction
        return SafetyViolation(
            violation_type=SafetyViolationType.TOOL_VIOLATION,
            severity="high",
            message=f"Parameter validation failed: {message}",
            location=location,
            details={
                "tool": tool_name,
                "parameter": rule.name,
                "issue": issue_type,
            },
            suggested_action="block",
            confidence=1.0,
        )

    def sanitize(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize tool arguments by removing dangerous content."""
        sanitized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # Remove dangerous patterns
                sanitized_value = value
                for pattern, _ in self.compiled_patterns:
                    sanitized_value = pattern.sub("[REDACTED]", sanitized_value)
                sanitized[key] = sanitized_value
            else:
                sanitized[key] = value
        return sanitized


# Pre-defined tool schemas for built-in tools
DEFAULT_TOOL_RULES = {
    "get_weather": [
        ToolParameterRule(name="city", type="string", required=True, max_length=100),
    ],
    "search_attractions": [
        ToolParameterRule(name="city", type="string", required=True, max_length=100),
        ToolParameterRule(
            name="category",
            type="string",
            required=False,
            enum=["accommodation", "activity", "sightseeing", "food", "nature", "service"],
        ),
    ],
}


def create_tool_guard(rules: Optional[Dict[str, List[ToolParameterRule]]] = None) -> 'ToolGuard':
    """Factory function to create tool guard."""
    merged_rules = DEFAULT_TOOL_RULES.copy()
    if rules:
        merged_rules.update(rules)
    return ToolGuard(tool_schemas=merged_rules)