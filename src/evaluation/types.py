"""Evaluation types and data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class EvaluatorStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class EvaluationCriterion(str, Enum):
    WEATHER_TOOL_CALLED = "weather_tool_called"
    ATTRACTION_TOOL_CALLED = "attraction_tool_called"
    ANSWER_NOT_EMPTY = "answer_not_empty"
    RAIN_SUITABILITY = "rain_suitability"
    GROUNDEDNESS = "groundedness"


@dataclass
class CriterionResult:
    """Result of a single evaluation criterion."""

    criterion: EvaluationCriterion
    passed: bool
    score: float = 1.0
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    status: EvaluatorStatus
    score: float  # 0.0 - 1.0
    criteria_results: List[CriterionResult] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)
    retry: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "score": self.score,
            "criteria_results": [
                {
                    "criterion": cr.criterion.value,
                    "passed": cr.passed,
                    "score": cr.score,
                    "message": cr.message,
                    "evidence": cr.evidence,
                }
                for cr in self.criteria_results
            ],
            "feedback": self.feedback,
            "retry": self.retry,
            "metadata": self.metadata,
        }

    @property
    def passed_criteria(self) -> int:
        return sum(1 for cr in self.criteria_results if cr.passed)

    @property
    def total_criteria(self) -> int:
        return len(self.criteria_results)


@dataclass
class ExpectedBehavior:
    """Expected behavior specification for evaluation."""

    required_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    required_facts: List[str] = field(default_factory=list)
    required_conditions: List[str] = field(default_factory=list)
    must_be_grounded: bool = False
    expected_outcome: Optional[str] = None
    category: Optional[str] = None
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: dict) -> "ExpectedBehavior":
        return cls(
            required_tools=data.get("required_tools", []),
            forbidden_tools=data.get("forbidden_tools", []),
            required_facts=data.get("required_facts", []),
            required_conditions=data.get("required_conditions", []),
            must_be_grounded=data.get("must_be_grounded", False),
            expected_outcome=data.get("expected_outcome"),
            category=data.get("category"),
            difficulty=data.get("difficulty", "medium"),
        )