"""Evaluator base class and registry."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.evaluation.types import (
    EvaluationResult,
    ExpectedBehavior,
    EvaluatorStatus,
    EvaluationCriterion,
    CriterionResult,
)
from src.state import AgentState


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    name: str = "base"
    description: str = "Base evaluator"

    @abstractmethod
    def evaluate(
        self,
        query: str,
        state: AgentState,
        expected: Optional[Any] = None,
    ) -> EvaluationResult:
        """Evaluate agent execution."""
        pass

    def _create_result(
        self,
        passed: bool,
        criteria_results: List = None,
        feedback: List[str] = None,
        score: float = None,
        metadata: dict = None,
    ) -> EvaluationResult:
        if criteria_results is None:
            criteria_results = []
        if feedback is None:
            feedback = []
        if score is None:
            score = 1.0 if passed else 0.5
        if metadata is None:
            metadata = {}

        return EvaluationResult(
            status=EvaluatorStatus.PASS if passed else EvaluatorStatus.FAIL,
            score=score,
            criteria_results=criteria_results,
            feedback=feedback,
            retry=not passed,
            metadata=metadata,
        )


class DeterministicEvaluator(BaseEvaluator):
    """Deterministic evaluator with 5 core checks."""

    name = "deterministic"
    description = "Deterministic checks: tool calls, answer quality, rain suitability, groundedness"

    WEATHER_KEYWORDS = [
        "天氣", "氣溫", "溫度", "下雨", "會不會下雨", "雨",
        "降雨", "濕度", "風", "颱風", "晴", "陰", "多雲",
    ]

    ATTRACTION_KEYWORDS = [
        "住宿", "飯店", "民宿", "旅館", "露營",
        "景點", "玩", "去哪", "推薦", "好玩",
        "活動", "夜市", "美食", "餐廳", "小吃", "宵夜",
        "好吃", "吃",
    ]

    RAIN_KEYWORDS = [
        "下雨", "雨天", "雨", "濕", "淋雨", "避雨",
    ]

    def evaluate(
        self,
        query: str,
        state: AgentState,
        expected: Optional[Any] = None,
    ) -> EvaluationResult:
        criteria_results = []
        all_feedback = []

        # Check 1: Weather tool called if needed
        if self._needs_weather(query):
            passed = state.has_called_tool("get_weather")
            criteria_results.append(CriterionResult(
                criterion=EvaluationCriterion.WEATHER_TOOL_CALLED,
                passed=passed,
                message="Weather query must call get_weather tool" if not passed else "",
            ))
            if not passed:
                all_feedback.append("查詢天氣相關問題但未呼叫 get_weather 工具")

        # Check 2: Attraction tool called if needed
        if self._needs_attraction(query):
            passed = state.has_called_tool("search_attractions")
            criteria_results.append(CriterionResult(
                criterion=EvaluationCriterion.ATTRACTION_TOOL_CALLED,
                passed=passed,
                message="Attraction query must call search_attractions tool" if not passed else "",
            ))
            if not passed:
                all_feedback.append("查詢景點/住宿/活動相關問題但未呼叫 search_attractions 工具")

        # Check 3: Answer not empty
        answer = state.answer or ""
        passed = bool(answer.strip())
        criteria_results.append(CriterionResult(
            criterion=EvaluationCriterion.ANSWER_NOT_EMPTY,
            passed=passed,
            message="回答內容為空" if not passed else "",
        ))
        if not passed:
            all_feedback.append("回答內容為空")

        # Check 4: Rain day suitability
        if self._is_rain_query(query):
            passed = self._check_rain_suitability(state)
            criteria_results.append(CriterionResult(
                criterion=EvaluationCriterion.RAIN_SUITABILITY,
                passed=passed,
                message="雨天查詢但推薦景點缺少雨天適合性證據" if not passed else "",
            ))
            if not passed:
                all_feedback.append("雨天查詢但推薦景點缺少雨天適合性證據（建議優先推薦室內/有遮蔽/住宿類景點）")

        # Check 5: Groundedness
        passed, grounding_feedback = self._check_groundedness(state)
        criteria_results.append(CriterionResult(
            criterion=EvaluationCriterion.GROUNDEDNESS,
            passed=passed,
            message="; ".join(grounding_feedback) if grounding_feedback else "",
        ))
        all_feedback.extend(grounding_feedback)

        all_passed = all(cr.passed for cr in criteria_results)
        score = sum(cr.score for cr in criteria_results) / len(criteria_results) if criteria_results else 1.0

        return self._create_result(
            passed=all_passed,
            criteria_results=criteria_results,
            feedback=all_feedback,
            score=score,
            metadata={"evaluator": self.name},
        )

    def _needs_weather(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.WEATHER_KEYWORDS)

    def _needs_attraction(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.ATTRACTION_KEYWORDS)

    def _is_rain_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.RAIN_KEYWORDS)

    def _check_rain_suitability(self, state: AgentState) -> bool:
        if not state.has_called_tool("search_attractions"):
            return False

        latest_result = state.get_latest_tool_result("search_attractions")
        if not latest_result or not latest_result.get("success"):
            return False

        results = latest_result.get("results", [])
        if not results:
            return False

        for r in results[:5]:
            category = (r.get("category") or "").lower()
            desc = (r.get("description") or "").lower()
            indoor = (r.get("indoor_outdoor") or "").lower()

            if category == "accommodation":
                return True
            if any(kw in desc for kw in ["室內", "遮雨", "屋內", "館", "有屋頂"]):
                return True
            if indoor in ["indoor", "室內", "有遮蔽"]:
                return True

        return False

    def _check_groundedness(self, state: AgentState) -> tuple:
        """Check if answer uses tool results."""
        answer = (state.answer or "").lower()
        feedback = []

        if state.has_called_tool("get_weather"):
            weather_result = state.get_latest_tool_result("get_weather")
            if weather_result and weather_result.get("success"):
                temp = weather_result.get("temperature")
                weather = weather_result.get("weather")
                weather_desc = weather_result.get("weather_description")
                precip = weather_result.get("precipitation_probability")

                has_temp = temp is not None and str(temp) in answer
                has_weather = weather and weather.lower() in answer
                has_weather_desc = weather_desc and weather_desc.lower() in answer
                has_precip = precip is not None and str(precip) in answer

                if not (has_temp or has_weather or has_weather_desc or has_precip):
                    feedback.append("回答未引用天氣工具回傳的關鍵資訊（溫度/天氣狀況/降雨機率）")

        if state.has_called_tool("search_attractions"):
            attr_result = state.get_latest_tool_result("search_attractions")
            if attr_result and attr_result.get("success"):
                results = attr_result.get("results", [])
                if results:
                    # Check if answer references at least one attraction name
                    found_ref = False
                    for r in results[:5]:
                        name = (r.get("name") or "").lower()
                        if name and name in answer:
                            found_ref = True
                            break
                    if not found_ref:
                        feedback.append("回答未引用景點搜尋結果中的任何景點名稱")
            else:
                feedback.append("景點搜尋工具執行失敗")

        return len(feedback) == 0, feedback


# Convenience function
def evaluate(query: str, state: AgentState) -> dict:
    """Convenience function for backward compatibility."""
    evaluator = DeterministicEvaluator()
    result = evaluator.evaluate(query, state)
    return result.to_dict()