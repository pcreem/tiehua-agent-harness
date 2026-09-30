"""Deterministic Evaluator for Agent Harness

Evaluates Agent execution results against deterministic criteria.
No LLM-as-a-Judge in MVP.
"""

from typing import Dict, Any, List, Optional
from src.state import AgentState


class Evaluator:
    """Deterministic evaluator for Agent results."""

    # Keywords that indicate weather intent
    WEATHER_KEYWORDS = [
        "天氣", "氣溫", "溫度", "下雨", "會不會下雨", "雨",
        "降雨", "濕度", "風", "颱風", "晴", "陰", "多雲",
        "weather", "rain", "temperature", "forecast"
    ]

    # Keywords that indicate attraction/activity intent
    ATTRACTION_KEYWORDS = [
        "住宿", "飯店", "民宿", "旅館", "露營",
        "景點", "玩", "去哪", "推薦", "好玩",
        "活動", "夜市", "美食", "餐廳", "小吃", "宵夜",
        "好吃", "吃", "美食",
        "鐵花", "鐵花村",
        "accommodation", "hotel", "attraction", "sightseeing",
        "food", "restaurant", "activity"
    ]

    # Keywords that indicate rain-related query
    RAIN_KEYWORDS = [
        "下雨", "雨天", "雨", "濕", "淋雨", "避雨",
        "rain", "rainy", "wet"
    ]

    def __init__(self):
        self.checks = [
            self._check_weather_tool_called,
            self._check_attraction_tool_called,
            self._check_answer_not_empty,
            self._check_rain_day_suitability,
            self._check_answer_groundedness,
        ]

    def evaluate(self, query: str, state: AgentState) -> Dict[str, Any]:
        """
        Evaluate agent execution result.

        Args:
            query: Original user query
            state: AgentState after execution

        Returns:
            Evaluation dict with status, score, error_type, feedback, retry
        """
        all_feedback = []
        error_types = []

        for check in self.checks:
            result = check(query, state)
            if result:
                feedback, error_type = result
                all_feedback.extend(feedback)
                if error_type:
                    error_types.append(error_type)

        if not error_types:
            return {
                "status": "pass",
                "score": 1.0,
                "error_type": None,
                "feedback": [],
                "retry": False
            }

        # Determine primary error type (first one)
        primary_error = error_types[0]

        return {
            "status": "fail",
            "score": 0.5,
            "error_type": primary_error,
            "feedback": all_feedback,
            "retry": True
        }

    def _check_weather_tool_called(self, query: str, state: AgentState) -> Optional[tuple]:
        """Check if weather-related query called get_weather tool."""
        if not self._needs_weather(query):
            return None

        if not state.has_called_tool("get_weather"):
            return (["查詢天氣相關問題但未呼叫 get_weather 工具"], "missing_evidence")
        return None

    def _check_attraction_tool_called(self, query: str, state: AgentState) -> Optional[tuple]:
        """Check if attraction-related query called search_attractions tool."""
        if not self._needs_attraction(query):
            return None

        if not state.has_called_tool("search_attractions"):
            return (["查詢景點/住宿/活動相關問題但未呼叫 search_attractions 工具"], "missing_evidence")
        return None

    def _check_answer_not_empty(self, query: str, state: AgentState) -> Optional[tuple]:
        """Check if answer is not empty."""
        answer = state.answer or ""
        if not answer.strip():
            return (["回答內容為空"], "incomplete_answer")
        return None

    def _check_rain_day_suitability(self, query: str, state: AgentState) -> Optional[tuple]:
        """Check if rain-related query has suitable attraction evidence."""
        if not self._is_rain_query(query):
            return None

        # Check if search_attractions was called
        if not state.has_called_tool("search_attractions"):
            return (["詢問雨天活動但未搜尋景點"], "missing_evidence")

        # Get latest attraction results
        latest_result = state.get_latest_tool_result("search_attractions")
        if not latest_result or not latest_result.get("success"):
            return (["景點搜尋失敗或無結果"], "tool_error")

        # Check if any results mention indoor/covered/rain-suitable
        # Our dataset doesn't have explicit rainy_day field, check description/indoor_outdoor
        results = latest_result.get("results", [])
        if not results:
            return (["雨天查詢但無景點結果"], "missing_evidence")

        # Look for indoor/covered indicators in descriptions
        rain_suitable_found = False
        for r in results[:5]:  # Check top 5
            desc = (r.get("description") or "").lower()
            indoor = (r.get("indoor_outdoor") or "").lower()
            category = (r.get("category") or "").lower()
            tags = [t.lower() for t in r.get("tags", [])]

            # Consider accommodation as generally rain-suitable
            if category == "accommodation":
                rain_suitable_found = True
                break
            # Check description for indoor/covered keywords
            if any(kw in desc for kw in ["室內", "遮雨", "屋內", "館", "室內", "有屋頂"]):
                rain_suitable_found = True
                break
            if indoor in ["indoor", "室內", "有遮蔽"]:
                rain_suitable_found = True
                break

        if not rain_suitable_found:
            return (["雨天查詢但推薦景點缺少雨天適合性證據（建議優先推薦室內/有遮蔽/住宿類景點）"], "missing_evidence")

        return None

    def _check_answer_groundedness(self, query: str, state: AgentState) -> Optional[tuple]:
        """Check if answer uses important information from tool results."""
        answer = (state.answer or "").lower()
        if not answer.strip():
            return None  # Already caught by _check_answer_not_empty

        feedback = []

        # Check if weather tool was called but answer doesn't mention key weather info
        if state.has_called_tool("get_weather"):
            weather_result = state.get_latest_tool_result("get_weather")
            if weather_result and weather_result.get("success"):
                temp = weather_result.get("temperature")
                weather = weather_result.get("weather")
                precip = weather_result.get("precipitation_probability")

                # Check if answer mentions temperature or weather condition
                has_temp = temp is not None and str(temp) in answer
                has_weather = weather and weather.lower() in answer
                has_precip = precip is not None and str(precip) in answer

                if not (has_temp or has_weather or has_precip):
                    feedback.append("回答未引用天氣工具回傳的關鍵資訊（溫度/天氣狀況/降雨機率）")

        # Check if attraction tool was called but answer doesn't mention any result names
        if state.has_called_tool("search_attractions"):
            attr_result = state.get_latest_tool_result("search_attractions")
            if attr_result and attr_result.get("success"):
                results = attr_result.get("results", [])
                if results:
                    # Check if any result name appears in answer
                    mentioned = False
                    for r in results[:5]:
                        name = r.get("name", "")
                        if name and name.lower() in answer:
                            mentioned = True
                            break
                    if not mentioned:
                        feedback.append("回答未引用景點搜尋工具回傳的具體景點名稱")

        if feedback:
            return (feedback, "missing_evidence")

        return None

    def _needs_weather(self, query: str) -> bool:
        """Check if query needs weather information."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.WEATHER_KEYWORDS)

    def _needs_attraction(self, query: str) -> bool:
        """Check if query needs attraction/activity information."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.ATTRACTION_KEYWORDS)

    def _is_rain_query(self, query: str) -> bool:
        """Check if query is about rain/rainy day."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.RAIN_KEYWORDS)


def evaluate(query: str, state: AgentState) -> Dict[str, Any]:
    """Convenience function to evaluate."""
    evaluator = Evaluator()
    return evaluator.evaluate(query, state)


if __name__ == "__main__":
    # Quick test with mock state
    from src.state import AgentState, initialize_state

    # Test 1: Weather query without tool call
    state1 = initialize_state("台東今天天氣如何？")
    state1.set_answer("今天天氣很好。")
    print("Test 1 (weather query, no tool):")
    print(evaluate("台東今天天氣如何？", state1))

    # Test 2: Weather query with tool call
    state2 = initialize_state("台東今天天氣如何？")
    state2.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
    state2.add_tool_result("get_weather", {"city": "Taitung"}, {
        "success": True, "temperature": 28, "weather": "cloudy", "precipitation_probability": 20
    }, "call_1")
    state2.set_answer("台東今天多雲，溫度 28°C，降雨機率 20%。")
    print("\nTest 2 (weather query with tool):")
    print(evaluate("台東今天天氣如何？", state2))

    # Test 3: Rain query with attraction
    state3 = initialize_state("台東下雨天適合去哪裡住宿？")
    state3.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
    state3.add_tool_result("get_weather", {"city": "Taitung"}, {
        "success": True, "temperature": 25, "weather": "rain", "precipitation_probability": 90
    }, "call_1")
    state3.add_tool_call("search_attractions", {"city": "Taitung", "category": "accommodation"}, "call_2")
    state3.add_tool_result("search_attractions", {"city": "Taitung", "category": "accommodation"}, {
        "success": True, "results": [{"name": "測試民宿", "category": "accommodation"}]
    }, "call_2")
    state3.set_answer("台東下雨天建議住宿，推薦測試民宿。")
    print("\nTest 3 (rain query with accommodation):")
    print(evaluate("台東下雨天適合去哪裡住宿？", state3))