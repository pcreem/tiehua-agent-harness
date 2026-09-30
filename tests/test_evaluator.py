"""Unit tests for deterministic evaluator"""

import pytest
from src.evaluator import Evaluator, evaluate
from src.state import AgentState, initialize_state


class TestEvaluatorInitialization:
    """Tests for Evaluator initialization"""

    def test_evaluator_creation(self):
        """Evaluator should be creatable"""
        evaluator = Evaluator()
        assert evaluator is not None
        assert hasattr(evaluator, "checks")
        assert len(evaluator.checks) == 5


class TestEvaluateWeatherQuery:
    """Tests for weather query evaluation"""

    def test_weather_query_without_tool_fails(self):
        """Weather query without get_weather call should fail"""
        state = initialize_state("台東今天天氣如何？")
        state.set_answer("今天天氣很好。")

        result = evaluate("台東今天天氣如何？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"
        assert any("未呼叫 get_weather" in f for f in result["feedback"])
        assert result["retry"] is True

    def test_weather_query_with_tool_passes(self):
        """Weather query with get_weather call should pass"""
        state = initialize_state("台東今天天氣如何？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 28, "weather": "cloudy", "precipitation_probability": 20
        }, "call_1")
        state.set_answer("台東今天多雲，溫度 28°C，降雨機率 20%。")

        result = evaluate("台東今天天氣如何？", state)
        assert result["status"] == "pass"
        assert result["score"] == 1.0
        assert result["retry"] is False

    def test_english_weather_query_detected(self):
        """English weather keywords should be detected"""
        state = initialize_state("What's the weather in Taitung?")
        state.set_answer("It's sunny.")

        result = evaluate("What's the weather in Taitung?", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"


class TestEvaluateAttractionQuery:
    """Tests for attraction query evaluation"""

    def test_attraction_query_without_tool_fails(self):
        """Attraction query without search_attractions call should fail"""
        state = initialize_state("台東有什麼住宿推薦？")
        state.set_answer("台東有很多民宿。")

        result = evaluate("台東有什麼住宿推薦？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"
        assert any("未呼叫 search_attractions" in f for f in result["feedback"])

    def test_attraction_query_with_tool_passes(self):
        """Attraction query with search_attractions call should pass"""
        state = initialize_state("台東有什麼住宿推薦？")
        state.add_tool_call("search_attractions", {"city": "Taitung", "category": "accommodation"}, "call_1")
        state.add_tool_result("search_attractions", {"city": "Taitung", "category": "accommodation"}, {
            "success": True, "results": [{"name": "測試民宿", "category": "accommodation"}]
        }, "call_1")
        state.set_answer("推薦測試民宿。")

        result = evaluate("台東有什麼住宿推薦？", state)
        assert result["status"] == "pass"

    def test_food_query_needs_attraction_tool(self):
        """Food query should need search_attractions"""
        state = initialize_state("台東有什麼好吃的？")
        state.set_answer("有很多小吃。")

        result = evaluate("台東有什麼好吃的？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"


class TestEvaluateAnswerNotEmpty:
    """Tests for empty answer check"""

    def test_empty_answer_fails(self):
        """Empty answer should fail"""
        state = initialize_state("台東天氣如何？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 28, "weather": "cloudy", "precipitation_probability": 20
        }, "call_1")
        state.set_answer("")

        result = evaluate("台東天氣如何？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "incomplete_answer"
        assert any("回答內容為空" in f for f in result["feedback"])

    def test_whitespace_answer_fails(self):
        """Whitespace-only answer should fail"""
        state = initialize_state("你好")
        state.set_answer("   ")

        result = evaluate("你好", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "incomplete_answer"


class TestEvaluateRainQuery:
    """Tests for rain-related query evaluation"""

    def test_rain_query_without_attraction_fails(self):
        """Rain query without attraction search should fail"""
        state = initialize_state("台東下雨天適合去哪裡？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 25, "weather": "rain", "precipitation_probability": 90
        }, "call_1")
        state.set_answer("會下雨。")

        result = evaluate("台東下雨天適合去哪裡？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"
        assert any("未搜尋景點" in f for f in result["feedback"])

    def test_rain_query_with_accommodation_passes(self):
        """Rain query with accommodation results should pass"""
        state = initialize_state("台東下雨天適合去哪裡住宿？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 25, "weather": "rain", "precipitation_probability": 90
        }, "call_1")
        state.add_tool_call("search_attractions", {"city": "Taitung", "category": "accommodation"}, "call_2")
        state.add_tool_result("search_attractions", {"city": "Taitung", "category": "accommodation"}, {
            "success": True, "results": [{"name": "測試民宿", "category": "accommodation"}]
        }, "call_2")
        state.set_answer("台東今天降雨機率 90%，下雨天建議住宿，推薦測試民宿。")

        result = evaluate("台東下雨天適合去哪裡住宿？", state)
        assert result["status"] == "pass"

    def test_rain_query_with_outdoor_only_fails(self):
        """Rain query with only outdoor activity should fail"""
        state = initialize_state("台東下雨天適合去哪裡玩？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 25, "weather": "rain", "precipitation_probability": 90
        }, "call_1")
        state.add_tool_call("search_attractions", {"city": "Taitung", "category": "activity"}, "call_2")
        state.add_tool_result("search_attractions", {"city": "Taitung", "category": "activity"}, {
            "success": True, "results": [{
                "name": "戶外步道", "category": "activity",
                "description": "戶外健行步道", "indoor_outdoor": "outdoor"
            }]
        }, "call_2")
        state.set_answer("去戶外步道走走。")

        result = evaluate("台東下雨天適合去哪裡玩？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"
        assert any("雨天適合性" in f for f in result["feedback"])


class TestEvaluateGroundedness:
    """Tests for answer groundedness check"""

    def test_answer_without_weather_info_fails(self):
        """Answer not mentioning weather tool results should fail"""
        state = initialize_state("台東今天天氣如何？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 28, "weather": "cloudy", "precipitation_probability": 20
        }, "call_1")
        state.set_answer("今天天氣不錯。")  # No specific data mentioned

        result = evaluate("台東今天天氣如何？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"
        assert any("未引用天氣工具" in f for f in result["feedback"])

    def test_answer_with_weather_info_passes(self):
        """Answer mentioning weather data should pass"""
        state = initialize_state("台東今天天氣如何？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 28, "weather": "cloudy", "precipitation_probability": 20
        }, "call_1")
        state.set_answer("台東今天多雲，溫度 28°C。")

        result = evaluate("台東今天天氣如何？", state)
        assert result["status"] == "pass"

    def test_answer_without_attraction_names_fails(self):
        """Answer not mentioning attraction names should fail"""
        state = initialize_state("台東住宿推薦？")
        state.add_tool_call("search_attractions", {"city": "Taitung", "category": "accommodation"}, "call_1")
        state.add_tool_result("search_attractions", {"city": "Taitung", "category": "accommodation"}, {
            "success": True, "results": [{"name": "蘭舍民宿", "category": "accommodation"}]
        }, "call_1")
        state.set_answer("台東有很多住宿選擇。")  # No specific name

        result = evaluate("台東住宿推薦？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"
        assert any("未引用景點搜尋工具回傳的具體景點名稱" in f for f in result["feedback"])

    def test_answer_with_attraction_names_passes(self):
        """Answer mentioning attraction name should pass"""
        state = initialize_state("台東住宿推薦？")
        state.add_tool_call("search_attractions", {"city": "Taitung", "category": "accommodation"}, "call_1")
        state.add_tool_result("search_attractions", {"city": "Taitung", "category": "accommodation"}, {
            "success": True, "results": [{"name": "蘭舍民宿", "category": "accommodation"}]
        }, "call_1")
        state.set_answer("推薦蘭舍民宿。")

        result = evaluate("台東住宿推薦？", state)
        assert result["status"] == "pass"


class TestEvaluateMixedQueries:
    """Tests for queries needing both weather and attractions"""

    def test_weather_and_attraction_both_needed(self):
        """Query needing both tools should pass only if both called"""
        state = initialize_state("台東今天會下雨嗎？下雨適合去哪裡住宿？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 25, "weather": "rain", "precipitation_probability": 90
        }, "call_1")
        state.add_tool_call("search_attractions", {"city": "Taitung", "category": "accommodation"}, "call_2")
        state.add_tool_result("search_attractions", {"city": "Taitung", "category": "accommodation"}, {
            "success": True, "results": [{"name": "測試民宿", "category": "accommodation"}]
        }, "call_2")
        state.set_answer("台東今天會下雨，降雨機率 90%。建議住測試民宿。")

        result = evaluate("台東今天會下雨嗎？下雨適合去哪裡住宿？", state)
        assert result["status"] == "pass"

    def test_only_weather_called_for_mixed_query_fails(self):
        """Missing attraction tool for mixed query should fail"""
        state = initialize_state("台東今天會下雨嗎？下雨適合去哪裡住宿？")
        state.add_tool_call("get_weather", {"city": "Taitung"}, "call_1")
        state.add_tool_result("get_weather", {"city": "Taitung"}, {
            "success": True, "temperature": 25, "weather": "rain", "precipitation_probability": 90
        }, "call_1")
        state.set_answer("會下雨。")

        result = evaluate("台東今天會下雨嗎？下雨適合去哪裡住宿？", state)
        assert result["status"] == "fail"
        assert result["error_type"] == "missing_evidence"


class TestConvenienceFunction:
    """Tests for evaluate() convenience function"""

    def test_evaluate_function_works(self):
        """evaluate() function should work same as Evaluator().evaluate()"""
        state = initialize_state("台東天氣？")
        state.set_answer("好天氣。")

        result = evaluate("台東天氣？", state)
        assert "status" in result
        assert "score" in result
        assert "error_type" in result
        assert "feedback" in result
        assert "retry" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])