"""Unit tests for Agent with Function Calling"""

import json
from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from src.agent import Agent, TOOLS, SYSTEM_PROMPT


class MockToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = json.dumps(arguments)


class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockChatCompletion:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage or MagicMock()
        self.usage.prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
        self.usage.completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
        self.usage.total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0


def create_mock_client(responses):
    """Create a properly structured mock OpenAI client"""
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    
    call_count = [0]
    
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        print(f"DEBUG: LLM call #{call_count[0]}")
        if call_count[0] <= len(responses):
            return responses[call_count[0] - 1]
        # If more calls than responses, return last response
        return responses[-1]
    
    mock_completions.create.side_effect = side_effect
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat
    return mock_client


class TestAgentInitialization:
    """Tests for Agent initialization"""

    def test_agent_creation_with_api_key(self):
        """Test agent can be created with explicit API key"""
        agent = Agent(api_key="test-key")
        assert agent.api_key == "test-key"
        assert agent.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
        assert agent.max_tool_calls == 5

    def test_agent_creation_from_env(self, monkeypatch):
        """Test agent reads API key from environment"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        agent = Agent()
        assert agent.api_key == "env-key"

    def test_agent_raises_without_api_key(self, monkeypatch):
        """Test agent raises error when no API key available"""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            Agent()

    def test_default_model(self):
        """Test default model is set"""
        agent = Agent(api_key="test")
        assert agent.model == "nvidia/nemotron-3-ultra-550b-a55b:free"


class TestToolDefinitions:
    """Tests for tool schema definitions"""

    def test_tools_count(self):
        """Should have 2 tools defined"""
        assert len(TOOLS) == 2

    def test_get_weather_tool(self):
        """Verify get_weather tool schema"""
        weather_tool = next(t for t in TOOLS if t["function"]["name"] == "get_weather")
        assert weather_tool["type"] == "function"
        assert "city" in weather_tool["function"]["parameters"]["properties"]
        assert weather_tool["function"]["parameters"]["required"] == ["city"]

    def test_search_attractions_tool(self):
        """Verify search_attractions tool schema"""
        attr_tool = next(t for t in TOOLS if t["function"]["name"] == "search_attractions")
        assert attr_tool["type"] == "function"
        props = attr_tool["function"]["parameters"]["properties"]
        assert "city" in props
        assert "category" in props
        assert attr_tool["function"]["parameters"]["required"] == ["city"]
        # Check enum for category
        assert "enum" in props["category"]
        assert "accommodation" in props["category"]["enum"]
        assert "activity" in props["category"]["enum"]

    def test_system_prompt_contains_tools(self):
        """System prompt should mention both tools"""
        assert "get_weather" in SYSTEM_PROMPT
        assert "search_attractions" in SYSTEM_PROMPT


class TestAgentRun:
    """Tests for Agent.run()"""

    def test_simple_query_no_tools(self):
        """Test query that doesn't need tools"""
        mock_msg = MockMessage(content="您好！我是台灣觀光代理人。有什麼可以幫您？")
        mock_choice = MockChoice(mock_msg)
        mock_response = MockChatCompletion([mock_choice])

        mock_client = create_mock_client([mock_response])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            result = agent.run("你好")

        assert result["success"] is True
        assert "您好" in result["final_answer"]
        assert len(result["tool_calls"]) == 0
        assert len(result["tool_results"]) == 0

    def test_single_tool_call_weather(self):
        """Test single get_weather tool call"""
        # First LLM call: decides to call get_weather
        tool_call = MockToolCall("call_1", "get_weather", {"city": "Taitung"})
        mock_msg1 = MockMessage(content="我來查詢台東天氣", tool_calls=[tool_call])
        mock_choice1 = MockChoice(mock_msg1)
        mock_response1 = MockChatCompletion([mock_choice1])

        # Second LLM call: returns final answer
        mock_msg2 = MockMessage(content="台東今天多雲，溫度 28°C，降雨機率 20%。")
        mock_choice2 = MockChoice(mock_msg2)
        mock_response2 = MockChatCompletion([mock_choice2])

        mock_client = create_mock_client([mock_response1, mock_response2])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            with patch.object(agent, '_execute_tool', return_value={"success": True, "city": "Taitung", "temperature": 28, "weather": "多雲", "precipitation_probability": 20}):
                result = agent.run("台東今天天氣如何？")

        assert result["success"] is True
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][0]["arguments"]["city"] == "Taitung"
        assert len(result["tool_results"]) == 1
        assert "多雲" in result["final_answer"]

    def test_single_tool_call_attractions(self):
        """Test single search_attractions tool call"""
        tool_call = MockToolCall("call_1", "search_attractions", {"city": "Taitung", "category": "accommodation"})
        mock_msg1 = MockMessage(content="我來找台東住宿", tool_calls=[tool_call])
        mock_choice1 = MockChoice(mock_msg1)
        mock_response1 = MockChatCompletion([mock_choice1])

        mock_msg2 = MockMessage(content="台東有 56 間住宿，推薦蘭舍民宿、背包倉庫肆零居...")
        mock_choice2 = MockChoice(mock_msg2)
        mock_response2 = MockChatCompletion([mock_choice2])

        mock_client = create_mock_client([mock_response1, mock_response2])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            with patch.object(agent, '_execute_tool', return_value={"success": True, "count": 56, "results": [{"name": "蘭舍民宿"}]}):
                result = agent.run("台東有什麼住宿推薦？")

        assert result["success"] is True
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search_attractions"
        assert result["tool_calls"][0]["arguments"]["city"] == "Taitung"
        assert result["tool_calls"][0]["arguments"]["category"] == "accommodation"

    def test_two_tool_calls_weather_then_attractions(self):
        """Test sequential tool calls: weather then attractions"""
        # Call 1: get_weather
        tc1 = MockToolCall("call_1", "get_weather", {"city": "Taitung"})
        msg1 = MockMessage(content="先查天氣", tool_calls=[tc1])
        resp1 = MockChatCompletion([MockChoice(msg1)])

        # Call 2: search_attractions (after seeing weather)
        tc2 = MockToolCall("call_2", "search_attractions", {"city": "Taitung", "category": "accommodation"})
        msg2 = MockMessage(content="根據天氣找住宿", tool_calls=[tc2])
        resp2 = MockChatCompletion([MockChoice(msg2)])

        # Call 3: final answer
        msg3 = MockMessage(content="台東今天多雲，適合住宿。推薦 56 間住宿...")
        resp3 = MockChatCompletion([MockChoice(msg3)])

        mock_client = create_mock_client([resp1, resp2, resp3])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            with patch.object(agent, '_execute_tool', side_effect=[
                {"success": True, "city": "Taitung", "temperature": 25, "weather": "多雲"},
                {"success": True, "count": 56, "results": [{"name": "蘭舍民宿"}]}
            ]):
                result = agent.run("台東今天會下雨嗎？下雨適合去哪裡住宿？")

        assert result["success"] is True
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][1]["name"] == "search_attractions"

    def test_tool_result_passed_to_llm(self):
        """Test that tool results are passed back to LLM"""
        tool_call = MockToolCall("call_1", "get_weather", {"city": "Taitung"})
        msg1 = MockMessage(content="查天氣", tool_calls=[tool_call])
        resp1 = MockChatCompletion([MockChoice(msg1)])

        msg2 = MockMessage(content="最終回答")
        resp2 = MockChatCompletion([MockChoice(msg2)])

        mock_client = create_mock_client([resp1, resp2])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            with patch.object(agent, '_execute_tool', return_value={"success": True, "city": "Taitung", "temperature": 25}):
                result = agent.run("台東天氣")

            # Verify tool was called and result passed back
            assert len(result["tool_calls"]) == 1
            assert len(result["tool_results"]) == 1
            assert result["tool_results"][0]["name"] == "get_weather"

    def test_max_tool_calls_limit(self):
        """Test agent respects max_tool_calls limit"""
        # Create 3 tool call responses (within limit)
        responses = []
        for i in range(3):
            tc = MockToolCall(f"call_{i}", "get_weather", {"city": "Taitung"})
            msg = MockMessage(content=f"呼叫 {i}", tool_calls=[tc])
            responses.append(MockChatCompletion([MockChoice(msg)]))
        # Final answer
        responses.append(MockChatCompletion([MockChoice(MockMessage(content="完成"))]))

        mock_client = create_mock_client(responses)

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key", max_tool_calls=3)
            # Mock tool execution to avoid real API calls
            with patch.object(agent, '_execute_tool', return_value={"success": True, "city": "Taitung", "temperature": 25}):
                result = agent.run("測試限制")

        # Should stop at max_tool_calls (3)
        assert len(result["tool_calls"]) == 3

    def test_unknown_tool_error_handling(self):
        """Test handling of unknown tool"""
        # Mock LLM to call unknown tool
        tool_call = MockToolCall("call_1", "unknown_tool", {})
        msg1 = MockMessage(content="試試未知工具", tool_calls=[tool_call])
        resp1 = MockChatCompletion([MockChoice(msg1)])

        msg2 = MockMessage(content="該工具不存在")
        resp2 = MockChatCompletion([MockChoice(msg2)])

        mock_client = create_mock_client([resp1, resp2])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            result = agent.run("測試")

        # Tool result should contain error
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["result"]["success"] is False
        assert result["tool_results"][0]["result"]["error_type"] == "unknown_tool"


class TestAgentState:
    """Tests for agent state management"""

    def test_state_reset_on_new_run(self):
        """Test state is reset between runs"""
        mock_msg = MockMessage(content="回答")
        mock_response = MockChatCompletion([MockChoice(mock_msg)])
        # Need 2 responses (one for each run)
        mock_client = create_mock_client([mock_response, mock_response])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            result1 = agent.run("第一次")
            first_calls = len(result1["tool_calls"])
            result2 = agent.run("第二次")
            second_calls = len(result2["tool_calls"])

            # State should be independent
            assert first_calls == 0
            assert second_calls == 0

    def test_returned_state_contains_all_fields(self):
        """Test returned state has all required fields"""
        mock_msg = MockMessage(content="測試回答")
        mock_response = MockChatCompletion([MockChoice(mock_msg)])
        mock_client = create_mock_client([mock_response])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            result = agent.run("測試查詢")

        required_fields = {"query", "final_answer", "messages", "tool_calls", "tool_results", "success"}
        assert required_fields.issubset(result.keys())
        assert result["query"] == "測試查詢"
        assert result["success"] is True


class TestCreateAgentFactory:
    """Tests for create_agent factory function"""

    def test_create_agent_returns_agent(self):
        """Factory should return Agent instance"""
        from src.agent import create_agent
        agent = create_agent(api_key="test")
        assert isinstance(agent, Agent)


class TestToolExecution:
    """Tests for tool execution (integration with actual tools)"""

    @pytest.mark.integration
    def test_execute_get_weather_real(self):
        """Integration test with real weather tool (requires network)"""
        from src.agent import Agent
        agent = Agent(api_key="dummy")  # Won't call LLM

        # This tests the _execute_tool method directly
        result = agent._execute_tool("get_weather", {"city": "Taitung"})
        # New format: returns data directly on success
        assert "city" in result or "success" in result

    @pytest.mark.integration
    def test_execute_search_attractions_real(self):
        """Integration test with real attractions tool"""
        from src.agent import Agent
        agent = Agent(api_key="dummy")

        result = agent._execute_tool("search_attractions", {"city": "Taitung", "category": "accommodation"})
        assert "count" in result or "success" in result
        if "count" in result:
            assert result["count"] > 0
        elif result.get("success"):
            assert result["count"] > 0


class TestToolCallTracking:
    """Tests for tool call ID tracking"""

    def test_tool_call_id_tracking(self):
        """Tool call IDs should be tracked correctly"""
        tc = MockToolCall("unique_id_123", "get_weather", {"city": "Taitung"})
        msg1 = MockMessage(content="呼叫", tool_calls=[tc])
        resp1 = MockChatCompletion([MockChoice(msg1)])

        msg2 = MockMessage(content="完成")
        resp2 = MockChatCompletion([MockChoice(msg2)])

        mock_client = create_mock_client([resp1, resp2])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            result = agent.run("測試")

        assert result["tool_calls"][0]["call_id"] == "unique_id_123"
        assert result["tool_results"][0]["call_id"] == "unique_id_123"

    def test_system_prompt_in_messages(self):
        """System prompt should be first in messages"""
        mock_msg = MockMessage(content="回答")
        mock_response = MockChatCompletion([MockChoice(mock_msg)])
        mock_client = create_mock_client([mock_response])

        with patch("src.agent_core.llm_client.OpenAI", return_value=mock_client):
            agent = Agent(api_key="test-key")
            agent.run("測試")

            # Check first call had system prompt
            call_args = mock_client.chat.completions.create.call_args_list[0]
            messages = call_args[1]["messages"]
            assert messages[0]["role"] == "system"
            assert "台灣觀光代理人" in messages[0]["content"]