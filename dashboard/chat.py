"""
Tiehua Tourism Agent - Chat Interface with Statistics Dashboard.

A Streamlit chat interface that:
- Lets users chat with the tourism agent
- Shows conversation history
- Displays live statistics (tool usage, eval scores, safety, etc.)
- Supports multilingual (zh-TW, en, ja)
"""

import streamlit as st
import plotly.express as px
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.harness.runner import HarnessRunner
from src.state import initialize_state
from dotenv import load_dotenv


# Custom runner that supports language parameter
class LanguageAwareRunner(HarnessRunner):
    """Extended runner that accepts language parameter and supports multi-turn conversation."""
    
    def __init__(self, *args, language: str = "zh-TW", **kwargs):
        super().__init__(*args, **kwargs)
        self.language = language
    
    def run(self, query: str, conversation_id: str = None, history: list = None) -> "HarnessResult":
        """Run the harness with a user query, language, and conversation history.
        
        Args:
            query: User query string
            conversation_id: Optional conversation ID for multi-turn context (used as thread_id)
            history: Optional list of previous messages for context
        """
        from src.harness.result import HarnessResult
        from src.observability import (
            get_metrics,
            harness_logger,
        )
        from src.observability.tracing import trace_agent_run
        
        start_time = time.time()
        metrics = get_metrics()
        
        # Use conversation_id as thread_id for checkpointing across turns
        thread_id = conversation_id or ""
        
        with trace_agent_run(query, thread_id) as span:
            try:
                # Create agent
                agent = self.agent_factory()
                
                # Initialize state with language and history
                state = initialize_state(query, max_retries=self.max_retries)
                state.language = self.language
                
                # Add conversation history to state messages if provided
                if history:
                    # Filter out metadata messages and keep only user/assistant/tool messages
                    for msg in history:
                        if msg.get("role") in ("user", "assistant", "tool"):
                            # Reconstruct message in state format
                            state.add_message(
                                msg["role"],
                                msg.get("content", ""),
                                tool_calls=msg.get("tool_calls"),
                                tool_call_id=msg.get("tool_call_id") if msg.get("role") == "tool" else None
                            )
                
                if conversation_id:
                    # Ensure metadata is a dict
                    if state.metadata is None:
                        state.metadata = {}
                    state.metadata["conversation_id"] = conversation_id
                
                if self.use_langgraph:
                    # Override thread_id with conversation_id for checkpoint persistence
                    final_state = self._run_with_langgraph_with_thread(agent, state, thread_id)
                else:
                    final_state = self._run_with_loop(agent, state)
                
                duration = time.time() - start_time
                
                # Build result
                result = HarnessResult(
                    query=final_state.query,
                    final_answer=final_state.answer or "",
                    state=final_state,
                    success=final_state.status == "passed",
                    status=final_state.status,
                    retry_count=final_state.attempt,
                    tool_calls=final_state.tool_calls,
                    tool_results=final_state.tool_results,
                    evaluation=final_state.evaluation,
                    feedback=final_state.feedback,
                    duration_ms=int(duration * 1000),
                )
                
                # Record metrics
                metrics.record_agent_run(result.status, duration)
                
                harness_logger.info("harness_completed",
                    query=query[:100],
                    status=result.status,
                    retries=result.retry_count,
                    duration=duration
                )
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                
                # Monitoring failures should not mask original exception
                try:
                    metrics.record_error(type(e).__name__, "harness")
                    metrics.record_agent_run("failed", duration)
                    harness_logger.error("harness_failed", query=query[:100], error=str(e), duration=duration)
                except Exception:
                    pass
                
                raise
    
    def _run_with_langgraph_with_thread(self, agent, state, thread_id: str):
        """Execute using LangGraph-based harness with custom thread_id."""
        from src.harness.graph import HarnessGraphRunner
        
        graph_runner = HarnessGraphRunner(
            agent_fn=lambda s: agent.run_with_state(s),
            evaluator_fn=self.evaluator_fn,
            max_retries=self.max_retries,
            thread_id=thread_id,  # Use conversation_id as thread_id for checkpoint persistence
        )
        return graph_runner.run(state.query, max_retries=self.max_retries)


load_dotenv()

# Page config
st.set_page_config(
    page_title="Tiehua Tourism Agent - Chat",
    page_icon="🏮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .tool-call {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        font-family: monospace;
    }
    .tool-result {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        font-family: monospace;
    }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    .status-fallback { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# === Session State Initialization ===

def reset_stats():
    """Reset only statistics data."""
    st.session_state.stats = {
        "total_queries": 0,
        "completed_queries": 0,
        "total_tool_calls": 0,
        "tool_usage": {},
        "languages": {},
        "intents": {},
        "statuses": {},
        "total_retries": 0,
        "passed": 0,
        "failed": 0,
        "fallback": 0,
        "total_duration_ms": 0,
        "stats_start_time": time.time(),
    }


def init_session_state(force_reset: bool = False):
    """Initialize all session state variables.
    
    Args:
        force_reset: If True, reinitialize all state including stats and messages.
    """
    if "app_start_time" not in st.session_state or force_reset:
        st.session_state.app_start_time = time.time()
    if "messages" not in st.session_state or force_reset:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state or force_reset:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "runner" not in st.session_state or force_reset:
        st.session_state.runner = None
    if "runner_config" not in st.session_state or force_reset:
        st.session_state.runner_config = {"max_retries": 2, "use_langgraph": True, "language": "zh-TW"}
    if "stats" not in st.session_state or force_reset:
        reset_stats()
    if "current_language" not in st.session_state or force_reset:
        st.session_state.current_language = "zh-TW"
    if "max_retries" not in st.session_state or force_reset:
        st.session_state.max_retries = 2
    if "use_langgraph" not in st.session_state or force_reset:
        st.session_state.use_langgraph = True


def get_runner(max_retries: int, use_langgraph: bool, language: str) -> LanguageAwareRunner:
    """Get or create LanguageAwareRunner with current settings."""
    # Check if we need to recreate runner due to config change
    current_config = st.session_state.runner_config
    if (st.session_state.runner is None or
        current_config["max_retries"] != max_retries or
        current_config["use_langgraph"] != use_langgraph or
        current_config.get("language") != language):
        st.session_state.runner = LanguageAwareRunner(
            max_retries=max_retries,
            use_langgraph=use_langgraph,
            language=language
        )
        st.session_state.runner_config = {
            "max_retries": max_retries,
            "use_langgraph": use_langgraph,
            "language": language
        }
    return st.session_state.runner


def update_stats(result):
    """Update statistics from harness result."""
    stats = st.session_state.stats
    try:
        stats["total_queries"] += 1
        stats["completed_queries"] += 1
        
        # Tool usage - use safe extraction
        tool_calls = safe_get_tool_calls(getattr(result.state, 'tool_calls', []))
        for tc in tool_calls:
            tool_name = tc["name"]
            stats["tool_usage"][tool_name] = stats["tool_usage"].get(tool_name, 0) + 1
            stats["total_tool_calls"] += 1
        
        # Language & Intent
        if hasattr(result, 'state'):
            lang = getattr(result.state, 'language', 'unknown')
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            intent = getattr(result.state, 'user_intent', 'unknown')
            stats["intents"][intent] = stats["intents"].get(intent, 0) + 1
        
        # Status
        status = getattr(result, 'status', 'unknown')
        stats["statuses"][status] = stats["statuses"].get(status, 0) + 1
        
        # Retries
        stats["total_retries"] += getattr(result, 'retry_count', 0)
        
        # Duration
        stats["total_duration_ms"] += getattr(result, 'duration_ms', 0)
        
        # Pass/Fail/Fallback classification
        if status == "passed":
            stats["passed"] += 1
        elif status == "fallback":
            stats["fallback"] += 1
        else:
            stats["failed"] += 1
    except Exception as e:
        # Log error but don't crash the UI
        import logging
        logging.getLogger(__name__).warning(f"Failed to update stats: {e}")


def record_failed_query(language: str = "unknown", intent: str = "unknown", duration_ms: int = 0):
    """Record statistics for a failed query (exception during runner.run())."""
    stats = st.session_state.stats
    try:
        stats["total_queries"] += 1
        stats["completed_queries"] += 1  # Track as completed for avg calculation
        stats["statuses"]["error"] = stats["statuses"].get("error", 0) + 1
        stats["failed"] += 1
        stats["languages"][language] = stats["languages"].get(language, 0) + 1
        stats["intents"][intent] = stats["intents"].get(intent, 0) + 1
        if duration_ms > 0:
            stats["total_duration_ms"] += duration_ms
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to record failed query stats: {e}")


def render_sidebar():
    """Render sidebar with settings and stats."""
    with st.sidebar:
        st.title("🏮 Tiehua Chat")
        st.caption("台東觀光代理人對話介面")
        
        # Settings
        st.markdown("---")
        st.subheader("⚙️ 設定")
        
        languages = ["zh-TW", "en", "ja"]
        # Safe index lookup with fallback to 0
        try:
            lang_index = languages.index(st.session_state.current_language)
        except ValueError:
            lang_index = 0
            st.session_state.current_language = languages[0]
        
        st.session_state.current_language = st.selectbox(
            "回應語言 / Language",
            languages,
            index=lang_index,
            help="模型回應語言；查詢語言會自動偵測"
        )
        
        st.session_state.max_retries = st.slider(
            "最大重試次數",
            0, 5, st.session_state.max_retries,
            help="評估失敗後最多重試幾次"
        )
        
        st.session_state.use_langgraph = st.checkbox(
            "使用 LangGraph",
            value=st.session_state.use_langgraph,
            help="啟用 LangGraph 以獲得更好的可觀測性"
        )
        
        if st.button("🔄 重置對話", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            # Keep stats across conversations unless user wants reset
            st.rerun()
        
        if st.button("🗑️ 重置統計", use_container_width=True):
            reset_stats()
            st.rerun()
        
        st.markdown("---")
        st.subheader("📊 即時統計")
        
        stats = st.session_state.stats
        
        # Quick metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("對話次數", stats["total_queries"])
            st.metric("完成次數", stats.get("completed_queries", 0))
            st.metric("工具呼叫", stats["total_tool_calls"])
        with col2:
            pass_rate = 0
            if stats["total_queries"] > 0:
                pass_rate = stats["passed"] / stats["total_queries"] * 100
            st.metric("通過率 (所有查詢)", f"{pass_rate:.0f}%")
            completed = stats.get("completed_queries", 0)
            if completed > 0:
                completed_pass_rate = stats["passed"] / completed * 100
                st.metric("完成查詢通過率", f"{completed_pass_rate:.0f}%")
            st.metric("總重試", stats["total_retries"])
            st.metric("Fallback", stats.get("fallback", 0))
        
        # Tool usage breakdown
        if stats["tool_usage"]:
            st.markdown("**工具使用分布**")
            for tool, count in stats["tool_usage"].items():
                st.progress(count / max(stats["total_tool_calls"], 1), text=f"{tool}: {count}")
        
        # Language distribution
        if stats["languages"]:
            st.markdown("**語言分布**")
            for lang, count in stats["languages"].items():
                st.write(f"  {lang}: {count}")
        
        # Intent distribution
        if stats["intents"]:
            st.markdown("**意圖分布**")
            for intent, count in stats["intents"].items():
                st.write(f"  {intent}: {count}")
        
        # Status distribution
        if stats["statuses"]:
            st.markdown("**狀態分布**")
            for status, count in stats["statuses"].items():
                if status == "passed":
                    emoji = "✅"
                elif status == "failed":
                    emoji = "❌"
                elif status == "fallback":
                    emoji = "⚠️"
                elif status == "error":
                    emoji = "💥"
                else:
                    emoji = "❓"
                st.write(f"  {emoji} {status}: {count}")
        
        # Uptime
        uptime = time.time() - st.session_state.get("app_start_time", time.time())
        stats_uptime = time.time() - stats.get("stats_start_time", time.time())
        st.caption(f"應用運行時間: {int(uptime//60)}分 {int(uptime%60)}秒 | 統計區間: {int(stats_uptime//60)}分 {int(stats_uptime%60)}秒")


def format_result_for_display(result_dict: Dict) -> str:
    """Format tool result for user-friendly display."""
    if not isinstance(result_dict, dict):
        return str(result_dict)
    
    # Only treat as failure if success is explicitly False (not just missing/falsy)
    if result_dict.get("success") is False:
        return f"❌ 失敗: {result_dict.get('message', '未知錯誤')}"
    
    # Weather
    if "weather_description" in result_dict:
        parts = []
        if result_dict.get("city"):
            parts.append(f"📍 {result_dict['city']}")
        if result_dict.get("weather_description"):
            parts.append(f"🌤️ {result_dict['weather_description']}")
        if result_dict.get("temperature") is not None:
            parts.append(f"🌡️ {result_dict['temperature']}°C")
        if result_dict.get("precipitation_probability") is not None:
            parts.append(f"💧 降雨機率 {result_dict['precipitation_probability']}%")
        if result_dict.get("daily_precipitation_probability_max") is not None:
            parts.append(f"📊 今日最高降雨 {result_dict['daily_precipitation_probability_max']}%")
        return " | ".join(parts)
    
    # Attractions
    if "results" in result_dict and isinstance(result_dict["results"], list):
        count = len(result_dict["results"])
        category = result_dict.get("category", "all")
        lines = [f"找到 {count} 個 {category} 景點:"]
        for i, r in enumerate(result_dict["results"][:5], 1):
            name = r.get("name", "未知")
            dist = r.get("district", "")
            cat = r.get("category", "")
            info = f"{i}. {name}"
            if dist:
                info += f" ({dist})"
            if cat:
                info += f" [{cat}]"
            lines.append(info)
        if count > 5:
            lines.append(f"... 還有 {count - 5} 個")
        return "\n".join(lines)
    
    # Generic
    return json.dumps(result_dict, ensure_ascii=False, indent=2)


# Sensitive data keys that should be masked in display
SENSITIVE_KEYS = {
    "api_key", "apikey", "api_token", "token", "secret", "password",
    "passwd", "pwd", "auth", "authorization", "credential", "credentials",
    "access_key", "secret_key", "private_key", "jwt", "session_id",
    "sessionid", "cookie", "csrf", "xsrf", "pin", "otp", "mfa"
}


def mask_sensitive_data(data: Any) -> Any:
    """Recursively mask sensitive values in dict/list structures."""
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in SENSITIVE_KEYS):
                masked[key] = "***MASKED***"
            else:
                masked[key] = mask_sensitive_data(value)
        return masked
    elif isinstance(data, list):
        return [mask_sensitive_data(item) for item in data]
    else:
        return data


def _extract_dict(obj: Any) -> Dict:
    """Extract dictionary from various object types (dict, Pydantic model, dataclass, etc.)."""
    if isinstance(obj, dict):
        return obj
    # Pydantic v2 models
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1 models
    if hasattr(obj, 'dict'):
        try:
            return obj.dict()
        except Exception:
            pass
    # Dataclasses
    if hasattr(obj, '__dataclass_fields__'):
        try:
            import dataclasses
            return dataclasses.asdict(obj)
        except Exception:
            pass
    # Objects with __dict__
    if hasattr(obj, '__dict__'):
        try:
            return vars(obj)
        except Exception:
            pass
    # Fallback: try to get attributes
    result = {}
    for attr in ['name', 'arguments', 'result', 'call_id', 'id', 'function']:
        if hasattr(obj, attr):
            result[attr] = getattr(obj, attr)
    return result if result else {"raw": str(obj)}


def safe_get_tool_calls(tool_calls: Any) -> List[Dict]:
    """Safely extract tool calls from result state.
    
    Handles: list of dicts, Pydantic models, dataclasses, objects with __dict__
    """
    if not isinstance(tool_calls, (list, tuple)):
        return []

    safe_calls = []

    for tc in tool_calls:
        tc_dict = _extract_dict(tc)

        # Handle function field - could be dict, string, or other
        function = tc_dict.get("function")
        if isinstance(function, dict):
            func_dict = function
        elif function is not None:
            # Could be string or object - extract dict representation
            func_dict = _extract_dict(function)
        else:
            func_dict = {}

        # Extract name: prefer top-level name, then function.name
        name = tc_dict.get("name") or func_dict.get("name") or "unknown"

        # Extract arguments: prefer top-level, then function.arguments
        arguments = tc_dict.get("arguments")
        if arguments is None:
            arguments = func_dict.get("arguments", {})

        # Ensure arguments is a dict
        if not isinstance(arguments, dict):
            arguments = _extract_dict(arguments) if arguments is not None else {}

        safe_calls.append({
            "name": name,
            "arguments": arguments,
            "call_id": tc_dict.get("call_id") or tc_dict.get("id", "")
        })

    return safe_calls


def safe_get_tool_results(tool_results: Any) -> List[Dict]:
    """Safely extract tool results from result state.
    
    Handles: list of dicts, Pydantic models, dataclasses, objects with __dict__
    """
    if not isinstance(tool_results, (list, tuple)):
        return []
    safe_results = []
    for tr in tool_results:
        tr_dict = _extract_dict(tr)
        safe_results.append({
            "name": tr_dict.get("name", "unknown"),
            "arguments": tr_dict.get("arguments", {}),
            "result": tr_dict.get("result", {}),
            "call_id": tr_dict.get("call_id", tr_dict.get("id", ""))
        })
    return safe_results


# === Main App ===

def main():
    st.title("🏮 Tiehua Tourism Agent - Chat")
    st.caption("台東觀光代理人 • 多語言支援 (zh-TW / en / ja) • 工具增強 RAG")
    
    init_session_state()
    render_sidebar()
    
    # Main chat area
    st.markdown("---")
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    # Show tool calls/results if any
                    if "tool_calls" in msg and msg["tool_calls"]:
                        for tc in msg["tool_calls"]:
                            with st.expander(f"🔧 工具呼叫: {tc['name']}", expanded=False):
                                st.json(mask_sensitive_data(tc.get("arguments", {})), expanded=False)
                    if "tool_results" in msg and msg["tool_results"]:
                        for tr in msg["tool_results"]:
                            with st.expander(f"📥 工具結果: {tr['name']}", expanded=False):
                                st.markdown(format_result_for_display(mask_sensitive_data(tr.get("result", {}))))
                    # Show metadata if available
                    if "metadata" in msg:
                        meta = msg["metadata"]
                        if "status" in meta:
                            status_emoji = "✅" if meta["status"] == "passed" else "❌" if meta["status"] == "failed" else "⚠️"
                            duration = meta.get("duration_ms", 0) / 1000
                            st.caption(f"{status_emoji} 狀態: {meta['status']} | 語言: {meta.get('language', 'N/A')} | 意圖: {meta.get('intent', 'N/A')} | 重試: {meta.get('retry_count', 0)} | 執行時間: {duration:.1f}s")
            elif msg["role"] == "tool":
                # Legacy format
                with st.chat_message("assistant"):
                    st.markdown(f"`工具訊息`: {msg['content'][:200]}...")
    
    # Chat input
    user_input = st.chat_input(
        placeholder="輸入您的問題... (例如: 台東明天天氣如何？ / 台東有什麼推薦住宿？)"
    )
    
    # Initialize logger
    import logging
    logger = logging.getLogger(__name__)
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Run agent
        query_start_time = time.time()
        with st.spinner("思考中..."):
            try:
                runner = get_runner(
                    st.session_state.max_retries,
                    st.session_state.use_langgraph,
                    st.session_state.current_language
                )
                
                result = runner.run(
                    user_input,
                    conversation_id=st.session_state.conversation_id,
                    history=st.session_state.messages[:-1]  # Exclude the just-added user message
                )
                
                # Build assistant message
                tool_calls = safe_get_tool_calls(getattr(result.state, 'tool_calls', []))
                tool_results = safe_get_tool_results(getattr(result.state, 'tool_results', []))
                
                # Mask sensitive data in stored messages
                masked_tool_calls = [
                    {**tc, "arguments": mask_sensitive_data(tc.get("arguments", {}))}
                    for tc in tool_calls
                ]
                masked_tool_results = [
                    {**tr, "result": mask_sensitive_data(tr.get("result", {}))}
                    for tr in tool_results
                ]
                
                assistant_msg = {
                    "role": "assistant",
                    "content": result.final_answer or "(空白回應)",
                    "tool_calls": masked_tool_calls,
                    "tool_results": masked_tool_results,
                    "metadata": {
                        "run_id": getattr(result.state, 'run_id', ''),
                        "status": getattr(result, 'status', 'unknown'),
                        "language": getattr(result.state, 'language', 'unknown'),
                        "intent": getattr(result.state, 'user_intent', 'unknown'),
                        "retry_count": getattr(result, 'retry_count', 0),
                        "duration_ms": getattr(result, 'duration_ms', 0),
                    }
                }
                st.session_state.messages.append(assistant_msg)
                
                # Update stats
                update_stats(result)
                
            except Exception as e:
                query_duration_ms = int((time.time() - query_start_time) * 1000)
                # Log full error to server
                logger.exception("Tourism agent execution failed")
                # Show user-friendly message
                st.error("抱歉，服務暫時發生問題，請稍後再試。")
                # Record failed query stats with duration
                record_failed_query(
                    language=st.session_state.current_language,
                    intent="unknown",
                    duration_ms=query_duration_ms
                )
                # Add error message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "抱歉，服務暫時發生問題，請稍後再試。",
                    "tool_calls": [],
                    "tool_results": [],
                    "metadata": {"error": str(e), "status": "error", "duration_ms": query_duration_ms}
                })
        
        # Rerun to display new messages via the chat history loop
        st.rerun()
    
    # Bottom stats dashboard
    if st.session_state.stats["total_queries"] > 0:
        st.markdown("---")
        st.subheader("📈 完整統計儀表板")
        
        stats = st.session_state.stats
        
        # Row 1: Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("總查詢", stats["total_queries"])
        with col2:
            st.metric("工具呼叫", stats["total_tool_calls"])
        with col3:
            pass_rate = stats["passed"] / max(stats["total_queries"], 1) * 100
            st.metric("通過率", f"{pass_rate:.0f}%")
        with col4:
            st.metric("平均重試", f"{stats['total_retries'] / max(stats['total_queries'], 1):.1f}")
        with col5:
            completed = stats.get("completed_queries", 0)
            avg_duration = (
                stats["total_duration_ms"] / completed / 1000
                if completed > 0
                else 0
            )
            st.metric("平均耗時", f"{avg_duration:.1f}s")
        
# Row 2: Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Tool usage pie
            if stats["tool_usage"]:
                fig = px.pie(
                    values=list(stats["tool_usage"].values()),
                    names=list(stats["tool_usage"].keys()),
                    title="工具使用分布",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Status pie
            if stats["statuses"]:
                status_colors = {"passed": "#28a745", "failed": "#dc3545", "fallback": "#ffc107", "error": "#6c757d"}
                fig = px.pie(
                    values=list(stats["statuses"].values()),
                    names=list(stats["statuses"].keys()),
                    title="執行狀態分布",
                    color=list(stats["statuses"].keys()),
                    color_discrete_map=status_colors
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Language & Intent
        col1, col2 = st.columns(2)
        
        with col1:
            if stats["languages"]:
                fig = px.bar(
                    x=list(stats["languages"].values()),
                    y=list(stats["languages"].keys()),
                    labels={"x": "次數", "y": "語言"},
                    color=list(stats["languages"].values()),
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if stats["intents"]:
                fig = px.bar(
                    x=list(stats["intents"].values()),
                    y=list(stats["intents"].keys()),
                    orientation="h",
                    title="意圖分布",
                    labels={"x": "次數", "y": "意圖"},
                    color=list(stats["intents"].values()),
                    color_continuous_scale="Greens"
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()