"""
Tiehua Tourism Agent - Chat Interface with Statistics Dashboard.

A Streamlit chat interface that:
- Lets users chat with the tourism agent
- Shows conversation history
- Displays live statistics (tool usage, eval scores, safety, etc.)
- Supports multilingual (zh-TW, en, ja)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.harness.runner import HarnessRunner
from src.agent import create_agent
from src.state import initialize_state
from dotenv import load_dotenv

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
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 85%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: auto;
    }
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    .status-fallback { color: #ffc107; font-weight: bold; }
    .sidebar-section {
        border-bottom: 1px solid #eee;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# === Session State Initialization ===

def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "runner" not in st.session_state:
        st.session_state.runner = None
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "total_queries": 0,
            "total_tool_calls": 0,
            "tool_usage": {},
            "languages": {},
            "intents": {},
            "statuses": {},
            "total_retries": 0,
            "passed": 0,
            "failed": 0,
            "start_time": time.time(),
        }
    if "current_language" not in st.session_state:
        st.session_state.current_language = "zh-TW"
    if "max_retries" not in st.session_state:
        st.session_state.max_retries = 2
    if "use_langgraph" not in st.session_state:
        st.session_state.use_langgraph = True


def get_runner(max_retries: int, use_langgraph: bool) -> HarnessRunner:
    """Get or create HarnessRunner with current settings."""
    key = f"runner_{max_retries}_{use_langgraph}"
    if key not in st.session_state or st.session_state[key] is None:
        st.session_state[key] = HarnessRunner(max_retries=max_retries, use_langgraph=use_langgraph)
    return st.session_state[key]


def update_stats(result):
    """Update statistics from harness result."""
    stats = st.session_state.stats
    stats["total_queries"] += 1
    
    # Tool usage
    for tc in result.state.tool_calls:
        tool_name = tc.get("name", "unknown")
        stats["tool_usage"][tool_name] = stats["tool_usage"].get(tool_name, 0) + 1
        stats["total_tool_calls"] += 1
    
    # Language & Intent
    lang = result.state.language
    stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
    
    intent = result.state.user_intent
    stats["intents"][intent] = stats["intents"].get(intent, 0) + 1
    
    # Status
    status = result.status
    stats["statuses"][status] = stats["statuses"].get(status, 0) + 1
    
    # Retries
    stats["total_retries"] += result.retry_count
    
    # Pass/Fail
    if status == "passed":
        stats["passed"] += 1
    else:
        stats["failed"] += 1


def render_sidebar():
    """Render sidebar with settings and stats."""
    with st.sidebar:
        st.title("🏮 Tiehua Chat")
        st.caption("台東觀光代理人對話介面")
        
        # Settings
        st.markdown("---")
        st.subheader("⚙️ 設定")
        
        st.session_state.current_language = st.selectbox(
            "回應語言 / Language",
            ["zh-TW", "en", "ja"],
            index=["zh-TW", "en", "ja"].index(st.session_state.current_language),
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
            st.session_state.conversation_id = None
            # Keep stats across conversations unless user wants reset
            st.rerun()
        
        if st.button("🗑️ 重置統計", use_container_width=True):
            init_session_state()
            st.rerun()
        
        st.markdown("---")
        st.subheader("📊 即時統計")
        
        stats = st.session_state.stats
        
        # Quick metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("對話次數", stats["total_queries"])
            st.metric("工具呼叫", stats["total_tool_calls"])
        with col2:
            pass_rate = 0
            if stats["total_queries"] > 0:
                pass_rate = stats["passed"] / stats["total_queries"] * 100
            st.metric("通過率", f"{pass_rate:.0f}%")
            st.metric("總重試", stats["total_retries"])
        
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
                emoji = "✅" if status == "passed" else "❌" if status == "failed" else "⚠️"
                st.write(f"  {emoji} {status}: {count}")
        
        # Uptime
        uptime = time.time() - stats["start_time"]
        st.caption(f"執行時間: {int(uptime//60)}分 {int(uptime%60)}秒")


def render_tool_expander(label: str, content: Dict, expanded: bool = False):
    """Render tool call/result in an expander."""
    with st.expander(label, expanded=expanded):
        st.json(content, expanded=False)


def format_result_for_display(result_dict: Dict) -> str:
    """Format tool result for user-friendly display."""
    if not isinstance(result_dict, dict):
        return str(result_dict)
    
    if not result_dict.get("success", False):
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
                                st.json(tc.get("arguments", {}), expanded=False)
                    if "tool_results" in msg and msg["tool_results"]:
                        for tr in msg["tool_results"]:
                            with st.expander(f"📥 工具結果: {tr['name']}", expanded=False):
                                st.markdown(format_result_for_display(tr.get("result", {})))
            elif msg["role"] == "tool":
                # Legacy format
                with st.chat_message("assistant"):
                    st.markdown(f"`工具訊息`: {msg['content'][:200]}...")
    
    # Chat input
    user_input = st.chat_input(
        placeholder="輸入您的問題... (例如: 台東明天天氣如何？ / 台東有什麼推薦住宿？)"
    )
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    runner = get_runner(
                        st.session_state.max_retries,
                        st.session_state.use_langgraph
                    )
                    
                    result = runner.run(user_input)
                    
                    # Build assistant message
                    assistant_msg = {
                        "role": "assistant",
                        "content": result.final_answer or "(空白回應)",
                        "tool_calls": result.state.tool_calls,
                        "tool_results": [
                            {"name": tr["name"], "result": tr["result"]} 
                            for tr in result.state.tool_results
                        ],
                        "metadata": {
                            "run_id": result.state.run_id,
                            "status": result.status,
                            "language": result.state.language,
                            "intent": result.state.user_intent,
                            "retry_count": result.retry_count,
                            "duration_ms": getattr(result, 'duration_ms', 0),
                        }
                    }
                    st.session_state.messages.append(assistant_msg)
                    
                    # Update stats
                    update_stats(result)
                    
                    # Display result
                    st.markdown(result.final_answer or "(空白回應)")
                    
                    # Show tool calls/results inline
                    if result.state.tool_calls:
                        for tc in result.state.tool_calls:
                            with st.expander(f"🔧 工具呼叫: {tc['name']}", expanded=False):
                                st.json(tc.get("arguments", {}), expanded=False)
                    
                    if result.state.tool_results:
                        for tr in result.state.tool_results:
                            with st.expander(f"📥 工具結果: {tr['name']}", expanded=False):
                                st.markdown(format_result_for_display(tr.get("result", {})))
                    
                    # Status indicator
                    status_emoji = "✅" if result.status == "passed" else "❌" if result.status == "failed" else "⚠️"
                    st.caption(f"{status_emoji} 狀態: {result.status} | 語言: {result.state.language} | 意圖: {result.state.user_intent} | 重試: {result.retry_count} | 執行時間: {getattr(result, 'duration_ms', 0)/1000:.1f}s")
                    
                except Exception as e:
                    st.error(f"執行失敗: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Rerun to show new messages
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
            st.metric("平均耗時", "~15s")  # Placeholder
        
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
                status_colors = {"passed": "#28a745", "failed": "#dc3545", "fallback": "#ffc107"}
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
                    orientation="h",
                    title="語言分布",
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