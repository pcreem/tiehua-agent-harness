"""
Evaluation Dashboard for Tiehua Tourism Agent Harness.

Streamlit-based dashboard for visualizing agent evaluation results,
LLM Judge scores, safety metrics, and agent performance metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Page config
st.set_page_config(
    page_title="Tiehua Tourism Agent - Evaluation Dashboard",
    page_icon="🏮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    .status-review { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_evaluation_data(data_dir: str = "evaluation/reports") -> Dict[str, Any]:
    """Load evaluation reports from JSON files."""
    reports_dir = Path(data_dir)
    if not reports_dir.exists():
        return {}
    
    data = {
        "reports": [],
        "summary": {}
    }
    
    for report_file in reports_dir.glob("*.json"):
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
                report["_file"] = report_file.name
                report["_mtime"] = datetime.fromtimestamp(report_file.stat().st_mtime)
                data["reports"].append(report)
        except Exception as e:
            st.warning(f"Failed to load {report_file}: {e}")
    
    # Generate summary statistics
    if data["reports"]:
        reports = data["reports"]
        data["summary"] = {
            "total_runs": len(reports),
            "passed": sum(1 for r in reports if r.get("status") == "pass"),
            "failed": sum(1 for r in reports if r.get("status") == "fail"),
            "avg_score": sum(r.get("score", 0) for r in reports) / len(reports) if reports else 0,
            "avg_retry": sum(r.get("retry_count", 0) for r in reports) / len(reports) if reports else 0,
            "tool_usage": {},
            "safety_violations": 0,
        }
        
        for r in reports:
            for tc in r.get("tool_calls", []):
                tool = tc.get("name", "unknown")
                data["summary"]["tool_usage"][tool] = data["summary"]["tool_usage"].get(tool, 0) + 1
            data["summary"]["safety_violations"] += len(r.get("safety_violations", []))
    
    return data


def render_overview_tab(data: Dict[str, Any]):
    """Render overview tab with key metrics."""
    st.header("📊 Overview")
    
    if not data.get("reports"):
        st.info("No evaluation data found. Run evaluations to generate reports.")
        return
    
    summary = data.get("summary", {})
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Runs",
            value=summary.get("total_runs", 0),
            delta=None
        )
    
    with col2:
        pass_rate = 0
        if summary.get("total_runs", 0) > 0:
            pass_rate = summary.get("passed", 0) / summary.get("total_runs", 1) * 100
        st.metric(
            label="Pass Rate",
            value=f"{pass_rate:.1f}%",
            delta=f"{summary.get('passed', 0)} passed / {summary.get('failed', 0)} failed"
        )
    
    with col3:
        st.metric(
            label="Avg Score",
            value=f"{summary.get('avg_score', 0):.2f}",
            delta=f"Target: ≥ 0.8"
        )
    
    with col4:
        st.metric(
            label="Avg Retries",
            value=f"{summary.get('avg_retry', 0):.1f}",
            delta=f"Max: 2"
        )
    
    with col5:
        st.metric(
            label="Safety Violations",
            value=summary.get("safety_violations", 0),
            delta="Target: 0"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        reports = data.get("reports", [])
        if reports:
            status_counts = pd.DataFrame([
                {"status": r.get("status", "unknown"), "count": 1}
                for r in data["reports"]
            ]).groupby("status").size().reset_index(name="count")
            
            fig = px.pie(
                status_counts, 
                values="count", 
                names="status",
                title="Evaluation Status Distribution",
                color="status",
                color_discrete_map={"pass": "#28a745", "fail": "#dc3545", "review": "#ffc107"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score distribution
        reports = data.get("reports", [])
        if data.get("reports"):
            scores = [r.get("score", 0) for r in data["reports"]]
            fig = px.histogram(
                x=scores,
                nbins=20,
                title="Score Distribution",
                labels={"x": "Score", "y": "Count"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                          annotation_text="Target: 0.8")
            st.plotly_chart(fig, use_container_width=True)


def render_tool_usage_tab(data: Dict[str, Any]):
    """Render tool usage analytics."""
    st.header("🔧 Tool Usage Analytics")
    
    tool_usage = data.get("summary", {}).get("tool_usage", {})
    
    if not tool_usage:
        st.info("No tool usage data available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        df = pd.DataFrame(
            list(tool_usage.items()), 
            columns=["Tool", "Usage Count"]
        ).sort_values("Usage Count", ascending=True)
        
        fig = px.bar(
            df, 
            y="Tool", 
            x="Usage Count",
            orientation="h",
            title="Tool Usage Frequency",
            color="Usage Count",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart
        fig = px.pie(
            values=list(tool_usage.values()),
            names=list(tool_usage.keys()),
            title="Tool Usage Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tool performance table
    st.subheader("Tool Performance Details")
    # Would add per-tool success rates, latency, etc. if data available


def render_safety_tab(data: Dict[str, Any]):
    """Render safety metrics tab."""
    st.header("🛡️ Safety Metrics")
    
    reports = data.get("reports", [])
    if not reports:
        st.info("No safety data available.")
        return
    
    # Safety violations breakdown
    violation_types = {}
    for r in data.get("reports", []):
        for v in r.get("safety_violations", []):
            vtype = v.get("type", "unknown")
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
    
    if violation_types:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=list(violation_types.values()),
                y=list(violation_types.keys()),
                orientation="h",
                title="Safety Violations by Type",
                labels={"x": "Count", "y": "Violation Type"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity distribution
            severity_counts = {}
            for r in data.get("reports", []):
                for v in r.get("safety_violations", []):
                    sev = v.get("severity", "unknown")
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            if severity_counts:
                fig = px.pie(
                    values=list(severity_counts.values()),
                    names=list(severity_counts.keys()),
                    title="Violations by Severity",
                    color_discrete_map={
                        "critical": "#dc3545",
                        "high": "#fd7e14",
                        "medium": "#ffc107",
                        "low": "#28a745"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Safety violations detail table
    if any(r.get("safety_violations") for r in data.get("reports", [])):
        st.subheader("Safety Violations Detail")
        violations_data = []
        for r in data.get("reports", []):
            for v in r.get("safety_violations", []):
                violations_data.append({
                    "Run ID": r.get("_file", "unknown")[:20],
                    "Type": v.get("type", "unknown"),
                    "Severity": v.get("severity", "unknown"),
                    "Message": v.get("message", "")[:100],
                    "Action": v.get("suggested_action", "unknown")
                })
        
        if violations_data:
            st.dataframe(pd.DataFrame(violations_data), use_container_width=True)


def render_llm_judge_tab(data: Dict[str, Any]):
    """Render LLM Judge evaluation tab."""
    st.header("🤖 LLM Judge Evaluations")
    
    reports = data.get("reports", [])
    llm_judge_reports = [r for r in data.get("reports", []) if r.get("evaluator") == "llm_judge"]
    
    if not llm_judge_reports:
        st.info("No LLM Judge evaluations found.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution for LLM Judge
        scores = [r.get("score", 0) for r in llm_judge_reports]
        if scores:
            fig = px.histogram(
                x=scores,
                nbins=10,
                title="LLM Judge Score Distribution",
                labels={"x": "Score", "y": "Count"}
            )
            fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                          annotation_text="Target: 0.8")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pass/Fail
        passed = sum(1 for r in llm_judge_reports if r.get("status") == "pass")
        failed = len(llm_judge_reports) - passed
        
        fig = px.pie(
            values=[passed, failed],
            names=["Passed", "Failed"],
            title="LLM Judge Pass/Fail",
            color_discrete_sequence=["#28a745", "#dc3545"]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Judge feedback
    st.subheader("Judge Feedback")
    for r in llm_judge_reports:
        with st.expander(f"Query: {r.get('query', '')[:50]}..."):
            st.write(f"**Score:** {r.get('score', 0):.2f}")
            st.write(f"**Status:** {r.get('status', 'unknown')}")
            for fb in r.get("feedback", []):
                st.write(f"- {fb}")


def render_timeline_tab(data: Dict[str, Any]):
    """Render evaluation timeline."""
    st.header("📅 Evaluation Timeline")
    
    reports = data.get("reports", [])
    if not reports:
        st.info("No timeline data available.")
        return
    
    # Create timeline data
    timeline_data = []
    for r in data.get("reports", []):
        mtime = r.get("_mtime")
        if mtime:
            timeline_data.append({
                "Time": mtime,
                "Query": r.get("query", "")[:50],
                "Status": r.get("status", "unknown"),
                "Score": r.get("score", 0),
                "Retries": r.get("retry_count", 0),
                "Tools": ", ".join(set(tc.get("name", "") for tc in r.get("tool_calls", [])))
            })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data).sort_values("Time", ascending=False)
        
        # Status color mapping
        status_colors = {
            "pass": "green",
            "fail": "red",
            "review": "orange"
        }
        
        st.dataframe(
            df.style.format({"Score": "{:.2f}"}).applymap(
                lambda x: f"color: {status_colors.get(x, 'black')}" if x in status_colors else "",
                subset=["Status"]
            ),
            use_container_width=True,
            height=400
        )
        
        # Timeline chart
        fig = px.scatter(
            df, 
            x="Time", 
            y="Score",
            color="Status",
            size="Retries",
            hover_data=["Query", "Tools"],
            title="Evaluation Scores Over Time",
            color_discrete_map={"pass": "#28a745", "fail": "#dc3545", "review": "#ffc107"}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_sidebar():
    """Render sidebar with controls."""
    st.sidebar.title("🏮 Tiehua Evaluation Dashboard")
    
    # Data source selection
    data_dir = st.sidebar.text_input(
        "Evaluation Data Directory",
        value="evaluation/reports",
        help="Directory containing evaluation JSON reports"
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        st.rerun()
    
    # Filter options
    st.sidebar.subheader("Filters")
    
    # Load data to get filter options
    data = load_evaluation_data()
    reports = data.get("reports", [])
    
    if reports:
        statuses = list(set(r.get("status", "unknown") for r in data.get("reports", [])))
        status_filter = st.sidebar.multiselect(
            "Status Filter",
            options=statuses,
            default=statuses
        )
        
        categories = list(set(r.get("category", "unknown") for r in reports if r.get("category")))
        if categories:
            category_filter = st.sidebar.multiselect(
                "Category Filter",
                options=categories,
                default=categories
            )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    return data_dir


def main():
    """Main dashboard entry point."""
    st.title("🏮 Tiehua Tourism Agent - Evaluation Dashboard")
    st.markdown("*AI Agent Evaluation & Safety Monitoring Dashboard*")
    
    # Sidebar
    data_dir = render_sidebar()
    
    # Load data
    data = load_evaluation_data()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🔧 Tool Usage", 
        "🛡️ Safety", 
        "🤖 LLM Judge", 
        "📅 Timeline",
        "⚙️ Settings"
    ])
    
    with tab1:
        render_overview_tab(data)
    
    with tab2:
        render_tool_usage_tab(data)
    
    with tab3:
        render_safety_tab(data)
    
    with tab4:
        render_llm_judge_tab(data)
    
    with tab5:
        render_timeline_tab(data)
    
    with tab6:
        st.header("⚙️ Settings")
        st.info("Settings panel - configure dashboard preferences")
        
        st.subheader("Data Sources")
        st.code("Evaluation reports loaded from: evaluation/reports/")
        
        st.subheader("Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Pass Threshold", value=0.8, min_value=0.0, max_value=1.0, step=0.1)
            st.number_input("Max Retries", value=2, min_value=0, max_value=5)
        with col2:
            st.number_input("Safety Threshold", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
            st.number_input("Max Retries", value=2, min_value=0, max_value=5)
        
        st.subheader("Export")
        if st.button("📥 Export Report (JSON)"):
            data = load_evaluation_data()
            st.download_button(
                label="Download JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()