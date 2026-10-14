"""
Evaluation Dashboard for Tiehua Tourism Agent Harness.

Streamlit-based dashboard for visualizing agent evaluation results,
LLM Judge scores, safety metrics, and agent performance metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import math


def is_valid_score(value: Any) -> bool:
    """Check if a value is a valid numeric score (not bool, NaN, or infinite).
    
    Expects score in range [0.0, 1.0] for evaluation scores.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        score = float(value)
    except (ValueError, OverflowError):
        return False
    return math.isfinite(score) and 0.0 <= score <= 1.0


def is_valid_retry(value: Any) -> bool:
    """Check if a value is a valid retry count (non-negative integer).
    
    Rejects non-integer floats (e.g., 1.5) and negative values.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and (pd.isna(value) or not math.isfinite(value)):
        return False
    # Reject non-integer floats (e.g., 1.5, 2.7)
    if isinstance(value, float) and not value.is_integer():
        return False
    return int(value) >= 0


def is_valid_safety_violations(value: Any) -> bool:
    """Check if safety_violations is a valid list."""
    return isinstance(value, list)


def normalize_status(value: Any) -> str:
    """Normalize status value to lowercase string.
    
    Non-string values become "unknown". Empty strings become "unknown".
    """
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    return value if value else "unknown"


def safe_text(value: Any, default: str = "unknown") -> str:
    """Safely convert value to stripped string with default for non-strings/empty.
    
    Args:
        value: Any value to convert to text
        default: Default string if value is not a valid string
        
    Returns:
        Stripped string or default
    """
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


def calculate_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from a list of reports.
    
    Args:
        reports: List of evaluation report dictionaries
        
    Returns:
        Dictionary with summary statistics
    """
    if not reports:
        return {}
    
    # Initialize counters
    status_counts = {"pass": 0, "fail": 0, "error": 0, "unknown": 0, "review": 0}
    scores = []
    retry_counts = []
    tool_usage = {}
    safety_violations = 0
    llm_judge_reports = 0
    reports_with_safety_data = 0
    reports_with_violations = 0
    
    for report in reports:
        if not isinstance(report, dict):
            continue
            
        # Count statuses (normalize various status formats)
        status = normalize_status(report.get("status"))
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["unknown"] += 1
        
        # Collect valid scores
        score = report.get("score")
        if is_valid_score(score):
            scores.append(float(score))
        
        # Collect valid retry counts
        retry = report.get("retry_count")
        if is_valid_retry(retry):
            retry_counts.append(int(retry))
        
        # Count tool usage (validate tool_calls is a list)
        tool_calls = report.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tool = tc.get("name")
                    if isinstance(tool, str) and tool and tool != "unknown":
                        tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        # Count safety violations - validate safety_violations is a list
        safety_violations_data = report.get("safety_violations")
        if is_valid_safety_violations(safety_violations_data):
            reports_with_safety_data += 1
            # Count only dict violations
            valid_violations = [v for v in safety_violations_data if isinstance(v, dict)]
            violation_count = len(valid_violations)
            safety_violations += violation_count
            if violation_count > 0:
                reports_with_violations += 1
        
        # Count LLM Judge reports
        if report.get("evaluator") == "llm_judge":
            llm_judge_reports += 1
    
    total_runs = len(reports)
    reports_without_safety_data = total_runs - reports_with_safety_data
    
    return {
        "total_runs": total_runs,
        "passed": status_counts["pass"],
        "failed": status_counts["fail"],
        "errors": status_counts["error"],
        "unknown": status_counts["unknown"],
        "review": status_counts["review"],
        "pass_rate": (status_counts["pass"] / total_runs * 100) if total_runs > 0 else 0,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "avg_retry": sum(retry_counts) / len(retry_counts) if retry_counts else 0,
        "tool_usage": tool_usage,
        "safety_violations": safety_violations,
        "reports_with_safety_data": reports_with_safety_data,
        "reports_with_violations": reports_with_violations,
        "reports_without_safety_data": reports_without_safety_data,
        "llm_judge_reports": llm_judge_reports,
        "valid_scores_count": len(scores),
        "valid_retry_count": len(retry_counts)
    }


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
    .status-error { color: #6f42c1; font-weight: bold; }
    .status-unknown { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_evaluation_data(data_dir: str = "evaluation/reports") -> Dict[str, Any]:
    """Load evaluation reports from JSON files.
    
    Args:
        data_dir: Directory containing evaluation JSON reports
        
    Returns:
        Dictionary with reports list and summary statistics
    """
    reports_dir = Path(data_dir)
    if not reports_dir.exists():
        return {"reports": [], "summary": {}}
    
    data = {
        "reports": [],
        "summary": {}
    }
    
    # Load all JSON files
    for report_file in reports_dir.glob("*.json"):
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
                
                # Validate root is a dictionary
                if not isinstance(report, dict):
                    st.warning(f"Skipping invalid report (root not object): {report_file.name}")
                    continue
                
                # Add metadata without modifying original report
                report_with_meta = report.copy()
                report_with_meta["_file"] = report_file.name
                report_with_meta["_mtime"] = datetime.fromtimestamp(report_file.stat().st_mtime)
                
                data["reports"].append(report_with_meta)
        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse JSON in {report_file}: {e}")
        except Exception as e:
            st.warning(f"Failed to load {report_file}: {e}")
    
    # Generate summary statistics from valid data only
    if data["reports"]:
        data["summary"] = calculate_summary(data["reports"])
    
    return data


def apply_filters(reports: List[Dict], status_filter: List[str], category_filter: List[str]) -> List[Dict]:
    """Apply status and category filters to reports.
    
    Args:
        reports: List of evaluation reports
        status_filter: List of status values to include (empty = all)
        category_filter: List of category values to include (empty = all)
        
    Returns:
        Filtered list of reports
    """
    if not reports:
        return []
    
    filtered = []
    
    # Normalize status filter values
    normalized_statuses = {normalize_status(s) for s in status_filter} if status_filter else set()
    
    for report in reports:
        if not isinstance(report, dict):
            continue
        
        # Normalize status
        status = normalize_status(report.get("status"))
        
        # Apply status filter
        if normalized_statuses and status not in normalized_statuses:
            continue
        
        # Apply category filter
        category = report.get("category", "unknown")
        if category_filter and category not in category_filter:
            continue
        
        filtered.append(report)
    
    return filtered


def render_overview_tab(data: Dict[str, Any]):
    """Render overview tab with key metrics.
    
    Args:
        data: Loaded evaluation data
    """
    st.header("📊 Overview")
    
    pass_threshold = st.session_state.get("pass_threshold", 0.8)
    max_retries = st.session_state.get("max_retries", 2)
    
    if not data.get("reports"):
        st.info("No evaluation data found. Run evaluations to generate reports.")
        return
    
    summary = data.get("summary", {})
    reports = data.get("reports", [])
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Runs",
            value=summary.get("total_runs", 0),
            help="Total number of evaluation reports"
        )
    
    with col2:
        pass_rate = summary.get("pass_rate", 0)
        st.metric(
            label="Pass Rate",
            value=f"{pass_rate:.1f}%",
            delta=f"{summary.get('passed', 0)} passed / {summary.get('failed', 0)} failed",
            help=f"Percentage of reports with status 'pass' (threshold: {pass_threshold})"
        )
    
    with col3:
        avg_score = summary.get("avg_score", 0)
        valid_scores = summary.get("valid_scores_count", 0)
        score_color = "normal" if avg_score >= pass_threshold else "inverse"
        st.metric(
            label="Avg Score",
            value=f"{avg_score:.2f}",
            delta=f"Target: ≥ {pass_threshold} ({valid_scores} valid scores)",
            delta_color=score_color,
            help=f"Average score from {valid_scores} reports with valid scores"
        )
    
    with col4:
        avg_retry = summary.get("avg_retry", 0)
        valid_retries = summary.get("valid_retry_count", 0)
        retry_color = "normal" if avg_retry <= max_retries else "inverse"
        st.metric(
            label="Avg Retries",
            value=f"{avg_retry:.1f}",
            delta=f"Max: {max_retries} ({valid_retries} valid counts)",
            delta_color=retry_color,
            help=f"Average retry count from {valid_retries} reports with valid retry counts"
        )
    
    with col5:
        safety_violations = summary.get("safety_violations", 0)
        st.metric(
            label="Safety Violations",
            value=safety_violations,
            delta="Target: 0",
            delta_color="inverse" if safety_violations > 0 else "normal",
            help="Total number of safety violations across all reports"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        if reports:
            status_counts = {}
            for r in reports:
                status = normalize_status(r.get("status"))
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                df_status = pd.DataFrame([
                    {"status": status, "count": count}
                    for status, count in status_counts.items()
                ])
                
                # Define colors for statuses
                color_map = {
                    "pass": "#28a745",
                    "fail": "#dc3545", 
                    "error": "#6f42c1",
                    "unknown": "#6c757d",
                    "review": "#ffc107"
                }
                
                fig = px.pie(
                    df_status, 
                    values="count", 
                    names="status",
                    title="Evaluation Status Distribution",
                    color="status",
                    color_discrete_map={k: v for k, v in color_map.items() if k in status_counts}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score distribution - use is_valid_score for validation
        scores = [float(r.get("score")) for r in reports if is_valid_score(r.get("score"))]
        if scores:
            fig = px.histogram(
                x=scores,
                nbins=20,
                title="Score Distribution",
                labels={"x": "Score", "y": "Count"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig.add_vline(x=pass_threshold, line_dash="dash", line_color="red", 
                        annotation_text=f"Target: {pass_threshold}", annotation_position="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid score data available for distribution chart")


def render_tool_usage_tab(data: Dict[str, Any]):
    """Render tool usage analytics tab.
    
    Args:
        data: Loaded evaluation data
    """
    st.header("🔧 Tool Usage Analytics")
    
    tool_usage = data.get("summary", {}).get("tool_usage", {})
    
    if not tool_usage:
        st.info("No tool usage data available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        if tool_usage:
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
            fig.update_layout(height=max(300, len(tool_usage) * 25))  # Dynamic height
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart
        if tool_usage:
            fig = px.pie(
                values=list(tool_usage.values()),
                names=list(tool_usage.keys()),
                title="Tool Usage Distribution"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    # Tool performance table (if data available)
    st.subheader("Tool Usage Details")
    if tool_usage:
        df_tools = pd.DataFrame([
            {"Tool": tool, "Usage Count": count}
            for tool, count in tool_usage.items()
        ]).sort_values("Usage Count", ascending=False)
        st.dataframe(df_tools, use_container_width=True, hide_index=True)
    else:
        st.info("No tool usage data to display")


def render_safety_tab(data: Dict[str, Any]):
    """Render safety metrics tab.
    
    Args:
        data: Loaded evaluation data
    """
    st.header("🛡️ Safety Metrics")
    
    reports = data.get("reports", [])
    summary = data.get("summary", {})
    if not reports:
        st.info("No safety data available.")
        return
    
    # Safety violations breakdown - validate each violation entry
    violation_types = {}
    violation_severities = {}
    total_violations = 0
    
    for r in reports:
        safety_violations = r.get("safety_violations")
        if not is_valid_safety_violations(safety_violations):
            continue
        for v in safety_violations:
            if not isinstance(v, dict):
                continue
            total_violations += 1
            vtype = safe_text(v.get("type"))
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
            
            severity = safe_text(v.get("severity")).lower()
            violation_severities[severity] = violation_severities.get(severity, 0) + 1
    
    # Display metrics using summary data (pre-computed with validation)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Safety Violations", summary.get("safety_violations", 0))
    with col2:
        st.metric("Reports with Violations", summary.get("reports_with_violations", 0))
    with col3:
        st.metric("Reports with Safety Data", summary.get("reports_with_safety_data", 0))
    
    # Additional info row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reports Missing Safety Data", summary.get("reports_without_safety_data", 0))
    with col2:
        st.metric("Total Reports", summary.get("total_runs", len(reports)))
    with col3:
        pass  # Placeholder for future metric
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if violation_types:
            fig = px.bar(
                x=list(violation_types.values()),
                y=list(violation_types.keys()),
                orientation="h",
                title="Safety Violations by Type",
                labels={"x": "Count", "y": "Violation Type"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No safety violations found")
    
    with col2:
        if violation_severities:
            # Define severity colors
            severity_colors = {
                "critical": "#dc3545",
                "high": "#fd7e14",
                "medium": "#ffc107",
                "low": "#28a745",
                "unknown": "#6c757d"
            }
            
            fig = px.pie(
                values=list(violation_severities.values()),
                names=list(violation_severities.keys()),
                title="Violations by Severity",
                color_discrete_map={k: v for k, v in severity_colors.items() if k in violation_severities}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No safety violation severity data")
    
    # Safety violations detail table
    violations_exist = total_violations > 0
    if violations_exist:
        st.subheader("Safety Violations Detail")
        violations_data = []
        for r in reports:
            report_id = r.get("_file", "unknown").replace(".json", "")
            safety_violations = r.get("safety_violations")
            if not is_valid_safety_violations(safety_violations):
                continue
            for v in safety_violations:
                if not isinstance(v, dict):
                    continue
                
                # Use safe_text for all violation fields
                vtype = safe_text(v.get("type"))
                severity = safe_text(v.get("severity"))
                message = safe_text(v.get("message"), default="")
                action = safe_text(v.get("suggested_action"))
                
                violations_data.append({
                    "Report ID": report_id[:20] + "..." if len(report_id) > 20 else report_id,
                    "Type": vtype,
                    "Severity": severity,
                    "Message": message[:100] + "..." if len(message) > 100 else message,
                    "Action": action
                })
        
        if violations_data:
            st.dataframe(pd.DataFrame(violations_data), use_container_width=True, hide_index=True)
        else:
            st.info("No detailed safety violation data available")
    else:
        st.info("No safety violations recorded in the evaluation reports")


def render_llm_judge_tab(data: Dict[str, Any]):
    """Render LLM Judge evaluation tab.
    
    Args:
        data: Loaded evaluation data
    """
    st.header("🤖 LLM Judge Evaluations")
    
    pass_threshold = st.session_state.get("pass_threshold", 0.8)
    
    reports = data.get("reports", [])
    llm_judge_reports = [r for r in reports if r.get("evaluator") == "llm_judge"]
    
    if not llm_judge_reports:
        st.info("No LLM Judge evaluations found.")
        return
    
    # LLM Judge metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("LLM Judge Reports", len(llm_judge_reports))
    
    with col2:
        passed_llm = sum(1 for r in llm_judge_reports if r.get("status") == "pass")
        st.metric("Passed", passed_llm)
    
    with col3:
        failed_llm = sum(1 for r in llm_judge_reports if r.get("status") == "fail")
        st.metric("Failed", failed_llm)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution for LLM Judge - use is_valid_score for validation
        llm_scores = [float(r.get("score")) for r in llm_judge_reports if is_valid_score(r.get("score"))]
        if llm_scores:
            fig = px.histogram(
                x=llm_scores,
                nbins=10,
                title="LLM Judge Score Distribution",
                labels={"x": "Score", "y": "Count"}
            )
            fig.add_vline(x=pass_threshold, line_dash="dash", line_color="red", 
                        annotation_text=f"Target: {pass_threshold}", annotation_position="top")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid LLM Judge scores available")
    
    with col2:
        # Status breakdown for LLM Judge
        llm_status_counts = {}
        for r in llm_judge_reports:
            status = normalize_status(r.get("status"))
            llm_status_counts[status] = llm_status_counts.get(status, 0) + 1
        
        if llm_status_counts:
            df_llm_status = pd.DataFrame([
                {"status": status, "count": count}
                for status, count in llm_status_counts.items()
            ])
            
            fig = px.pie(
                df_llm_status,
                values="count",
                names="status",
                title="LLM Judge Status Distribution",
                color="status",
                color_discrete_map={
                    "pass": "#28a745",
                    "fail": "#dc3545",
                    "error": "#6f42c1",
                    "unknown": "#6c757d"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LLM Judge status data")
    
    # Judge feedback
    st.subheader("LLM Judge Feedback")
    for i, r in enumerate(llm_judge_reports):
        query = r.get("query")
        query_preview = safe_text(query, default="")[:50]
        if len(safe_text(query, default="")) > 50:
            query_preview += "..."
        with st.expander(f"Evaluation {i+1}: {query_preview}"):
            col1, col2 = st.columns(2)
            with col1:
                # Safe score display with type checking
                score = r.get("score")
                if is_valid_score(score):
                    st.write(f"**Score:** {float(score):.2f}")
                else:
                    st.write("**Score:** N/A")
                st.write(f"**Status:** {r.get('status', 'unknown')}")
                st.write(f"**Evaluator:** {r.get('evaluator', 'unknown')}")
            with col2:
                # Safe retry display with type checking
                retry = r.get("retry_count")
                if is_valid_retry(retry):
                    st.write(f"**Retry:** {int(retry)}")
                else:
                    st.write("**Retry:** N/A")
                st.write(f"**File:** {r.get('_file', 'unknown')}")
            
            # Handle feedback (could be string, list, or missing)
            feedback = r.get("feedback", [])
            if isinstance(feedback, str):
                feedback = [feedback]
            elif not isinstance(feedback, list):
                feedback = []
            
            if feedback:
                st.write("**Feedback:**")
                for fb in feedback:
                    st.write(f"- {fb}")
            else:
                st.write("*No feedback available*")


def render_timeline_tab(data: Dict[str, Any]):
    """Render evaluation timeline tab.
    
    Args:
        data: Loaded evaluation data
    """
    st.header("📅 Evaluation Timeline")
    
    reports = data.get("reports", [])
    if not reports:
        st.info("No timeline data available.")
        return
    
    # Create timeline data with valid timestamps and cleaned values
    timeline_data = []
    for r in reports:
        mtime = r.get("_mtime")
        if not mtime:
            continue
        
        # Validate and clean score
        score = r.get("score")
        if is_valid_score(score):
            score_val = float(score)
        else:
            score_val = None
        
        # Validate and clean retry count
        retry = r.get("retry_count")
        if is_valid_retry(retry):
            retry_val = int(retry)
        else:
            retry_val = 0
        
        # Validate and normalize status
        status_val = normalize_status(r.get("status"))
        
        # Safely extract tools
        tools_list = []
        tool_calls = r.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tool_name = tc.get("name")
                    if tool_name and isinstance(tool_name, str):
                        tools_list.append(tool_name)
        
        timeline_data.append({
            "Time": mtime,
            "Query": r.get("query", "")[:50] if isinstance(r.get("query"), str) else "",
            "Status": status_val,
            "Score": score_val,
            "Retries": retry_val,
            "Tools": ", ".join(sorted(set(tools_list))) if tools_list else "None",
            "File": r.get("_file", "unknown")
        })
    
    if not timeline_data:
        st.info("No reports with valid timestamps found.")
        return
    
    # Sort by time (newest first)
    df = pd.DataFrame(timeline_data).sort_values("Time", ascending=False)
    
    # Status color mapping
    status_colors = {
        "pass": "#28a745",
        "fail": "#dc3545",
        "error": "#6f42c1",
        "unknown": "#6c757d",
        "review": "#ffc107"
    }
    
    # Display timeline table
    st.subheader("Recent Evaluations")
    if not df.empty:
        # Style the dataframe - handle None scores in formatting
        def color_status(val):
            color = status_colors.get(val.lower(), "#000000")
            return f"color: {color}; font-weight: bold"
        
        def format_score(val):
            if val is None:
                return "N/A"
            return f"{val:.2f}"
        
        styled_df = df.style.format({"Score": format_score}).applymap(
            color_status, subset=["Status"]
        )
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Timeline chart - only use rows with valid scores
    st.subheader("Scores Over Time")
    chart_data = df[df["Score"].notna()].copy()
    if not chart_data.empty:
        fig = px.scatter(
            chart_data, 
            x="Time", 
            y="Score",
            color="Status",
            size="Retries",
            hover_data=["Query", "Tools", "File"],
            title="Evaluation Scores Over Time",
            color_discrete_map=status_colors
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Score"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid score data for timeline chart")


def render_sidebar():
    """Render sidebar with controls and return selected data directory.
    
    Returns:
        Selected data directory path
    """
    st.sidebar.title("🏮 Tiehua Evaluation Dashboard")
    
    # Data source selection
    data_dir = st.sidebar.text_input(
        "Evaluation Data Directory",
        value="evaluation/reports",
        help="Directory containing evaluation JSON reports"
    )
    
    # Auto-refresh controls
    st.sidebar.subheader("Refresh")
    
    # Initialize auto_refresh in session state
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh (30s)", 
        value=st.session_state.auto_refresh,
        key="auto_refresh"
    )
    
    # Handle auto-refresh using streamlit-autorefresh if available
    if st.session_state.auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            # Trigger refresh every 30 seconds
            st_autorefresh(interval=30 * 1000, key="data_refresh")
            st.sidebar.caption("Auto-refresh active - refreshes every 30s")
        except ImportError:
            st.sidebar.caption("Auto-refresh requires 'streamlit-autorefresh' package")
            st.sidebar.caption("Install with: pip install streamlit-autorefresh")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Filter options
    st.sidebar.subheader("Filters")
    
    # Initialize filter session state - use None to track first initialization
    if "filters_initialized" not in st.session_state:
        st.session_state.filters_initialized = False
    
    # Load data to get filter options (cached for 5 seconds)
    @st.cache_data(ttl=5)
    def get_filter_options(data_dir_param):
        temp_data = load_evaluation_data(data_dir_param)
        return temp_data.get("reports", [])
    
    reports_for_filters = get_filter_options(data_dir)
    
    if reports_for_filters:
        # Get unique statuses and categories (normalize status values)
        statuses = set()
        for r in reports_for_filters:
            statuses.add(normalize_status(r.get("status")))
        
        categories = set()
        for r in reports_for_filters:
            cat = r.get("category")
            if cat and isinstance(cat, str):
                categories.add(cat)
        
        # Initialize filters on first run
        if not st.session_state.filters_initialized:
            st.session_state.status_filter = sorted(statuses)
            st.session_state.category_filter = sorted(categories) if categories else []
            st.session_state.filters_initialized = True
        
        # Status filter - use session state for persistence
        status_filter = st.sidebar.multiselect(
            "Status Filter",
            options=sorted(statuses),
            default=st.session_state.status_filter,
            key="status_filter",
            help="Filter reports by status (empty = all)"
        )
        
        # Category filter - use session state for persistence
        if categories:
            category_filter = st.sidebar.multiselect(
                "Category Filter",
                options=sorted(categories),
                default=st.session_state.category_filter,
                key="category_filter",
                help="Filter reports by category (empty = all)"
            )
        else:
            category_filter = []
    else:
        status_filter = []
        category_filter = []
        st.sidebar.info("No data available for filter options")
    
    return data_dir


def render_settings_tab():
    """Render settings tab with configurable options."""
    st.header("⚙️ Settings")
    
    st.info("Configure dashboard behavior and evaluation thresholds")
    
    # Threshold settings
    st.subheader("Evaluation Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input(
            "Pass Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("pass_threshold", 0.8),
            step=0.05,
            key="pass_threshold",
            help="Minimum score required to consider an evaluation as passing"
        )
        
        st.number_input(
            "Maximum Retry Count",
            min_value=0,
            max_value=10,
            value=st.session_state.get("max_retries", 2),
            step=1,
            key="max_retries",
            help="Maximum number of retries allowed before marking as failed"
        )
    
    with col2:
        st.info("**Current Settings**")
        st.write(f"- Pass Threshold: {st.session_state.get('pass_threshold', 0.8)}")
        st.write(f"- Max Retries: {st.session_state.get('max_retries', 2)}")
        
        st.caption("Changes apply immediately to Overview, LLM Judge, and other tabs.")
    
    # Data source info
    st.subheader("Data Sources")
    st.code("Evaluation reports loaded from: evaluation/reports/")
    
    # Export settings
    st.subheader("Export Options")
    st.write("JSON export includes:")
    st.write("- All loaded evaluation reports")
    st.write("- Summary statistics")
    st.write("- Export timestamp")
    

def render_export_button(filtered_data: Dict[str, Any], raw_data: Dict[str, Any]):
    """Render JSON export button with options for filtered vs all data.
    
    Args:
        filtered_data: Currently filtered evaluation data
        raw_data: Complete unfiltered evaluation data
    """
    st.subheader("📥 Export Data")
    
    filtered_count = len(filtered_data.get("reports", []))
    total_count = len(raw_data.get("reports", []))
    
    if filtered_count == 0 and total_count == 0:
        st.warning("No data available to export")
        return
    
    # Show export info
    if filtered_count != total_count:
        st.caption(f"Current view: {filtered_count} of {total_count} reports (filtered)")
    else:
        st.caption(f"Total reports available: {total_count}")
    
    # Export options
    export_mode = st.radio(
        "Export Scope",
        options=["Filtered Results", "All Reports"],
        index=0 if filtered_count != total_count else 1,
        horizontal=True,
        help="Choose whether to export only the filtered results or all loaded reports"
    )
    
    export_data_source = filtered_data if export_mode == "Filtered Results" else raw_data
    export_count = len(export_data_source.get("reports", []))
    
    st.write(f"**Reports to export:** {export_count}")
    
    if st.button("Generate Export", type="primary", disabled=export_count == 0):
        if export_count > 0:
            # Prepare export data
            export_data = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_reports": export_count,
                    "export_scope": export_mode,
                    "export_version": "1.0"
                },
                "summary": export_data_source.get("summary", {}),
                "reports": export_data_source.get("reports", [])
            }
            
            # Convert to JSON
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2, default=str)
            
            # Create download button
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            scope_suffix = "filtered" if export_mode == "Filtered Results" else "all"
            filename = f"evaluation_export_{scope_suffix}_{timestamp}.json"
            
            st.download_button(
                label="💾 Download JSON Report",
                data=json_str,
                file_name=filename,
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("No data available to export")


def main():
    """Main dashboard entry point."""
    st.title("🏮 Tiehua Tourism Agent - Evaluation Dashboard")
    st.markdown("*AI Agent Evaluation & Safety Monitoring Dashboard*")
    
    # Initialize session state for settings if not present
    if "pass_threshold" not in st.session_state:
        st.session_state.pass_threshold = 0.8
    if "max_retries" not in st.session_state:
        st.session_state.max_retries = 2
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "status_filter" not in st.session_state:
        st.session_state.status_filter = []
    if "category_filter" not in st.session_state:
        st.session_state.category_filter = []
    
    # Sidebar
    data_dir = render_sidebar()
    
    # Get filter values from session state (set by sidebar via widget keys)
    status_filter = st.session_state.get("status_filter", [])
    category_filter = st.session_state.get("category_filter", [])
    
    # Load data
    raw_data = load_evaluation_data(data_dir)
    
    # Apply filters
    if raw_data.get("reports"):
        filtered_reports = apply_filters(
            raw_data["reports"], 
            status_filter, 
            category_filter
        )
        
        # Create filtered data structure
        filtered_data = raw_data.copy()
        filtered_data["reports"] = filtered_reports
        
        # Regenerate summary for filtered data using unified function
        if filtered_reports:
            filtered_data["summary"] = calculate_summary(filtered_reports)
        else:
            # Empty filtered data
            filtered_data["summary"] = {}
    else:
        filtered_data = raw_data
    
# Display filter info
    if status_filter or category_filter:
        total_available = len(raw_data.get("reports", []))
        total_filtered = len(filtered_data.get("reports", []))
        if total_available != total_filtered:
            st.caption(f"Showing {total_filtered} of {total_available} reports (filtered)")
    
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
        render_overview_tab(filtered_data)
        
    with tab2:
        render_tool_usage_tab(filtered_data)
        
    with tab3:
        render_safety_tab(filtered_data)
        
    with tab4:
        render_llm_judge_tab(filtered_data)
        
    with tab5:
        render_timeline_tab(filtered_data)
    
    with tab6:
        render_settings_tab()
        render_export_button(filtered_data, raw_data)


if __name__ == "__main__":
    main()