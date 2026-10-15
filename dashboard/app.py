"""
Evaluation Dashboard for Tiehua Tourism Agent Harness.

Streamlit-based dashboard for visualizing agent evaluation results,
LLM Judge scores, safety metrics, and agent performance metrics.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# === Data Validation Helpers ===


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
    if isinstance(value, float) and not value.is_integer():
        return False
    return int(value) >= 0


def is_valid_safety_violations(value: Any) -> bool:
    """Check if safety_violations is a valid list."""
    return isinstance(value, list)


def normalize_status(value: Any) -> str:
    """Normalize status value to lowercase string.

    Maps common status variants to standard keys:
    - "passed" -> "pass"
    - "failed" -> "fail"
    Non-string values become "unknown". Empty strings become "unknown".
    """
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    if value == "passed":
        return "pass"
    if value == "failed":
        return "fail"
    return value if value else "unknown"


def safe_text(value: Any, default: str = "unknown") -> str:
    """Safely convert value to stripped string with default for non-strings/empty."""
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


def safe_list(value: Any) -> list:
    """Safely return a list from a value, returning [] if not a list."""
    return value if isinstance(value, list) else []


# === Data Processing (separated from UI rendering) ===


def calculate_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from a list of reports.

    Args:
        reports: List of evaluation report dictionaries

    Returns:
        Dictionary with summary statistics. Returns empty dict for empty input.
    """
    if not reports:
        return {}

    status_counts: Dict[str, int] = {}
    scores: List[float] = []
    retry_counts: List[int] = []
    tool_usage: Dict[str, int] = {}
    violation_types: Dict[str, int] = {}
    violation_severities: Dict[str, int] = {}
    safety_violations = 0
    reports_with_safety_data = 0
    reports_with_violations = 0
    llm_judge_reports = 0

    for report in reports:
        if not isinstance(report, dict):
            continue

        # Count statuses (normalize various status formats)
        status = normalize_status(report.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

        # Collect valid scores
        score = report.get("score")
        if is_valid_score(score):
            scores.append(float(score))

        # Collect valid retry counts
        retry = report.get("retry_count")
        if is_valid_retry(retry):
            retry_counts.append(int(retry))

        # Count tool usage
        for tc in safe_list(report.get("tool_calls")):
            if isinstance(tc, dict):
                tool = tc.get("name")
                if isinstance(tool, str) and tool and tool != "unknown":
                    tool_usage[tool] = tool_usage.get(tool, 0) + 1

        # Count safety violations
        sv = report.get("safety_violations")
        if is_valid_safety_violations(sv):
            reports_with_safety_data += 1
            for v in sv:
                if isinstance(v, dict):
                    safety_violations += 1
                    vtype = safe_text(v.get("type"))
                    violation_types[vtype] = violation_types.get(vtype, 0) + 1
                    severity = safe_text(v.get("severity"), default="unknown").lower()
                    violation_severities[severity] = violation_severities.get(severity, 0) + 1

            if len([v for v in sv if isinstance(v, dict)]) > 0:
                reports_with_violations += 1

        # Count LLM Judge reports
        if report.get("evaluator") == "llm_judge":
            llm_judge_reports += 1

    total = sum(1 for r in reports if isinstance(r, dict))

    return {
        "total_runs": total,
        "status_counts": status_counts,
        "passed": status_counts.get("pass", 0),
        "failed": status_counts.get("fail", 0),
        "errors": status_counts.get("error", 0),
        "unknown": status_counts.get("unknown", 0),
        "review": status_counts.get("review", 0),
        "pass_rate": status_counts.get("pass", 0) / total * 100 if total > 0 else 0,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "avg_retry": sum(retry_counts) / len(retry_counts) if retry_counts else 0,
        "tool_usage": tool_usage,
        "safety_violations": safety_violations,
        "reports_with_safety_data": reports_with_safety_data,
        "reports_with_violations": reports_with_violations,
        "reports_without_safety_data": total - reports_with_safety_data,
        "llm_judge_reports": llm_judge_reports,
        "violation_types": violation_types,
        "violation_severities": violation_severities,
        "valid_scores_count": len(scores),
        "valid_retry_count": len(retry_counts),
    }


def apply_filters(
    reports: List[Dict], status_filter: List[str], category_filter: List[str]
) -> List[Dict]:
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

    normalized_statuses = {normalize_status(s) for s in status_filter} if status_filter else set()

    filtered = []
    for report in reports:
        if not isinstance(report, dict):
            continue

        # Apply status filter
        status = normalize_status(report.get("status"))
        if normalized_statuses and status not in normalized_statuses:
            continue

        # Apply category filter
        category = report.get("category", "unknown")
        if category_filter and category not in category_filter:
            continue

        filtered.append(report)

    return filtered


def load_evaluation_data(data_dir: str) -> Dict[str, Any]:
    """Load evaluation reports from JSON files in a directory.

    Args:
        data_dir: Directory containing evaluation JSON reports

    Returns:
        Dictionary with reports list and summary statistics
    """
    reports_dir = Path(data_dir)
    if not reports_dir.exists():
        return {"reports": [], "summary": {}}

    reports: List[Dict[str, Any]] = []

    for report_file in reports_dir.glob("*.json"):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)

            if not isinstance(report, dict):
                st.warning(f"Skipping invalid report (root not object): {report_file.name}")
                continue

            if "results" in report and isinstance(report.get("results"), list):
                # This is a batch/aggregate report - unpack individual results
                batch_meta = {k: v for k, v in report.items() if k != "results"}
                for i, result in enumerate(report["results"]):
                    if isinstance(result, dict):
                        item = result.copy()
                        # Merge batch metadata if not present in individual result
                        for k, v in batch_meta.items():
                            if k not in item:
                                item[k] = v
                        item["_file"] = report_file.name
                        item["_mtime"] = datetime.fromtimestamp(report_file.stat().st_mtime)
                        item["_batch_index"] = i
                        reports.append(item)
            else:
                # Single report
                report_copy = report.copy()
                report_copy["_file"] = report_file.name
                report_copy["_mtime"] = datetime.fromtimestamp(report_file.stat().st_mtime)
                reports.append(report_copy)

        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse JSON in {report_file.name}: {e}")
        except Exception as e:
            st.warning(f"Failed to load {report_file.name}: {e}")

    return {"reports": reports, "summary": calculate_summary(reports) if reports else {}}


# === Chart Rendering Helpers (consolidated) ===

STATUS_COLOR_MAP = {
    "pass": "#28a745",
    "fail": "#dc3545",
    "error": "#6f42c1",
    "unknown": "#6c757d",
    "review": "#ffc107",
}

SEVERITY_COLOR_MAP = {
    "critical": "#dc3545",
    "high": "#fd7e14",
    "medium": "#ffc107",
    "low": "#28a745",
    "unknown": "#6c757d",
}


def render_pie_chart(data: Dict[str, int], title: str, color_map: Optional[Dict[str, str]] = None):
    """Render a pie chart from a status count dictionary."""
    if not data:
        st.info(f"No data for {title.lower()}")
        return

    df = pd.DataFrame([{"status": k, "count": v} for k, v in data.items()])
    fig = px.pie(
        df,
        values="count",
        names="status",
        title=title,
        color="status",
        color_discrete_map={k: v for k, v in (color_map or STATUS_COLOR_MAP).items() if k in data},
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, width="stretch")


def render_bar_chart(
    data: Dict[str, int], title: str, x_label: str = "Count", horizontal: bool = True, color_scale: str = "Blues"
):
    """Render a bar chart from a count dictionary."""
    if not data:
        st.info(f"No data for {title.lower()}")
        return

    df = pd.DataFrame([{"name": k, "count": v} for k, v in data.items()]).sort_values(
        "count", ascending=not horizontal
    )

    if horizontal:
        fig = px.bar(df, x="count", y="name", orientation="h", title=title, labels={"x": x_label, "y": ""})
    else:
        fig = px.bar(df, x="name", y="count", title=title, labels={"y": x_label, "x": ""},
                      color="count", color_continuous_scale=color_scale)
    st.plotly_chart(fig, width="stretch")


def render_histogram(values: List[float], title: str, pass_threshold: Optional[float] = None, nbins: int = 20):
    """Render a histogram with optional pass threshold line."""
    if not values:
        st.info(f"No valid data for {title.lower()}")
        return

    fig = px.histogram(
        x=values,
        nbins=nbins,
        title=title,
        labels={"x": "Score", "y": "Count"},
        color_discrete_sequence=["#1f77b4"],
    )
    if pass_threshold is not None:
        fig.add_vline(
            x=pass_threshold, line_dash="dash", line_color="red",
            annotation_text=f"Target: {pass_threshold}", annotation_position="top",
        )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")


def render_scatter_timeline(
    df: pd.DataFrame, x: str, y: str, color: str, title: str, color_map: Dict[str, str], y_range: tuple = (-0.05, 1.05)
):
    """Render a scatter/timeline chart."""
    if df.empty:
        st.info("No valid data for timeline chart")
        return

    fig = px.scatter(
        df, x=x, y=y, color=color, size_max=15,
        hover_data=[c for c in df.columns if c not in (x, y, color)],
        title=title, color_discrete_map=color_map,
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    fig.update_layout(
        xaxis_title=x, yaxis_title=y,
        yaxis=dict(range=list(y_range)),
    )
    st.plotly_chart(fig, width="stretch")


# === Session State Initialization ===


def init_session_state():
    """Initialize session state defaults."""
    defaults = {
        "pass_threshold": 0.8,
        "max_retries": 2,
        "auto_refresh": False,
        "status_filter": [],
        "category_filter": [],
        "filters_initialized": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# === UI Components ===


def render_overview_section(filtered_data: Dict[str, Any]):
    """Render Overview section: key metrics, status distribution, score distribution."""
    st.header("📊 Overview")

    pass_threshold = st.session_state.get("pass_threshold", 0.8)
    max_retries = st.session_state.get("max_retries", 2)

    reports = filtered_data.get("reports", [])
    summary = filtered_data.get("summary", {})

    if not reports:
        st.info("No evaluation data found. Run evaluations to generate reports.")
        return

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Runs", summary.get("total_runs", 0))

    with col2:
        pass_count = summary.get("passed", 0)
        fail_count = summary.get("failed", 0)
        st.metric(
            "Pass Rate",
            f"{summary.get('pass_rate', 0):.1f}%",
            delta=f"{pass_count} passed / {fail_count} failed",
        )

    with col3:
        avg_score = summary.get("avg_score", 0)
        valid_scores = summary.get("valid_scores_count", 0)
        st.metric(
            "Avg Score",
            f"{avg_score:.2f}",
            delta=f"Target: ≥ {pass_threshold} ({valid_scores} valid)",
            delta_color="normal" if avg_score >= pass_threshold else "inverse",
        )

    with col4:
        avg_retry = summary.get("avg_retry", 0)
        valid_retries = summary.get("valid_retry_count", 0)
        st.metric(
            "Avg Retries",
            f"{avg_retry:.1f}",
            delta=f"Max: {max_retries} ({valid_retries} valid)",
            delta_color="normal" if avg_retry <= max_retries else "inverse",
        )

    with col5:
        safety_violations = summary.get("safety_violations", 0)
        st.metric(
            "Safety Violations",
            safety_violations,
            delta="Target: 0",
            delta_color="inverse" if safety_violations > 0 else "normal",
        )

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        # Status distribution
        status_counts = summary.get("status_counts", {})
        if not status_counts and reports:
            # Fallback: calculate from reports directly
            status_counts = {}
            for r in reports:
                status = normalize_status(r.get("status"))
                status_counts[status] = status_counts.get(status, 0) + 1
        render_pie_chart(status_counts or {}, "Evaluation Status Distribution", STATUS_COLOR_MAP)

    with col2:
        # Score distribution
        scores = [float(r.get("score")) for r in reports if is_valid_score(r.get("score"))]
        render_histogram(scores, "Score Distribution", pass_threshold=pass_threshold, nbins=20)


def render_quality_safety_section(filtered_data: Dict[str, Any], raw_data: Dict[str, Any], data_dir: str):
    """Render Quality & Safety section: Tool Usage, Safety Metrics, LLM Judge."""
    st.header("🔍 Quality & Safety")

    reports = filtered_data.get("reports", [])
    summary = filtered_data.get("summary", {})

    if not reports:
        st.info("No data available. Apply different filters or load reports.")
        return

    tool_usage = summary.get("tool_usage", {})

    # === Tool Usage Analytics ===
    st.subheader("🔧 Tool Usage Analytics")
    if tool_usage:
        col1, col2 = st.columns(2)
        with col1:
            render_bar_chart(tool_usage, "Tool Usage Frequency", x_label="Count")
        with col2:
            render_pie_chart(tool_usage, "Tool Usage Distribution")
        df_tools = (
            pd.DataFrame([{"Tool": t, "Usage Count": c} for t, c in tool_usage.items()])
            .sort_values("Usage Count", ascending=False)
        )
        st.dataframe(df_tools, width="stretch", hide_index=True)
    else:
        st.info("No tool usage data available.")

    # === Safety Metrics ===
    st.subheader("🛡️ Safety Metrics")
    if reports:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Safety Violations", summary.get("safety_violations", 0))
        with col2:
            st.metric("Reports with Violations", summary.get("reports_with_violations", 0))
        with col3:
            st.metric("Reports Without Safety Data", summary.get("reports_without_safety_data", 0))

        col1, col2 = st.columns(2)
        with col1:
            violation_types = summary.get("violation_types", {})
            render_bar_chart(violation_types, "Safety Violations by Type", x_label="Count")

        with col2:
            violation_severities = summary.get("violation_severities", {})
            render_pie_chart(violation_severities, "Violations by Severity", SEVERITY_COLOR_MAP)

        # Violations detail table
        if summary.get("safety_violations", 0) > 0:
            st.subheader("Safety Violations Detail")
            violations_data = []
            for r in reports:
                report_id = r.get("_file", "unknown").replace(".json", "")
                for v in safe_list(r.get("safety_violations")):
                    if not isinstance(v, dict):
                        continue
                    violations_data.append({
                        "Report ID": report_id[:50],
                        "Type": safe_text(v.get("type")),
                        "Severity": safe_text(v.get("severity")),
                        "Message": safe_text(v.get("message"), default="")[:100],
                        "Action": safe_text(v.get("suggested_action"), default="")[:80],
                    })
            if violations_data:
                st.dataframe(pd.DataFrame(violations_data), width="stretch", hide_index=True)

    # === LLM Judge ===
    st.subheader("🤖 LLM Judge Evaluations")
    llm_judge_reports = [r for r in reports if r.get("evaluator") == "llm_judge"]

    if not llm_judge_reports:
        st.info("No LLM Judge evaluations found in filtered data.")
        return

    pass_threshold = st.session_state.get("pass_threshold", 0.8)

    # LLM Judge metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LLM Judge Reports", len(llm_judge_reports))

    llm_status_counts = {}
    for r in llm_judge_reports:
        status = normalize_status(r.get("status"))
        llm_status_counts[status] = llm_status_counts.get(status, 0) + 1

    with col2:
        st.metric("Passed", llm_status_counts.get("pass", 0))
    with col3:
        st.metric("Failed/Review", llm_status_counts.get("fail", 0) + llm_status_counts.get("review", 0))

    # LLM Judge charts
    col1, col2 = st.columns(2)
    with col1:
        llm_scores = [float(r.get("score")) for r in llm_judge_reports if is_valid_score(r.get("score"))]
        render_histogram(llm_scores, "LLM Judge Score Distribution", pass_threshold=pass_threshold, nbins=10)

    with col2:
        render_pie_chart(llm_status_counts, "LLM Judge Status Distribution")

    # Judge feedback
    st.subheader("LLM Judge Feedback")
    for i, r in enumerate(llm_judge_reports):
        query = r.get("query")
        query_text = safe_text(query, default="")
        query_preview = query_text[:50] + "..." if len(query_text) > 50 else query_text
        with st.expander(f"Evaluation {i + 1}: {query_preview}"):
            # Display the user query and agent's final answer
            st.write(f"**Query:** {query_text}")
            final_answer = safe_text(r.get("final_answer", getattr(r, "final_answer", "")), default="")
            if final_answer:
                st.write(f"**Final Answer:** {final_answer[:200]}{'...' if len(final_answer) > 200 else ''}")
            else:
                st.write("*No final answer recorded*")

            col1, col2 = st.columns(2)
            with col1:
                score = r.get("score")
                st.write(f"**Score:** {f'{float(score):.2f}' if is_valid_score(score) else 'N/A'}")
                st.write(f"**Status:** {normalize_status(r.get('status'))}")
                st.write(f"**Evaluator:** {r.get('evaluator', 'unknown')}")
                st.write(f"**Model:** {r.get('llm_judge_model', r.get('llm_judge_model', 'unknown'))}")
            with col2:
                retry = r.get("retry_count")
                st.write(f"**Retry:** {int(retry) if is_valid_retry(retry) else 'N/A'}")
                st.write(f"**File:** {r.get('_file', 'unknown')}")

            feedback = r.get("feedback", [])
            if isinstance(feedback, str):
                feedback = [feedback]
            elif not isinstance(feedback, list):
                feedback = []

            # Also check llm_judge_feedback
            llm_feedback = r.get("llm_judge_feedback", [])
            if isinstance(llm_feedback, str):
                llm_feedback = [llm_feedback]
            elif not isinstance(llm_feedback, list):
                llm_feedback = []

            all_feedback = feedback or llm_feedback
            if all_feedback:
                st.write("**Feedback:**")
                for fb in all_feedback:
                    if isinstance(fb, dict):
                        code = fb.get("code", "UNKNOWN")
                        severity = fb.get("severity", "info")
                        message = fb.get("message", "")
                        st.write(f"- **{code}** ({severity}): {message}")
                    else:
                        st.write(f"- {fb}")
            else:
                st.write("*No feedback available*")


def render_history_section(filtered_data: Dict[str, Any]):
    """Render History section: Evaluation Timeline, historical records, score trends."""
    st.header("📅 History")

    reports = filtered_data.get("reports", [])
    if not reports:
        st.info("No timeline data available.")
        return

    pass_threshold = st.session_state.get("pass_threshold", 0.8)

    # Build timeline data
    timeline_data = []
    for r in reports:
        mtime = r.get("_mtime")
        if not mtime:
            continue

        score = r.get("score")
        score_val = float(score) if is_valid_score(score) else None

        retry = r.get("retry_count")
        retry_val = int(retry) if is_valid_retry(retry) else 0

        tools_list = []
        for tc in safe_list(r.get("tool_calls")):
            if isinstance(tc, dict):
                tool_name = tc.get("name")
                if tool_name and isinstance(tool_name, str):
                    tools_list.append(tool_name)

        timeline_data.append({
            "Time": mtime,
            "Query": safe_text(r.get("query"), default="")[:50],
            "Status": normalize_status(r.get("status")),
            "Score": score_val,
            "Retries": retry_val,
            "Tools": ", ".join(sorted(set(tools_list))) if tools_list else "None",
            "Answer": safe_text(r.get("final_answer", getattr(r, "final_answer", "")), default="")[:50],
            "File": r.get("_file", "unknown"),
        })

    if not timeline_data:
        st.info("No reports with valid timestamps found.")
        return

    df = pd.DataFrame(timeline_data).sort_values("Time", ascending=False)

    # Timeline table
    st.subheader("Recent Evaluations")
    status_colors = STATUS_COLOR_MAP.copy()

    def color_status(val):
        color = status_colors.get(val.lower(), "#000000")
        return f"color: {color}; font-weight: bold"

    def format_score(val):
        return "N/A" if val is None else f"{val:.2f}"

    styled_df = (
        df.style.format({"Score": format_score})
        .map(color_status, subset=["Status"])
        .set_properties(**{"text-align": "left"})
    )
    st.dataframe(styled_df, width="stretch", height=400)

    # Score trend chart
    st.subheader("Scores Over Time")
    chart_data = df[df["Score"].notna()].copy()
    render_scatter_timeline(
        chart_data, x="Time", y="Score", color="Status",
        title="Evaluation Scores Over Time", color_map=status_colors,
    )


def render_settings_section(raw_data: Dict[str, Any], data_dir: str):
    """Render Settings & Export section: thresholds, Judge model, run evaluation, JSON export."""
    st.header("⚙️ Settings & Export")

    # === Evaluation Thresholds ===
    st.subheader("Evaluation Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Pass Score Threshold",
            min_value=0.0, max_value=1.0, value=st.session_state.get("pass_threshold", 0.8),
            step=0.05, key="pass_threshold",
            help="Minimum score required to consider an evaluation as passing",
        )
        st.number_input(
            "Maximum Retry Count",
            min_value=0, max_value=10, value=st.session_state.get("max_retries", 2),
            step=1, key="max_retries",
            help="Maximum number of retries allowed before marking as failed",
        )
    with col2:
        st.info("**Current Settings**")
        st.write(f"- Pass Threshold: {st.session_state.get('pass_threshold', 0.8)}")
        st.write(f"- Max Retries: {st.session_state.get('max_retries', 2)}")

    # === LLM Judge Settings ===
    st.subheader("🤖 LLM Judge Evaluation")
    try:
        from src.evaluation.llm_judge import LLMJudge
    except Exception as e:
        st.warning(f"Cannot load LLM Judge module: {e}")
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        st.success("✅ OPENROUTER_API_KEY configured")
    else:
        st.warning("⚠️ OPENROUTER_API_KEY not set - LLM Judge will not work")
        st.code("export OPENROUTER_API_KEY='your-key'")

    model_options = LLMJudge.MODEL_FALLBACK_CHAIN
    selected_model = st.selectbox(
        "Primary Model",
        options=model_options,
        index=0,
        help="First model to try. Falls back to next on failure.",
        key="llm_judge_model",
    )

    with st.expander("Fallback Chain (tried in order)"):
        for i, m in enumerate(model_options):
            st.write(f"{i + 1}. `{m}`")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, help="Lower = more deterministic")

    st.markdown("---")

    # Get reports that don't have LLM Judge evaluation
    raw_reports = raw_data.get("reports", [])
    reports_without_llm = [
        r for r in raw_reports
        if r.get("evaluator") != "llm_judge" and normalize_status(r.get("status")) != "error"
    ]

    st.write(f"**Reports without LLM Judge evaluation:** {len(reports_without_llm)}")

    if st.button("🚀 Run LLM Judge on Pending Reports", type="primary",
                 disabled=not api_key or len(reports_without_llm) == 0):
        if not api_key:
            st.error("OPENROUTER_API_KEY not configured")
        elif len(reports_without_llm) == 0:
            st.info("No pending reports to evaluate")
        else:
            run_llm_judge_on_reports(reports_without_llm, selected_model, temperature, api_key, data_dir)

    # === Export ===
    st.subheader("📥 Export Data")
    render_export_controls(raw_data)


def render_export_controls(raw_data: Dict[str, Any]):
    """Render JSON export download button."""
    reports = raw_data.get("reports", [])
    if not reports:
        st.warning("No data available to export")
        return

    st.caption(f"Total reports available: {len(reports)}")

    export_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "total_reports": len(reports),
            "export_version": "1.0",
        },
        "summary": raw_data.get("summary", {}),
        "reports": reports,
    }

    json_str = json.dumps(export_data, ensure_ascii=False, indent=2, default=str)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_export_all_{timestamp}.json"

    st.download_button(
        label="💾 Download JSON Report",
        data=json_str,
        file_name=filename,
        mime="application/json",
        width="stretch",
    )


def run_llm_judge_on_reports(
    reports: List[Dict], model: str, temperature: float, api_key: str, data_dir: str
):
    """Run LLM Judge evaluation on a list of reports, saving results to the specified data_dir."""
    from src.evaluation.llm_judge import LLMJudge
    from src.state import AgentState

    judge = LLMJudge(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    success_count = 0
    failed_count = 0

    reports_dir = Path(data_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for i, report in enumerate(reports):
        progress_bar.progress((i + 1) / len(reports))
        status_text.text(f"Evaluating {i + 1}/{len(reports)}: {report.get('_file', 'unknown')}")

        try:
            # Reconstruct AgentState from report
            state = AgentState(query=report.get("query", ""))
            state.answer = report.get("final_answer", "")
            state.tool_calls = report.get("tool_calls", [])
            state.tool_results = report.get("tool_results", [])
            state.language = report.get("language", "zh-TW")
            state.user_intent = report.get("category", "general")

            # Run LLM Judge
            judge_result = judge.evaluate(
                query=report.get("query", ""),
                state=state,
                expected=None,
            )

            # Update report with LLM Judge results (only evaluation fields, not status display)
            report["evaluator"] = "llm_judge"
            report["score"] = judge_result.score
            report["llm_judge_score"] = judge_result.score
            report["llm_judge_feedback"] = judge_result.issues
            report["llm_judge_reason"] = judge_result.reason
            report["llm_judge_model"] = judge_result.metadata.get("judge_model", model) if hasattr(judge_result, "metadata") else model

            # Store judge result in separate field instead of overwriting status
            # This ensures pass_threshold doesn't modify original evaluation results
            if judge_result.passed:
                report["llm_judge_result"] = "pass"
            else:
                report["llm_judge_result"] = "review"

            # Save updated report to the user's selected directory
            report_filename = report.get("_file", f"report_{i}.json")
            report_path = reports_dir / report_filename
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    save_report = {k: v for k, v in report.items() if not k.startswith("_")}
                    json.dump(save_report, f, ensure_ascii=False, indent=2, default=str)
                success_count += 1
                with results_container:
                    st.success(f"✅ {report_filename}: score={judge_result.score:.2f}, passed={judge_result.passed}")
            except (IOError, OSError) as e:
                failed_count += 1
                with results_container:
                    st.error(f"❌ Failed to save {report_filename}: {e}")

        except Exception as e:
            failed_count += 1
            with results_container:
                st.error(f"❌ {report.get('_file', 'unknown')}: {e}")

    status_text.text(f"Completed: {success_count} success, {failed_count} failed")
    st.cache_data.clear()
    st.rerun()


# === Sidebar ===


def render_sidebar():
    """Render sidebar with controls and return selected data directory."""
    st.sidebar.title("🏮 Tiehua Evaluation Dashboard")

    data_dir = st.sidebar.text_input(
        "Evaluation Data Directory",
        value="evaluation/reports",
        help="Directory containing evaluation JSON reports",
        key="data_dir",
    )

    # Auto-refresh controls
    st.sidebar.subheader("Refresh")

    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh (30s)",
        value=st.session_state.get("auto_refresh", False),
        key="auto_refresh",
    )

    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=30 * 1000, key="data_refresh")
            st.sidebar.caption("Auto-refresh active - refreshes every 30s")
        except ImportError:
            st.sidebar.caption("Auto-refresh requires 'streamlit-autorefresh' package")
            st.sidebar.caption("Install with: pip install streamlit-autorefresh")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Filter options
    st.sidebar.subheader("Filters")

    # Load data to get filter options
    reports_for_filters = load_evaluation_data(data_dir).get("reports", [])

    if reports_for_filters:
        statuses = set()
        for r in reports_for_filters:
            statuses.add(normalize_status(r.get("status")))

        categories = set()
        for r in reports_for_filters:
            cat = r.get("category")
            if cat and isinstance(cat, str):
                categories.add(cat)

        if not st.session_state.get("filters_initialized"):
            st.session_state.status_filter = sorted(statuses)
            st.session_state.category_filter = sorted(categories) if categories else []
            st.session_state.filters_initialized = True

        st.sidebar.multiselect(
            "Status Filter",
            options=sorted(statuses),
            key="status_filter",
            help="Filter reports by status (empty = all)",
        )

        if categories:
            st.sidebar.multiselect(
                "Category Filter",
                options=sorted(categories),
                key="category_filter",
                help="Filter reports by category (empty = all)",
            )
    else:
        st.sidebar.info("No data available for filter options")
    return data_dir








# === Main ===


@st.cache_data(ttl=30)
def load_evaluation_data_cached(data_dir: str) -> Dict[str, Any]:
    """Cached version of load_evaluation_data for automatic refresh."""
    return load_evaluation_data(data_dir)


def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="Tiehua Tourism Agent - Evaluation Dashboard",
        page_icon="🏮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏮 Tiehua Tourism Agent - Evaluation Dashboard")
    st.caption("AI Agent Evaluation & Safety Monitoring Dashboard")

    init_session_state()
    data_dir = render_sidebar()

    # Get filter values from session state
    status_filter = st.session_state.get("status_filter", [])
    category_filter = st.session_state.get("category_filter", [])

    # Load data using cached function
    raw_data = load_evaluation_data_cached(data_dir)

    # Apply filters
    raw_reports = raw_data.get("reports", [])
    if raw_reports:
        filtered_reports = apply_filters(raw_reports, status_filter, category_filter)
        filtered_data = {
            "reports": filtered_reports,
            "summary": calculate_summary(filtered_reports) if filtered_reports else {},
        }
    else:
        filtered_data = {"reports": [], "summary": {}}

    # Display filter info
    if status_filter or category_filter:
        total_available = len(raw_reports)
        total_filtered = len(filtered_data.get("reports", []))
        if total_available != total_filtered:
            st.caption(f"Showing {total_filtered} of {total_available} reports (filtered)")

    # Four main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔍 Quality & Safety",
        "📅 History",
        "⚙️ Settings & Export",
    ])

    with tab1:
        render_overview_section(filtered_data)

    with tab2:
        render_quality_safety_section(filtered_data, raw_data, data_dir)

    with tab3:
        render_history_section(filtered_data)

    with tab4:
        render_settings_section(raw_data, data_dir)


if __name__ == "__main__":
    main()
