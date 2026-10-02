#!/usr/bin/env python3
"""QA Report Generator - Automated Conversation Quality Analysis.

Reads evaluation logs and uses LLM-as-a-Judge to generate quality reports
with error classification analysis.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import create_llm_client
from src.evaluation.llm_judge import LLMJudge


@dataclass
class ConversationAnalysis:
    """Analysis result for a single conversation."""
    conversation_id: str
    language: str
    intent: str
    num_turns: int
    num_retries: int
    final_status: str
    groundedness_score: float
    helpfulness_score: float
    tone_score: float
    completeness_score: float
    structure_score: float
    safety_score: float
    overall_score: float
    error_classifications: List[str]
    issues: List[Dict[str, Any]]
    judge_reasoning: str


@dataclass
class QAReport:
    """Complete QA Report structure."""
    generated_at: str
    summary: Dict[str, Any]
    error_classification: Dict[str, int]
    by_language: Dict[str, Dict[str, Any]]
    by_intent: Dict[str, Dict[str, Any]]
    retry_analysis: Dict[str, Any]
    detailed_findings: List[Dict[str, Any]]


class QAReportGenerator:
    """Generates QA reports from evaluation logs using LLM-as-a-Judge."""

    def __init__(
        self,
        logs_dir: Path,
        judge_model: str = "poolside/laguna-s-2.1:free",
        api_key: Optional[str] = None,
    ):
        self.logs_dir = Path(logs_dir)
        self.judge_model = judge_model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.llm_client = create_llm_client(
            model=judge_model,
            api_key=self.api_key,
            timeout=60.0,
        )
        self.judge = LLMJudge(client=self.llm_client)

    def generate_report(self, output_path: Optional[Path] = None) -> QAReport:
        """Generate full QA report from logs."""
        # 1. Load all evaluation logs
        logs = self._load_logs()

        # 2. Group by conversation
        conversations = self._group_by_conversation(logs)

        # 3. Analyze each conversation with LLM Judge
        analyses = []
        for conv_id, conv_logs in conversations.items():
            print(f"Analyzing conversation: {conv_id}")
            analysis = self._analyze_conversation(conv_id, conv_logs)
            analyses.append(analysis)

        # 4. Aggregate statistics
        report = self._aggregate_report(analyses)

        # 5. Save report
        if output_path:
            self._save_report(report, output_path)

        return report

    def _load_logs(self) -> List[Dict]:
        """Load all evaluation log files."""
        logs = []
        log_files = list(self.logs_dir.glob("*.jsonl")) + list(self.logs_dir.glob("*.json"))

        for log_file in log_files:
            try:
                if log_file.suffix == ".jsonl":
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                logs.append(json.loads(line))
                else:
                    with open(log_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            logs.extend(data)
                        else:
                            logs.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {log_file}: {e}")

        print(f"Loaded {len(logs)} log entries from {len(log_files)} files")
        return logs

    def _group_by_conversation(self, logs: List[Dict]) -> Dict[str, List[Dict]]:
        """Group logs by conversation/run ID."""
        conversations = {}
        for log in logs:
            # Try different ID fields
            conv_id = (
                log.get("run_id") or
                log.get("conversation_id") or
                log.get("state", {}).get("run_id") or
                log.get("id") or
                "unknown"
            )
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(log)
        return conversations

    def _analyze_conversation(self, conv_id: str, logs: List[Dict]) -> ConversationAnalysis:
        """Analyze a single conversation using LLM Judge."""
        # Extract key information
        last_log = logs[-1] if logs else {}
        state = last_log.get("state", {})

        language = state.get("language", "zh-TW")
        intent = state.get("user_intent", "unknown")
        num_retries = state.get("retry_count", 0)
        final_status = state.get("status", "unknown")
        answer = state.get("answer", "")
        query = state.get("query", "")

        # Build conversation history for judge
        messages = []
        for log in logs:
            if "messages" in log.get("state", {}):
                messages = log["state"]["messages"]
                break

        # Use LLM Judge for quality evaluation
        judge_result = self.judge.evaluate_conversation(
            query=query,
            answer=answer,
            messages=messages,
            tool_calls=state.get("tool_calls", []),
            tool_results=state.get("tool_results", []),
            language=language,
        )

        # Classify errors from evaluation feedback
        error_classifications = self._classify_errors(state, logs)

        return ConversationAnalysis(
            conversation_id=conv_id,
            language=language,
            intent=intent,
            num_turns=len(messages) if messages else 0,
            num_retries=num_retries,
            final_status=final_status,
            groundedness_score=judge_result.get("groundedness", 0.0),
            helpfulness_score=judge_result.get("helpfulness", 0.0),
            tone_score=judge_result.get("tone_appropriateness", 0.0),
            completeness_score=judge_result.get("completeness", 0.0),
            structure_score=judge_result.get("structure_clarity", 0.0),
            safety_score=judge_result.get("safety_compliance", 1.0),
            overall_score=judge_result.get("overall_score", 0.0),
            error_classifications=error_classifications,
            issues=judge_result.get("issues", []),
            judge_reasoning=judge_result.get("reasoning", ""),
        )

    def _classify_errors(self, state: Dict, logs: List[Dict]) -> List[str]:
        """Classify errors from evaluation feedback."""
        errors = []

        # From evaluation results
        evaluation = state.get("evaluation", {})
        if evaluation:
            error_type = evaluation.get("error_type")
            if error_type:
                errors.append(error_type)

        # From feedback
        feedback = state.get("feedback", [])
        for fb in feedback:
            if isinstance(fb, dict):
                code = fb.get("code") or fb.get("error_type")
                if code:
                    errors.append(code)
            elif isinstance(fb, str):
                # Extract error codes from string feedback
                if "missing_evidence" in fb:
                    errors.append("missing_evidence")
                elif "groundedness" in fb:
                    errors.append("groundedness_failure")
                elif "tool_error" in fb:
                    errors.append("tool_error")
                elif "incomplete" in fb:
                    errors.append("incomplete_answer")

        return list(set(errors))  # Deduplicate

    def _aggregate_report(self, analyses: List[ConversationAnalysis]) -> QAReport:
        """Aggregate analyses into final report."""
        total = len(analyses)
        if total == 0:
            return QAReport(
                generated_at=datetime.now().isoformat(),
                summary={"total_conversations": 0},
                error_classification={},
                by_language={},
                by_intent={},
                retry_analysis={},
                detailed_findings=[],
            )

        # Summary
        passed = sum(1 for a in analyses if a.overall_score >= 0.75)
        avg_score = sum(a.overall_score for a in analyses) / total
        avg_retries = sum(a.num_retries for a in analyses) / total

        summary = {
            "total_conversations": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3),
            "avg_overall_score": round(avg_score, 3),
            "avg_retries": round(avg_retries, 2),
            "status_distribution": dict(Counter(a.final_status for a in analyses)),
        }

        # Error classification
        all_errors = []
        for a in analyses:
            all_errors.extend(a.error_classifications)
        error_classification = dict(Counter(all_errors))

        # By language
        by_language = {}
        for lang in ["zh-TW", "en", "ja"]:
            lang_analyses = [a for a in analyses if a.language == lang]
            if lang_analyses:
                by_language[lang] = {
                    "count": len(lang_analyses),
                    "pass_rate": round(sum(1 for a in lang_analyses if a.overall_score >= 0.75) / len(lang_analyses), 3),
                    "avg_score": round(sum(a.overall_score for a in lang_analyses) / len(lang_analyses), 3),
                    "avg_retries": round(sum(a.num_retries for a in lang_analyses) / len(lang_analyses), 2),
                }

        # By intent
        by_intent = {}
        for intent in ["weather", "attraction", "general", "mixed"]:
            intent_analyses = [a for a in analyses if a.intent == intent]
            if intent_analyses:
                by_intent[intent] = {
                    "count": len(intent_analyses),
                    "pass_rate": round(sum(1 for a in intent_analyses if a.overall_score >= 0.75) / len(intent_analyses), 3),
                    "avg_score": round(sum(a.overall_score for a in intent_analyses) / len(intent_analyses), 3),
                    "avg_retries": round(sum(a.num_retries for a in intent_analyses) / len(intent_analyses), 2),
                }

        # Retry analysis
        retry_analysis = {
            "avg_retries": round(avg_retries, 2),
            "retry_distribution": dict(Counter(a.num_retries for a in analyses)),
            "retry_success_rate": 0.0,
        }
        # Calculate retry success rate (conversations that passed after retry)
        retried = [a for a in analyses if a.num_retries > 0]
        if retried:
            retry_analysis["retry_success_rate"] = round(
                sum(1 for a in retried if a.overall_score >= 0.75) / len(retried), 3
            )

        # Detailed findings
        detailed_findings = [asdict(a) for a in analyses]

        return QAReport(
            generated_at=datetime.now().isoformat(),
            summary=summary,
            error_classification=error_classification,
            by_language=by_language,
            by_intent=by_intent,
            retry_analysis=retry_analysis,
            detailed_findings=detailed_findings,
        )

    def _save_report(self, report: QAReport, output_path: Path):
        """Save report as JSON and Markdown."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

        # Save Markdown
        md_path = output_path.with_suffix(".md")
        self._write_markdown_report(report, md_path)

        print(f"Report saved to {json_path} and {md_path}")

    def _write_markdown_report(self, report: QAReport, md_path: Path):
        """Write human-readable Markdown report."""
        lines = [
            "# QA Quality Report",
            f"\nGenerated: {report.generated_at}",
            "\n## Summary",
            f"- Total Conversations: {report.summary.get('total_conversations', 0)}",
            f"- Passed: {report.summary.get('passed', 0)}",
            f"- Pass Rate: {report.summary.get('pass_rate', 0):.1%}",
            f"- Average Score: {report.summary.get('avg_overall_score', 0):.2f}",
            f"- Average Retries: {report.summary.get('avg_retries', 0):.2f}",
            "\n## Error Classification",
        ]

        for error, count in sorted(report.error_classification.items(), key=lambda x: -x[1]):
            lines.append(f"- **{error}**: {count}")

        lines.append("\n## By Language")
        for lang, stats in report.by_language.items():
            lines.append(f"### {lang}")
            lines.append(f"- Count: {stats['count']}")
            lines.append(f"- Pass Rate: {stats['pass_rate']:.1%}")
            lines.append(f"- Avg Score: {stats['avg_score']:.2f}")

        lines.append("\n## By Intent")
        for intent, stats in report.by_intent.items():
            lines.append(f"### {intent}")
            lines.append(f"- Count: {stats['count']}")
            lines.append(f"- Pass Rate: {stats['pass_rate']:.1%}")
            lines.append(f"- Avg Score: {stats['avg_score']:.2f}")

        lines.append("\n## Retry Analysis")
        lines.append(f"- Average Retries: {report.retry_analysis.get('avg_retries', 0):.2f}")
        lines.append(f"- Retry Success Rate: {report.retry_analysis.get('retry_success_rate', 0):.1%}")

        md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate QA Report from evaluation logs")
    parser.add_argument("--logs-dir", default="evaluation/logs", help="Directory containing evaluation logs")
    parser.add_argument("--output", default="evaluation/reports/qa_report", help="Output path (without extension)")
    parser.add_argument("--judge-model", default="poolside/laguna-s-2.1:free", help="LLM Judge model")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")

    args = parser.parse_args()

    generator = QAReportGenerator(
        logs_dir=Path(args.logs_dir),
        judge_model=args.judge_model,
        api_key=args.api_key,
    )

    report = generator.generate_report(Path(args.output))

    # Print summary
    print("\n=== QA Report Summary ===")
    print(f"Total: {report.summary.get('total_conversations', 0)}")
    print(f"Pass Rate: {report.summary.get('pass_rate', 0):.1%}")
    print(f"Avg Score: {report.summary.get('avg_overall_score', 0):.2f}")
    print(f"Top Errors: {list(report.error_classification.keys())[:5]}")


if __name__ == "__main__":
    main()