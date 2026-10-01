#!/usr/bin/env python
"""Evaluation runner for Tiehua Tourism Agent Harness."""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.harness.runner import HarnessRunner
from src.state import AgentState, initialize_state
from src.evaluation import run_evaluation, get_registry
from src.evaluation.deterministic import DeterministicEvaluator


class EvaluationRunner:
    """Runs evaluation dataset against harness."""

    def __init__(self, max_retries: int = 2):
        self.runner = HarnessRunner(max_retries=max_retries)
        self.evaluator = DeterministicEvaluator()
        self.results = []

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load evaluation dataset from JSON file."""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_case(self, case: Dict) -> Dict:
        """Run a single evaluation case."""
        case_id = case["id"]
        query = case["query"]
        expected = case["expected_behavior"]

        print(f"\nRunning case: {case_id}")
        print(f"  Query: {query}")

        try:
            # Run harness
            harness_result = self.runner.run(query)

            # Evaluate
            state = harness_result.state
            eval_results = self.evaluator.evaluate(query, state, expected)

            # Check expected tools
            expected_tools = expected.get("required_tools", [])
            actual_tools = [tc["name"] for tc in harness_result.tool_calls]
            tools_correct = all(tool in actual_tools for tool in expected_tools)

            # Check forbidden tools
            forbidden_tools = expected.get("forbidden_tools", [])
            forbidden_correct = all(tool not in actual_tools for tool in forbidden_tools)

            case_result = {
                "id": case_id,
                "query": query,
                "status": "passed" if (
                    eval_results.status.value == "pass" and
                    tools_correct and
                    forbidden_correct
                ) else "failed",
                "harness_status": harness_result.status,
                "eval_status": eval_results.status.value,
                "eval_score": eval_results.score,
                "tools_correct": tools_correct,
                "forbidden_correct": forbidden_correct,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "eval_feedback": eval_results.feedback,
                "final_answer": harness_result.final_answer[:200],
            }

            status = "✓ PASS" if case_result["status"] == "passed" else "✗ FAIL"
            print(f"  {status} (harness: {harness_result.status}, eval: {eval_results.status.value}, score: {eval_results.score:.2f})")
            if case_result["status"] == "failed":
                print(f"  Feedback: {eval_results.feedback}")

            return case_result

        except Exception as e:
            case_result = {
                "id": case_id,
                "query": query,
                "status": "error",
                "error": str(e),
            }
            print(f"  ✗ ERROR: {e}")
            return case_result

    def run_dataset(self, dataset_path: str) -> Dict:
        """Run full dataset."""
        cases = self.load_dataset(dataset_path)
        print(f"Loaded {len(cases)} cases from {dataset_path}")

        self.results = []
        for case in cases:
            result = self.run_case(case)
            self.results.append(result)

        return self.summarize()

    def summarize(self) -> Dict:
        """Generate summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("status") == "passed")
        failed = sum(1 for r in self.results if r.get("status") == "failed")
        errors = sum(1 for r in self.results if r.get("status") == "error")

        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
        }

        print(f"\n{'='*50}")
        print(f"SUMMARY")
        print(f"{'='*50}")
        print(f"Total:  {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")

        if failed > 0:
            print(f"\nFailed cases:")
            for r in self.results:
                if r.get("status") == "failed":
                    print(f"  - {r['id']}: {r['query'][:50]}")
                    print(f"    Eval: {r.get('eval_status')}, Tools: {r.get('tools_correct')}/{r.get('forbidden_correct')}")
                    print(f"    Feedback: {r.get('eval_feedback')}")

        return summary

    def save_report(self, output_path: str) -> None:
        """Save evaluation report."""
        summary = self.summarize()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/datasets/smoke.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/report.json",
        help="Output report path"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries for harness"
    )
    args = parser.parse_args()

    runner = EvaluationRunner(max_retries=args.max_retries)
    summary = runner.run_dataset(args.dataset)
    runner.save_report(args.output)

    # Exit with error code if any failures
    if summary["failed"] > 0 or summary["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()