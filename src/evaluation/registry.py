"""Evaluator registry for managing multiple evaluators."""

from typing import Any, Dict, List, Optional
from src.evaluation.deterministic import DeterministicEvaluator, DeterministicEvaluator
from src.evaluation.types import ExpectedBehavior, EvaluationResult


class EvaluatorRegistry:
    """Registry for managing and running multiple evaluators."""

    def __init__(self):
        self._evaluators: Dict[str, Any] = {}
        self._register_default_evaluators()

    def _register_default_evaluators(self):
        """Register built-in evaluators."""
        self.register("deterministic", DeterministicEvaluator())

    def register(self, name: str, evaluator: Any) -> None:
        """Register an evaluator."""
        if not hasattr(evaluator, 'evaluate'):
            raise ValueError(f"Evaluator must have 'evaluate' method")
        self._evaluators[name] = evaluator

    def unregister(self, name: str) -> bool:
        """Unregister an evaluator."""
        if name in self._evaluators:
            del self._evaluators[name]
            return True
        return False

    def get(self, name: str) -> Optional[Any]:
        """Get evaluator by name."""
        return self._evaluators.get(name)

    def all(self) -> List[Any]:
        """Get all evaluators."""
        return list(self._evaluators.values())

    def names(self) -> List[str]:
        """Get all evaluator names."""
        return list(self._evaluators.keys())

    def evaluate(
        self,
        query: str,
        state,
        expected: Optional[Any] = None,
        evaluator_names: Optional[List[str]] = None,
    ) -> Dict[str, EvaluationResult]:
        """Run evaluation with specified evaluators."""
        if evaluator_names is None:
            evaluators_to_run = self._evaluators
        else:
            evaluators_to_run = {n: self._evaluators[n] for n in evaluator_names if n in self._evaluators}

        results = {}
        for name, evaluator in evaluators_to_run.items():
            try:
                result = evaluator.evaluate(query, state, None)
                results[name] = result
            except Exception as e:
                from src.evaluation.types import EvaluationResult, EvaluatorStatus
                results[name] = EvaluationResult(
                    status="error",
                    score=0.0,
                    feedback=[f"Evaluator {name} failed: {str(e)}"],
                    retry=False,
                    metadata={"error": str(e)},
                )
        return results

    def aggregate(self, results: Dict[str, EvaluationResult]) -> Any:
        """Aggregate multiple evaluation results."""
        from src.evaluation.types import EvaluationResult, EvaluatorStatus

        if not results:
            return EvaluationResult(
                status=EvaluatorStatus.PASS,
                score=1.0,
                feedback=[],
            )

        all_passed = all(r.status.value == "pass" for r in results.values())
        scores = [r.score for r in results.values() if r.score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.5
        all_feedback = []
        for r in results.values():
            all_feedback.extend(r.feedback)

        return EvaluationResult(
            status=EvaluatorStatus.PASS if all_passed else EvaluatorStatus.FAIL,
            score=avg_score,
            feedback=all_feedback,
            retry=not all_passed,
            metadata={"evaluators": list(results.keys())},
        )


# Global registry instance
_global_registry: Optional[EvaluatorRegistry] = None


def get_registry() -> EvaluatorRegistry:
    """Get global evaluator registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = EvaluatorRegistry()
    return _global_registry


def register_evaluator(name: str, evaluator: Any) -> None:
    """Register evaluator in global registry."""
    get_registry().register(name, evaluator)


def run_evaluation(
    query: str,
    state,
    expected: Any = None,
    evaluator_names: Optional[List[str]] = None,
) -> Any:
    """Run evaluation with global registry."""
    return get_registry().evaluate(query, state, expected, evaluator_names)