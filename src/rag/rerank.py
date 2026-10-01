"""Reranking for RAG retrieval results."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from .types import RetrievalResult


class BaseReranker(ABC):
    """Base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Rerank retrieval results."""
        pass


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        if not results:
            return []

        self._load_model()

        # Prepare pairs for cross-encoder
        pairs = [(query, r.chunk.content) for r in results]

        # Get scores from cross-encoder
        scores = self._model.predict(pairs)

        # Create new results with cross-encoder scores
        reranked = []
        for i, (result, score) in enumerate(zip(results, scores)):
            reranked.append(RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                rank=result.rank,
            ))

        # Sort by cross-encoder score
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, r in enumerate(reranked[:top_k]):
            r.rank = i

        return reranked[:top_k]


class LLMReranker(BaseReranker):
    """Reranker using LLM for relevance scoring."""

    def __init__(self, llm_client, model: str = None):
        self.llm_client = llm_client
        self.model = model

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        if not results:
            return []

        # Create prompt for LLM reranking
        prompt = self._build_prompt(query, results)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a relevance ranking expert. Rank the documents by relevance to the query."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            import json
            ranking = json.loads(response.choices[0].message.content)

            # Reorder results
            reranked = []
            for item in ranking.get("ranking", []):
                idx = item.get("index", 0)
                if 0 <= idx < len(results):
                    reranked.append(RetrievalResult(
                        chunk=results[idx].chunk,
                        score=item.get("score", 0.0),
                        rank=item.get("rank", 0),
                    ))

            # Add any missing results
            existing_indices = {r.chunk.id for r in reranked}
            for r in results:
                if r.chunk.id not in existing_indices:
                    reranked.append(r)

            for i, r in enumerate(reranked[:top_k]):
                r.rank = i

            return reranked[:top_k]

        except Exception as e:
            # Fallback to original order
            return results[:top_k]

    def _build_prompt(self, query: str, results: List[RetrievalResult]) -> str:
        prompt = f"Query: {query}\n\nDocuments to rank:\n"
        for i, result in enumerate(results):
            prompt += f"\n{i}. {result.chunk.content[:500]}...\n"
        prompt += """
Return JSON with format:
{
  "ranking": [
    {"index": 0, "score": 0.9, "rank": 0},
    {"index": 1, "score": 0.7, "rank": 1}
  ]
}
"""
        return prompt


class SimpleReranker(BaseReranker):
    """Simple reranker using heuristic scoring."""

    def __init__(self, boost_factors: Optional[Dict[str, float]] = None):
        self.boost_factors = boost_factors or {
            "recency": 0.1,
            "length_penalty": 0.05,
        }

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        if not results:
            return []

        query_terms = set(query.lower().split())

        for result in results:
            score = result.score

            # Boost for query term coverage
            content_terms = set(result.chunk.content.lower().split())
            query_coverage = len(query_terms & content_terms) / max(len(query_terms), 1)
            score += 0.1 * query_coverage

            # Penalize very long chunks
            chunk_len = len(result.chunk.content)
            if chunk_len > 1000:
                score -= self.boost_factors["length_penalty"] * (chunk_len / 1000)

            result.score = max(0.0, min(1.0, score))

        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)

        for i, r in enumerate(results[:top_k]):
            r.rank = i

        return results[:top_k]


class RerankerFactory:
    """Factory for creating rerankers."""

    @staticmethod
    def create(
        reranker_type: str,
        **kwargs,
    ) -> BaseReranker:
        if reranker_type == "cross_encoder":
            return CrossEncoderReranker(**kwargs)
        elif reranker_type == "llm":
            return LLMReranker(**kwargs)
        elif reranker_type == "simple":
            return SimpleReranker(**kwargs)
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")