"""Retrieval system for RAG."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .types import (
    Chunk,
    RetrievalResult,
    RetrievalConfig,
    RetrievalStrategy,
    EmbeddingConfig,
    VectorStoreConfig,
)
from .embeddings import BaseEmbedder, EmbeddingFactory
from .vectorstore import BaseVectorStore, VectorStoreFactory


@dataclass
class RetrievalContext:
    """Context for retrieval."""
    query: str
    query_embedding: List[float]
    config: RetrievalConfig
    filters: Optional[Dict[str, Any]] = None


class BaseRetriever(ABC):
    """Base retriever interface."""

    @abstractmethod
    def retrieve(self, context: RetrievalContext) -> List[RetrievalResult]:
        """Retrieve relevant chunks."""
        pass


class VectorRetriever(BaseRetriever):
    """Vector similarity retriever."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        config: RetrievalConfig,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config

    def retrieve(self, context: RetrievalContext) -> List[RetrievalResult]:
        return self.vector_store.search(
            query_embedding=context.query_embedding,
            top_k=self.config.top_k,
            threshold=self.config.similarity_threshold,
            filter_metadata=context.filters,
        )


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector and keyword search."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        config: RetrievalConfig,
        keyword_weight: float = 0.3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config
        self.keyword_weight = keyword_weight

    def retrieve(self, context: RetrievalContext) -> List[RetrievalResult]:
        # Get more candidates for hybrid scoring
        vector_results = self.vector_store.search(
            query_embedding=context.query_embedding,
            top_k=self.config.rerank_top_k or self.config.top_k * 2,
            threshold=self.config.similarity_threshold,
            filter_metadata=context.filters,
        )

        # Apply keyword scoring
        scored_results = []
        for result in vector_results:
            keyword_score = self._keyword_score(context.query, result.chunk.content)
            hybrid_score = (
                (1 - self.keyword_weight) * result.score +
                self.keyword_weight * keyword_score
            )
            scored_results.append(RetrievalResult(
                chunk=result.chunk,
                score=hybrid_score,
                rank=result.rank,
            ))

        scored_results.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(scored_results):
            r.rank = i

        return scored_results[:self.config.top_k]

    def _keyword_score(self, query: str, content: str) -> float:
        """Simple TF-IDF-like keyword matching."""
        query_terms = set(query.lower().split())
        content_terms = content.lower().split()

        if not query_terms:
            return 0.0

        matches = sum(1 for term in query_terms if term in content_terms)
        return matches / len(query_terms)


class RetrieverFactory:
    """Factory for creating retrievers."""

    @staticmethod
    def create(
        strategy: RetrievalStrategy,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        config: RetrievalConfig,
        **kwargs,
    ) -> BaseRetriever:
        if strategy == RetrievalStrategy.VECTOR:
            return VectorRetriever(embedder, vector_store, config)
        elif strategy == RetrievalStrategy.HYBRID:
            return HybridRetriever(embedder, vector_store, config, **kwargs)
        elif strategy == RetrievalStrategy.RERANKED:
            # Reranked uses vector + reranker
            return VectorRetriever(embedder, vector_store, config)
        else:
            return VectorRetriever(embedder, vector_store, config)

    @staticmethod
    def create_full_pipeline(
        config: RetrievalConfig,
        embedding_config: EmbeddingConfig,
        vector_config: VectorStoreConfig,
        **kwargs,
    ) -> BaseRetriever:
        """Create complete retrieval pipeline."""
        embedder = EmbeddingFactory.create(embedding_config)
        vector_store = VectorStoreFactory.create(vector_config)
        return RetrieverFactory.create(
            config.strategy, embedder, vector_store, config, **kwargs
        )