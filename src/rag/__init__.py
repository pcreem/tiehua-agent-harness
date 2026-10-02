"""Main RAG module - high-level interface."""

from typing import List, Optional, Dict, Any

from .types import (
    Document,
    Chunk,
    RetrievalResult,
    ChunkConfig,
    RetrievalConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    ChunkStrategy,
    RetrievalStrategy,
)
from .ingestion import (
    IngestionPipeline,
    IngestionPipelineBuilder,
    DocumentLoader,
    IngestionResult,
    create_ingestion_pipeline,
)
from .chunking import create_chunker, BaseChunker
from .embeddings import BaseEmbedder, EmbeddingFactory, EmbeddingConfig
from .vectorstore import BaseVectorStore, VectorStoreFactory, VectorStoreConfig
from .retrieval import BaseRetriever, RetrieverFactory, RetrievalConfig
from .rerank import BaseReranker, RerankerFactory


class RAGSystem:
    """High-level RAG system combining all components."""

    def __init__(
        self,
        retriever,
        embedder,
        reranker=None,
        retrieval_config: RetrievalConfig = None,
    ):
        self.retriever = retriever
        self.embedder = embedder
        self.reranker = reranker
        self.retrieval_config = retrieval_config or RetrievalConfig()

    @classmethod
    def from_config(
        cls,
        retrieval_config: RetrievalConfig,
        embedding_config: EmbeddingConfig,
        vector_config: VectorStoreConfig,
        reranker_type: str = "simple",
        **reranker_kwargs,
    ) -> "RAGSystem":
        """Create RAG system from configuration."""
        # Create embedder
        embedder = EmbeddingFactory.create(embedding_config)

        # Create retriever
        retriever = RetrieverFactory.create_full_pipeline(
            retrieval_config, embedding_config, vector_config
        )

        # Create reranker
        reranker = None
        if reranker_type:
            reranker = RerankerFactory.create(reranker_type, **reranker_kwargs)

        return cls(retriever=retriever, embedder=embedder, reranker=reranker, retrieval_config=retrieval_config)

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Retrieve relevant chunks for query."""
        from .types import RetrievalContext

        # Get query embedding using the embedder
        query_embedding = self.embedder.embed_query(query)

        context = RetrievalContext(
            query=query,
            query_embedding=query_embedding,
            config=self.retrieval_config,
            filters=filters,
        )

        results = self.retriever.retrieve(context)

        # Rerank if configured
        if self.reranker:
            results = self.reranker.rerank(query, results, self.retrieval_config.top_k)

        return results

    def ingest(self, document: Document) -> "IngestionResult":
        """Ingest a document into the RAG system."""
        # This would require access to ingestion pipeline
        raise NotImplementedError("Use IngestionPipeline directly for ingestion")

    def query(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Query the RAG system."""
        if top_k:
            original_top_k = self.retrieval_config.top_k
            self.retrieval_config.top_k = top_k

        results = self.retrieve(query, filters)

        if top_k:
            self.retrieval_config.top_k = original_top_k

        return results


# Convenience functions for quick setup
def create_rag_system(
    chunk_strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "nvidia/nemotron-3-embed-1b:free",
    vector_provider: str = "chroma",
    persist_dir: str = "./data/vectorstore",
    retrieval_strategy: str = "vector",
    reranker_type: str = "simple",
    top_k: int = 5,
) -> RAGSystem:
    """Create a complete RAG system with sensible defaults."""
    # Build configs
    chunk_config = ChunkConfig(
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embedding_config = EmbeddingConfig(
        provider="openrouter",
        model=embedding_model,
    )

    vector_config = VectorStoreConfig(
        provider=vector_provider,
        persist_dir=persist_dir,
    )

    retrieval_config = RetrievalConfig(
        strategy=RetrievalStrategy(retrieval_strategy),
        top_k=top_k,
    )

    # Create embedder
    embedder = EmbeddingFactory.create(embedding_config)

    # Create retriever
    retriever = RetrieverFactory.create_full_pipeline(
        retrieval_config, embedding_config, vector_config
    )

    # Create reranker
    reranker = RerankerFactory.create(reranker_type)

    return RAGSystem(
        retriever=retriever,
        embedder=embedder,
        reranker=reranker,
        retrieval_config=retrieval_config,
    )


__all__ = [
    # Types
    "Document",
    "Chunk",
    "RetrievalResult",
    "ChunkConfig",
    "RetrievalConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "ChunkStrategy",
    "RetrievalStrategy",
    # Ingestion
    "IngestionPipeline",
    "IngestionPipelineBuilder",
    "DocumentLoader",
    "IngestionResult",
    "create_ingestion_pipeline",
    # Chunking
    "create_chunker",
    "BaseChunker",
    # Embeddings
    "BaseEmbedder",
    "EmbeddingFactory",
    # Vector Store
    "BaseVectorStore",
    "VectorStoreFactory",
    # Retrieval
    "BaseRetriever",
    "RetrieverFactory",
    "RetrievalConfig",
    # Rerank
    "BaseReranker",
    "RerankerFactory",
    # High-level
    "RAGSystem",
    "create_rag_system",
]