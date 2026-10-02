"""Document ingestion for RAG."""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

from .types import Document, Chunk, ChunkConfig
from .chunking import create_chunker, BaseChunker
from .embeddings import BaseEmbedder, EmbeddingFactory, EmbeddingConfig
from .vectorstore import BaseVectorStore, VectorStoreFactory, VectorStoreConfig


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    chunks_created: int
    success: bool
    error: Optional[str] = None


class DocumentLoader:
    """Load documents from various sources."""

    @staticmethod
    def from_file(path: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Load document from file."""
        path = Path(path)
        content = path.read_text(encoding="utf-8")

        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        meta = metadata or {}
        meta.update({
            "source_path": str(path),
            "file_name": path.name,
            "file_size": len(content),
        })

        return Document(
            id=doc_id,
            content=content,
            metadata=meta,
            source=str(path),
        )

    @staticmethod
    def from_directory(
        directory: str,
        pattern: str = "**/*.txt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Load all matching files from directory."""
        path = Path(directory)
        documents = []
        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    doc = DocumentLoader.from_file(file_path, metadata)
                    documents.append(doc)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
        return documents

    @staticmethod
    def from_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Create document from raw text."""
        content = text
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        meta = metadata or {}
        meta.update({
            "source": "text",
        })

        return Document(
            id=doc_id,
            content=content,
            metadata=meta,
        )


class IngestionPipeline:
    """End-to-end document ingestion pipeline."""

    def __init__(
        self,
        chunker: BaseChunker,
        embedder,
        vector_store,
        chunk_config: ChunkConfig,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_config = chunk_config

    def ingest(self, document: Document) -> IngestionResult:
        """Ingest a single document."""
        try:
            # Chunk document
            chunks = self.chunker.chunk(document)

            # Generate embeddings
            if chunks:
                texts = [c.content for c in chunks]
                embeddings = self.embedder.embed(texts)

                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding

            # Store in vector store
            self.vector_store.add_chunks(chunks)
            self.vector_store.persist()

            return IngestionResult(
                document_id=document.id,
                chunks_created=len(chunks),
                success=True,
            )

        except Exception as e:
            return IngestionResult(
                document_id=document.id,
                chunks_created=0,
                success=False,
                error=str(e),
            )

    def ingest_batch(self, documents: List[Document]) -> List[IngestionResult]:
        """Ingest multiple documents."""
        results = []
        for doc in documents:
            result = self.ingest(doc)
            results.append(result)
        return results

    def ingest_from_directory(
        self,
        directory: str,
        pattern: str = "**/*.txt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[IngestionResult]:
        """Ingest all documents from directory."""
        documents = DocumentLoader.from_directory(directory, pattern, metadata)
        return self.ingest_batch(documents)


class IngestionPipelineBuilder:
    """Builder for creating ingestion pipeline with configuration."""

    def __init__(self):
        self._chunk_config = ChunkConfig()
        self._embedding_config = EmbeddingConfig()
        self._vector_config = VectorStoreConfig()
        self._custom_chunker = None

    def with_chunk_config(self, config: ChunkConfig) -> "IngestionPipelineBuilder":
        self._chunk_config = config
        return self

    def with_embedding_config(self, config: EmbeddingConfig) -> "IngestionPipelineBuilder":
        self._embedding_config = config
        return self

    def with_vector_config(self, config: VectorStoreConfig) -> "IngestionPipelineBuilder":
        self._vector_config = config
        return self

    def with_custom_chunker(self, chunker: BaseChunker) -> "IngestionPipelineBuilder":
        self._custom_chunker = chunker
        return self

    def build(self) -> IngestionPipeline:
        chunker = self._custom_chunker or create_chunker(self._chunk_config)
        embedder = EmbeddingFactory.create(self._embedding_config)
        vector_store = VectorStoreFactory.create(self._vector_config)

        return IngestionPipeline(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            chunk_config=self._chunk_config,
        )


def create_ingestion_pipeline(
    chunk_strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "nvidia/nemotron-3-embed-1b:free",
    vector_provider: str = "chroma",
    persist_dir: str = "./data/vectorstore",
) -> IngestionPipeline:
    """Convenience function to create ingestion pipeline with defaults."""
    builder = IngestionPipelineBuilder()
    builder._chunk_config = ChunkConfig(
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    builder._embedding_config = EmbeddingConfig(
        provider="openrouter",
        model=embedding_model,
    )
    builder._vector_config = VectorStoreConfig(
        provider=vector_provider,
        persist_dir=persist_dir,
    )
    return builder.build()