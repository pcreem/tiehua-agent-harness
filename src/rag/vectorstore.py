"""Vector store abstraction for RAG."""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import numpy as np

from .types import Chunk, VectorStoreConfig, RetrievalResult


class BaseVectorStore(ABC):
    """Base class for vector stores."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs."""
        pass

    @abstractmethod
    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist to disk."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        try:
            import chromadb
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(
                path=config.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": config.distance_metric},
            )
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        ids = [c.id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                **c.metadata,
                "document_id": c.document_id,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        where = filter_metadata if filter_metadata else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        if results["ids"] and results["ids"][0]:
            for rank, (chunk_id, content, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Convert distance to similarity score
                if self.config.distance_metric == "cosine":
                    score = 1 - distance
                elif self.config.distance_metric == "l2":
                    score = 1 / (1 + distance)
                else:
                    score = 1 - distance

                if score >= threshold:
                    chunk = Chunk(
                        id=chunk_id,
                        document_id=metadata.get("document_id", ""),
                        content=content,
                        start_char=metadata.get("start_char", 0),
                        end_char=metadata.get("end_char", len(content)),
                        metadata=metadata,
                        embedding=None,
                    )
                    retrieval_results.append(RetrievalResult(
                        chunk=chunk,
                        score=score,
                        rank=rank,
                    ))

        return retrieval_results

    def delete(self, chunk_ids: List[str]) -> None:
        self.collection.delete(ids=chunk_ids)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        results = self.collection.get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])
        if results["ids"]:
            meta = results["metadatas"][0]
            return Chunk(
                id=chunk_id,
                document_id=meta.get("document_id", ""),
                content=results["documents"][0],
                start_char=meta.get("start_char", 0),
                end_char=meta.get("end_char", 0),
                metadata=meta,
                embedding=results["embeddings"][0] if results["embeddings"] else None,
            )
        return None

    def persist(self) -> None:
        # ChromaDB auto-persists
        pass


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for testing."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.chunks: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        query_vec = np.array(query_embedding)

        results = []
        for chunk in self.chunks.values():
            if filter_metadata:
                match = all(chunk.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue

            if chunk.embedding is None:
                continue

            chunk_vec = np.array(chunk.embedding)
            # Cosine similarity
            score = np.dot(query_vec, chunk_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
            )

            if score >= threshold:
                results.append(RetrievalResult(chunk=chunk, score=score, rank=0))

        results.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(results[:top_k]):
            r.rank = i

        return results[:top_k]

    def delete(self, chunk_ids: List[str]) -> None:
        for cid in chunk_ids:
            self.chunks.pop(cid, None)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def persist(self) -> None:
        pass


class VectorStoreFactory:
    """Factory for creating vector stores."""

    @staticmethod
    def create(config: VectorStoreConfig) -> BaseVectorStore:
        """Create vector store based on provider."""
        if config.provider == "chroma":
            return ChromaVectorStore(config)
        elif config.provider == "memory":
            return InMemoryVectorStore(config)
        else:
            raise ValueError(f"Unknown vector store provider: {config.provider}")