"""RAG types and data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"  # vector + keyword
    RERANKED = "reranked"  # vector + rerank


@dataclass
class Document:
    """Source document for RAG."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    source_url: str = ""


@dataclass
class Chunk:
    """Text chunk after chunking."""
    id: str
    document_id: str
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    chunk: Chunk
    score: float
    rank: int


class ChunkConfig(BaseModel):
    """Configuration for chunking."""
    strategy: ChunkStrategy = ChunkStrategy.FIXED
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 50


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank_top_k: int = 10


class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""
    provider: str = "openrouter"  # openai, openrouter, mock
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    provider: str = "chroma"  # chroma, pgvector, faiss
    collection_name: str = "tiehua_rag"
    persist_dir: str = "./data/vectorstore"
    distance_metric: str = "cosine"  # cosine, l2, ip