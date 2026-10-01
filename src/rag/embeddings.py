"""Embeddings generation for RAG."""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import openai
from pydantic import BaseModel

from .types import EmbeddingConfig


class BaseEmbedder(ABC):
    """Base class for text embedders."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings (works with OpenRouter)."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=config.api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=config.base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            response = self.client.embeddings.create(
                model=self.config.model,
                input=batch,
                dimensions=self.config.dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed([query])[0]


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing without API calls."""

    def __init__(self, config: EmbeddingConfig, dimension: int = 1536):
        super().__init__(config)
        self.dimension = dimension
        import hashlib
        self._hashlib = hashlib

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings based on text hash."""
        import random
        embeddings = []
        for text in texts:
            # Use hash for deterministic results
            seed = int(self._hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            embedding = [random.uniform(-1, 1) for _ in range(self.dimension)]
            # Normalize
            norm = sum(x * x for x in embedding) ** 0.5
            embedding = [x / norm for x in embedding]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embed([query])[0]


class EmbeddingFactory:
    """Factory for creating embedders."""

    @staticmethod
    def create(config: EmbeddingConfig) -> BaseEmbedder:
        """Create embedder based on config."""
        # Use mock if no API key available and not explicitly requesting real provider
        if config.provider in ("openai", "openrouter"):
            api_key = config.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                # Fallback to mock if no credentials
                return MockEmbedder(config)
            return OpenAIEmbedder(config)
        elif config.provider == "mock":
            return MockEmbedder(config)
        else:
            # Default to mock for testing
            return MockEmbedder(config)