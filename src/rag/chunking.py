"""Text chunking strategies for RAG."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from .types import Chunk, Document, ChunkConfig, ChunkStrategy


class BaseChunker(ABC):
    """Base class for text chunkers."""

    def __init__(self, config: ChunkConfig):
        self.config = config

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into chunks."""
        pass


class FixedSizeChunker(BaseChunker):
    """Fixed-size character-based chunking with overlap."""

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if len(text) <= chunk_size:
            return [Chunk(
                id=f"{document.id}_chunk_0",
                document_id=document.id,
                content=text,
                start_char=0,
                end_char=len(text),
                metadata=document.metadata.copy(),
            )]

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars
                search_start = max(start, end - 100)
                sentence_end = text.rfind(".", search_start, end)
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={**document.metadata, "chunk_index": chunk_idx},
                ))
                chunk_idx += 1

            start = end - overlap
            if start >= len(text):
                break

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive character-based chunking with separators."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "．", ".", "！", "!", "？", "?", "，", ",", "、", " "]

    def __init__(self, config: ChunkConfig, separators: Optional[List[str]] = None):
        super().__init__(config)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        chunks = self._recursive_split(text, self.separators)
        return self._create_chunks(document, chunks)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text[i:i + self.config.chunk_size] for i in range(0, len(text), self.config.chunk_size)]

        sep = separators[0]
        parts = text.split(sep)
        results = []

        for part in parts:
            if len(part) <= self.config.chunk_size:
                if part.strip():
                    results.append(part)
            else:
                # Recurse with next separator
                results.extend(self._recursive_split(part, separators[1:]))

        # Merge small chunks
        merged = []
        current = ""
        for chunk in results:
            if len(current) + len(chunk) <= self.config.chunk_size:
                current += chunk
            else:
                if current.strip():
                    merged.append(current)
                current = chunk
        if current.strip():
            merged.append(current)

        return merged

    def _create_chunks(self, document: Document, texts: List[str]) -> List[Chunk]:
        chunks = []
        char_pos = 0
        for i, text in enumerate(texts):
            text = text.strip()
            if not text or len(text) < self.config.min_chunk_size:
                continue
            chunks.append(Chunk(
                id=f"{document.id}_chunk_{i}",
                document_id=document.id,
                content=text,
                start_char=char_pos,
                end_char=char_pos + len(text),
                metadata={**document.metadata, "chunk_index": i},
            ))
            char_pos += len(text)
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking using sentence embeddings (requires embeddings model)."""

    def __init__(self, config: ChunkConfig, embedder=None):
        super().__init__(config)
        self.embedder = embedder

    def chunk(self, document: Document) -> List[Chunk]:
        # Split into sentences first
        sentences = self._split_sentences(document.content)
        if len(sentences) <= 1:
            return [Chunk(
                id=f"{document.id}_chunk_0",
                document_id=document.id,
                content=document.content,
                start_char=0,
                end_char=len(document.content),
                metadata=document.metadata.copy(),
            )]

        # Group sentences by semantic similarity
        groups = self._group_sentences(sentences)
        return self._create_chunks(document, groups)

    def _split_sentences(self, text: str) -> List[str]:
        # Split by Chinese and English sentence endings
        pattern = r'([。．.!?！？]+)'
        parts = re.split(pattern, text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sent = parts[i]
            if i + 1 < len(parts):
                sent += parts[i + 1]
            if sent.strip():
                sentences.append(sent.strip())
        return sentences

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        # Simplified: group by chunk_size characters
        # Full implementation would use embeddings for semantic similarity
        groups = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.config.chunk_size:
                current += sent
            else:
                if current:
                    groups.append(current)
                current = sent
        if current:
            groups.append(current)
        return groups

    def _create_chunks(self, document: Document, texts: List[str]) -> List[Chunk]:
        chunks = []
        char_pos = 0
        for i, text in enumerate(texts):
            text = text.strip()
            if not text or len(text) < self.config.min_chunk_size:
                continue
            chunks.append(Chunk(
                id=f"{document.id}_chunk_{i}",
                document_id=document.id,
                content=text,
                start_char=char_pos,
                end_char=char_pos + len(text),
                metadata={**document.metadata, "chunk_index": i},
            ))
            char_pos += len(text)
        return chunks


def create_chunker(config: ChunkConfig, embedder=None) -> BaseChunker:
    """Factory function to create chunker based on strategy."""
    if config.strategy == ChunkStrategy.FIXED:
        return FixedSizeChunker(config)
    elif config.strategy == ChunkStrategy.RECURSIVE:
        return RecursiveChunker(config)
    elif config.strategy == ChunkStrategy.SEMANTIC:
        return SemanticChunker(config, embedder)
    else:
        return FixedSizeChunker(config)