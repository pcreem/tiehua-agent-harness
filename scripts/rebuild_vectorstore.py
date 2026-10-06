#!/usr/bin/env python3
"""
Rebuild vectorstore from attractions_v2.json (6,176 records).

Uses local multilingual embedding model (sentence-transformers).
Default: intfloat/multilingual-e5-small (384 dim, fast, good quality)
Can switch to: intfloat/multilingual-e5-large (1024 dim, better quality)
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.ingestion import IngestionPipeline
from rag.types import Document, ChunkConfig, VectorStoreConfig
from rag.chunking import create_chunker
from rag.vectorstore import VectorStoreFactory
from sentence_transformers import SentenceTransformer


# Use small multilingual model for speed (384 dim)
# Change to "intfloat/multilingual-e5-large" for 1024 dim better quality
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class LocalEmbedder:
    """Local embedding using sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        print(f"Model loaded. Dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # E5 models need "passage: " prefix for best results
        prefixed_texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(prefixed_texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()


def attractions_to_documents(attractions: List[Dict[str, Any]]) -> List[Document]:
    """Convert attraction records to Documents for vectorstore."""
    documents = []

    for attr in attractions:
        parts = []

        parts.append(f"名稱：{attr['names']['zh-TW']}")
        if attr['description']['zh-TW']:
            parts.append(f"描述：{attr['description']['zh-TW']}")

        parts.append(f"類別：{attr['category']}")
        if attr['class_names']['zh-TW']:
            parts.append(f"分類：{attr['class_names']['zh-TW']}")

        loc = attr['location']
        parts.append(f"地址：{loc['address']['zh-TW']}")
        parts.append(f"城市：{loc['city']}")
        if loc['district']:
            parts.append(f"行政區：{loc['district']}")

        attrs = attr['attributes']
        if attrs.get('fees'):
            fee_text = "、".join([
                f"{f.get('Name', f.get('name', ''))}{f.get('Price', f.get('price', ''))}元"
                if f.get('Price') or f.get('price')
                else f.get('Name', f.get('name', ''))
                for f in attrs['fees']
            ])
            parts.append(f"票價：{fee_text}")

        if attrs.get('service_times'):
            st_text = "、".join([
                f"{st.get('Name', st.get('name', ''))}{st.get('StartTime', st.get('startTime', ''))}-{st.get('EndTime', st.get('endTime', ''))}"
                if st.get('StartTime') or st.get('startTime')
                else st.get('Description', st.get('description', ''))
                for st in attrs['service_times']
            ])
            parts.append(f"營業時間：{st_text}")

        if attrs.get('parking'):
            parts.append(f"停車資訊：{attrs['parking']}")

        if attrs.get('traffic_info'):
            parts.append(f"交通資訊：{attrs['traffic_info'][:500]}")

        if attrs.get('phone_numbers'):
            parts.append(f"電話：{', '.join(attrs['phone_numbers'])}")

        content = "\n".join(parts)

        metadata = {
            "attraction_id": attr["id"],
            "name_zh": attr["names"]["zh-TW"],
            "category": attr["category"],
            "all_categories": attr["all_categories"],
            "city": loc["city"],
            "district": loc["district"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "language": "zh-TW",
            "source": attr["provenance"]["source"],
            "source_url": attr["provenance"]["source_url"],
            "confidence": attr["quality"]["confidence"],
            "has_fees": attr["quality"]["has_fees"],
            "has_service_times": attr["quality"]["has_service_times"],
        }

        doc = Document(
            id=attr["id"],
            content=content,
            metadata=metadata,
            source=attr["provenance"]["source_url"],
        )
        documents.append(doc)

    return documents


def main():
    print("Loading attractions_v2.json...")
    v2_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "attractions_v2.json"

    if not v2_path.exists():
        print(f"❌ File not found: {v2_path}")
        print("Run scripts/merge_attraction_data.py first")
        sys.exit(1)

    with open(v2_path, encoding="utf-8") as f:
        data = json.load(f)

    attractions = data["attractions"]
    print(f"Loaded {len(attractions)} attractions")

    print("\nConverting to documents...")
    documents = attractions_to_documents(attractions)
    print(f"Created {len(documents)} documents")

    print("\nInitializing local multilingual embedder...")
    embedder = LocalEmbedder()

    print("\nBuilding ingestion pipeline...")
    import shutil
    vs_path = Path("./data/vectorstore")
    if vs_path.exists():
        shutil.rmtree(vs_path)
    vs_path.mkdir(parents=True, exist_ok=True)

    chunk_config = ChunkConfig(
        strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100,
        min_chunk_size=50,
    )

    chunker = create_chunker(chunk_config)
    vector_config = VectorStoreConfig(
        provider="chroma",
        persist_dir=str(vs_path),
        collection_name="tiehua_rag",
    )
    vector_store = VectorStoreFactory.create(vector_config)

    class EmbedderWrapper:
        def __init__(self, local_embedder):
            self.local_embedder = local_embedder

        def embed(self, texts: List[str]) -> List[List[float]]:
            return self.local_embedder.embed(texts)

    pipeline = IngestionPipeline(
        chunker=chunker,
        embedder=EmbedderWrapper(embedder),
        vector_store=vector_store,
        chunk_config=chunk_config,
    )

    print("\nIngesting documents in batches...")
    results = pipeline.ingest_batch(documents)

    success_count = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunks_created for r in results if r.success)

    print(f"\n✅ Ingestion complete:")
    print(f"  Successful: {success_count}/{len(documents)}")
    print(f"  Total chunks created: {total_chunks}")

    if success_count < len(documents):
        print("\n⚠️  Failed documents:")
        for r in results:
            if not r.success:
                print(f"  - {r.document_id}: {r.error}")

    client = vector_store.client
    collection = client.get_collection("tiehua_rag")
    print(f"  Vectorstore count: {collection.count()}")


if __name__ == "__main__":
    main()