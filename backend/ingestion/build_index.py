"""
Builds the vector index from raw dataset records:
load records -> chunk (metadata_aware by default) -> embed -> store in FAISS.

Run: python -m backend.ingestion.build_index
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.ingestion.chunking import chunk_document
from backend.retrieval.vector_store import VectorStore


def build(strategy: str = "hybrid", records_path: str = "backend/data/raw_records.jsonl"):
    with open(records_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    all_chunks = []
    for r in records:
        chunks = chunk_document(
            r["text"], doc_id=r["doc_id"], title=r.get("title", ""),
            section=r.get("section", ""), strategy=strategy, embedder=embedder,
        )
        all_chunks.extend(chunks)

    print(f"chunked {len(records)} docs into {len(all_chunks)} chunks using '{strategy}' strategy")

    texts = [c.text for c in all_chunks]
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    vectors = vectors.astype("float32")

    store = VectorStore(dim=vectors.shape[1])
    chunk_records = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
         "strategy": c.strategy, "metadata": c.metadata}
        for c in all_chunks
    ]
    store.add(vectors, chunk_records)
    store.save(settings.VECTOR_INDEX_DIR)
    print(f"index saved to {settings.VECTOR_INDEX_DIR}")


if __name__ == "__main__":
    build()
