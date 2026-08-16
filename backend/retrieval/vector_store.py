"""
Thin wrapper around FAISS. Kept local/in-memory (not a hosted vector DB
service) deliberately -- a network round trip to a hosted vector DB is
usually the single biggest latency cost in a RAG pipeline, and we're
targeting <200ms end to end.
"""

import json
import os
import numpy as np
import faiss


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized vectors + inner product
        self.chunks: list[dict] = []  # parallel array: index i -> chunk metadata

    def add(self, vectors: np.ndarray, chunk_records: list[dict]):
        assert vectors.shape[0] == len(chunk_records)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunk_records)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        query_vector = query_vector.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vector)
        scores, idxs = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            record = dict(self.chunks[idx])
            record["score"] = float(score)
            results.append(record)
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        store = cls(dim=index.d)
        store.index = index
        with open(os.path.join(path, "chunks.jsonl"), "r", encoding="utf-8") as f:
            store.chunks = [json.loads(line) for line in f]
        return store
