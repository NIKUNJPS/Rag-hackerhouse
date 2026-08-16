"""
Builds a small in-memory index separately for each chunking strategy and
compares retrieval hit rate on a labeled sample (query -> expected doc_id).
This is what justified picking 'metadata_aware' as the default in
build_index.py -- rerun this after changing the dataset sample to re-check.

Run: python -m backend.eval.chunking_eval
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.ingestion.chunking import chunk_document
from backend.retrieval.vector_store import VectorStore


def build_temp_index(records: list[dict], strategy: str, embedder) -> VectorStore:
    all_chunks = []
    for r in records:
        chunks = chunk_document(r["text"], doc_id=r["doc_id"], title=r.get("title", ""),
                                 section=r.get("section", ""), strategy=strategy, embedder=embedder)
        all_chunks.extend(chunks)

    texts = [c.text for c in all_chunks]
    vectors = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "strategy": c.strategy}
        for c in all_chunks
    ])
    return store


def hit_rate(store: VectorStore, embedder, labeled_queries: list[dict], top_k: int = 5) -> float:
    hits = 0
    for lq in labeled_queries:
        qvec = embedder.encode([lq["query"]], convert_to_numpy=True)[0].astype("float32")
        results = store.search(qvec, top_k=top_k)
        if any(r["doc_id"] == lq["expected_doc_id"] for r in results):
            hits += 1
    return hits / len(labeled_queries) if labeled_queries else 0.0


EVAL_QUERY_GROUPS = 250  # ~2.5k passage records -- keeps the 3x re-embed fast


def _first_n_query_groups(records: list[dict], n: int) -> list[dict]:
    """
    Slices by whole query groups (not a raw line count) so a query's selected
    passage is never split off from its sibling candidates by the cutoff --
    make_query_sets.py uses the same helper so labeled_queries.json always
    references doc_ids that are actually inside this eval's temp index.
    """
    seen_queries = []
    seen_set = set()
    for r in records:
        qid = r.get("query_id", r["doc_id"])
        if qid not in seen_set:
            if len(seen_set) >= n:
                break
            seen_set.add(qid)
            seen_queries.append(qid)
    return [r for r in records if r.get("query_id", r["doc_id"]) in seen_set]


def run(records_path="backend/data/raw_records.jsonl", labeled_queries_path="backend/eval/labeled_queries.json"):
    with open(records_path, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]
    records = _first_n_query_groups(all_records, EVAL_QUERY_GROUPS)
    with open(labeled_queries_path, "r", encoding="utf-8") as f:
        labeled_queries = json.load(f)

    embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    for strategy in ["fixed_size", "semantic", "metadata_aware"]:
        store = build_temp_index(records, strategy, embedder)
        rate = hit_rate(store, embedder, labeled_queries)
        print(f"{strategy:15s} hit_rate={rate:.2%}  chunks={len(store.chunks)}")


if __name__ == "__main__":
    run()
