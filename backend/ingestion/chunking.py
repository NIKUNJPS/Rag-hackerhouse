"""
Three different chunking strategies, plus a router that picks between them
based on document shape. The eval script (backend/eval) compares retrieval
quality across all three so we can justify which one actually ships.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    strategy: str
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    # cheap sentence splitter, good enough for MSMARCO-style passages
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def fixed_size_chunks(text: str, doc_id: str, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Baseline: fixed word-count windows with overlap so we don't cut answers in half."""
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = start + size
        piece = " ".join(words[start:end])
        chunks.append(Chunk(
            text=piece,
            doc_id=doc_id,
            chunk_id=f"{doc_id}-fixed-{idx}",
            strategy="fixed_size",
            metadata={"start_word": start, "end_word": min(end, len(words))},
        ))
        idx += 1
        start += size - overlap
    return chunks


def semantic_chunks(text: str, doc_id: str, max_sentences: int = 6, sim_drop_threshold: float = 0.35,
                     embedder=None) -> list[Chunk]:
    """
    Groups sentences by semantic similarity instead of a fixed word count.
    Walks sentence by sentence and starts a new chunk when the topic drifts
    (cosine similarity to the running chunk centroid drops below threshold)
    or when we hit max_sentences.
    Falls back to sentence-count grouping if no embedder is passed in
    (keeps this function usable standalone / in tests without loading a model).
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    if embedder is None:
        # fallback: group every max_sentences sentences
        groups = [sentences[i:i + max_sentences] for i in range(0, len(sentences), max_sentences)]
    else:
        import numpy as np
        vecs = embedder.encode(sentences, normalize_embeddings=True)
        groups, current, current_vecs = [], [sentences[0]], [vecs[0]]
        for sent, vec in zip(sentences[1:], vecs[1:]):
            centroid = np.mean(current_vecs, axis=0)
            sim = float(np.dot(centroid, vec) / (np.linalg.norm(centroid) * np.linalg.norm(vec) + 1e-8))
            if sim < sim_drop_threshold or len(current) >= max_sentences:
                groups.append(current)
                current, current_vecs = [sent], [vec]
            else:
                current.append(sent)
                current_vecs.append(vec)
        if current:
            groups.append(current)

    chunks = []
    for i, group in enumerate(groups):
        chunks.append(Chunk(
            text=" ".join(group),
            doc_id=doc_id,
            chunk_id=f"{doc_id}-sem-{i}",
            strategy="semantic",
            metadata={"num_sentences": len(group)},
        ))
    return chunks


def metadata_aware_chunks(text: str, doc_id: str, title: str = "", section: str = "",
                           size: int = 180, overlap: int = 30) -> list[Chunk]:
    """
    Same fixed-window splitting as above, but every chunk carries doc-level
    metadata (title/section) so retrieval can filter/boost by it and the
    generation step can cite where an answer came from.
    """
    base = fixed_size_chunks(text, doc_id, size=size, overlap=overlap)
    out = []
    for c in base:
        c.strategy = "metadata_aware"
        c.chunk_id = c.chunk_id.replace("fixed", "meta")
        c.metadata.update({"title": title, "section": section})
        out.append(c)
    return out


def chunk_document(text: str, doc_id: str, title: str = "", section: str = "",
                    strategy: str = "hybrid", embedder=None) -> list[Chunk]:
    """
    Router. 'hybrid' runs metadata-aware chunking for the primary index
    (best balance of precision + traceability) but this is easy to swap
    per experiment -- see backend/eval/chunking_eval.py for the comparison
    that justified picking metadata_aware as default.
    """
    if strategy == "fixed_size":
        return fixed_size_chunks(text, doc_id)
    if strategy == "semantic":
        return semantic_chunks(text, doc_id, embedder=embedder)
    if strategy == "metadata_aware":
        return metadata_aware_chunks(text, doc_id, title=title, section=section)
    if strategy == "hybrid":
        return metadata_aware_chunks(text, doc_id, title=title, section=section)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
