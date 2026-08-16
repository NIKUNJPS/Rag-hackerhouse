from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.store = VectorStore.load(settings.VECTOR_INDEX_DIR)

    def embed_query(self, query: str):
        return self.embedder.encode([query], convert_to_numpy=True)[0].astype("float32")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.TOP_K
        qvec = self.embed_query(query)
        return self.store.search(qvec, top_k=top_k)
