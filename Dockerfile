FROM python:3.11-slim

WORKDIR /app

# faiss + torch need a couple of system libs on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch build explicitly -- pip's default wheel bundles CUDA, which
# is ~2GB+ and useless on any free-tier host (no GPU anyway). This alone cuts
# the image and build time roughly in half.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# Pre-download the embedding model at build time so the first request after
# a cold start isn't stuck downloading ~470MB — bakes it into the image layer.
RUN python -c "from sentence_transformers import SentenceTransformer; import os; \
    SentenceTransformer(os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))"

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
