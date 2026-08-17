from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.retrieval.retriever import Retriever
from backend.stt import get_stt_provider
from backend.harness.orchestrator import Orchestrator

app = FastAPI(title="Voice RAG - HH Goa 2026")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# loaded once at startup, reused across requests -- loading the model/index
# per-request would blow the latency budget immediately
retriever = None
orchestrator = None


@app.on_event("startup")
def load_models():
    global retriever, orchestrator
    retriever = Retriever()
    stt = get_stt_provider()
    orchestrator = Orchestrator(retriever=retriever, stt_provider=stt)
    _warm_up_llm_connection()


def _warm_up_llm_connection():
    """
    The LLM SDK client is created lazily on first use and reused after that
    (backend/generation/llm_client.py), but "first use" still means a fresh
    TLS handshake -- which was costing 1-2.5s on top of actual inference on
    whichever request happened to be first. Paying that cost once at server
    boot instead of on a real user's first request is a pure win: nothing
    about it affects correctness, and it's best-effort (a warm-up failure
    shouldn't block the server from starting).
    """
    try:
        from backend.generation.llm_client import generate
        generate("ping", system="Reply with one word.", max_tokens=5)
    except Exception:
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


MAX_QUERY_CHARS = 2000
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25MB -- generous for a spoken question, cheap to reject above that

_EMPTY_RESULT = {"chunks_used": [], "timings": [], "total_ms": 0.0}


@app.post("/ask/text")
def ask_text(query: str = Form(...)):
    """Text-only entry point, mainly used by the latency benchmark and for debugging."""
    query = query.strip()
    if not query:
        return {"query": "", "status": "error", "error": "Empty query", **_EMPTY_RESULT}
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS]
    result = orchestrator.run(query=query)
    return result.model_dump()


@app.post("/ask/voice")
async def ask_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"query": "", "status": "error", "error": "Empty audio upload", **_EMPTY_RESULT}
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return {"query": "", "status": "error", "error": "Audio upload too large", **_EMPTY_RESULT}
    result = orchestrator.run(audio_bytes=audio_bytes, audio_filename=file.filename or "audio.wav")
    return result.model_dump()


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
