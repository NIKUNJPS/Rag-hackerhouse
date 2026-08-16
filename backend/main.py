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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask/text")
def ask_text(query: str = Form(...)):
    """Text-only entry point, mainly used by the latency benchmark and for debugging."""
    result = orchestrator.run(query=query)
    return result.model_dump()


@app.post("/ask/voice")
async def ask_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"query": "", "status": "error", "error": "Empty audio upload", "timings": [], "total_ms": 0.0, "chunks_used": []}
    result = orchestrator.run(audio_bytes=audio_bytes, audio_filename=file.filename or "audio.wav")
    return result.model_dump()


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
