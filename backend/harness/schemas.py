from typing import Optional
from pydantic import BaseModel


class StageTiming(BaseModel):
    stage: str
    ms: float


class PipelineResult(BaseModel):
    query: str
    answer: Optional[str] = None
    chunks_used: list[dict] = []
    status: str  # "ok" | "offtopic" | "unsafe" | "ungrounded" | "error"
    error: Optional[str] = None
    timings: list[StageTiming] = []
    total_ms: float = 0.0
