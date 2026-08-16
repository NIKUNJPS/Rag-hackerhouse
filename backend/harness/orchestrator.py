"""
This is the harness the task asks for: instead of one raw prompt-in/text-out
call, every stage is a named function with its own error handling, timing,
and retry policy. The orchestrator strings them together and always returns
a structured PipelineResult, never a raw exception.
"""

import time
import functools

from backend.guardrails.safety import is_unsafe
from backend.guardrails.offtopic import is_offtopic
from backend.guardrails.grounding import is_grounded
from backend.generation.answer import generate_answer
from backend.harness.schemas import PipelineResult, StageTiming


def retry(times: int = 2, delay: float = 0.3):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt < times - 1:
                        time.sleep(delay)
            raise last_err
        return wrapper
    return deco


class Orchestrator:
    def __init__(self, retriever, stt_provider=None):
        self.retriever = retriever
        self.stt_provider = stt_provider

    @retry(times=2)
    def _transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        return self.stt_provider.transcribe(audio_bytes, filename=filename)

    @retry(times=2)
    def _retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self.retriever.retrieve(query, top_k=top_k)

    @retry(times=2)
    def _generate(self, query: str, chunks: list[dict]) -> str:
        return generate_answer(query, chunks)

    def run(self, query: str | None = None, audio_bytes: bytes | None = None,
            audio_filename: str = "audio.wav") -> PipelineResult:
        timings: list[StageTiming] = []
        t_start = time.perf_counter()

        def timed(stage_name, fn, *args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            timings.append(StageTiming(stage=stage_name, ms=(time.perf_counter() - t0) * 1000))
            return result

        try:
            # stage 1: speech to text (skipped if a text query was passed directly, e.g. for benchmarking)
            if query is None:
                if audio_bytes is None:
                    return PipelineResult(query="", status="error", error="No query or audio provided")
                query = timed("stt", self._transcribe, audio_bytes, audio_filename)

            # stage 2: safety screen
            if is_unsafe(query):
                total = (time.perf_counter() - t_start) * 1000
                return PipelineResult(query=query, status="unsafe",
                                       error="Query flagged by safety guardrail",
                                       timings=timings, total_ms=total)

            # stage 3: retrieval
            chunks = timed("retrieval", self._retrieve, query)

            # stage 4: off-topic guardrail (based on top retrieval score)
            top_score = chunks[0]["score"] if chunks else 0.0
            if is_offtopic(top_score):
                total = (time.perf_counter() - t_start) * 1000
                return PipelineResult(query=query, status="offtopic",
                                       error="Query not covered by this dataset",
                                       chunks_used=chunks, timings=timings, total_ms=total)

            # stage 5: generation
            answer = timed("generation", self._generate, query, chunks)

            # stage 6: grounding check
            if not is_grounded(answer, chunks):
                total = (time.perf_counter() - t_start) * 1000
                return PipelineResult(
                    query=query,
                    answer="I found related information but can't confidently answer that from it.",
                    status="ungrounded", chunks_used=chunks, timings=timings, total_ms=total,
                )

            total = (time.perf_counter() - t_start) * 1000
            return PipelineResult(query=query, answer=answer, status="ok",
                                   chunks_used=chunks, timings=timings, total_ms=total)

        except Exception as e:
            total = (time.perf_counter() - t_start) * 1000
            return PipelineResult(query=query or "", status="error", error=str(e),
                                   timings=timings, total_ms=total)
