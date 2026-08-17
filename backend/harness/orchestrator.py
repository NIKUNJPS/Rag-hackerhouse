"""
This is the harness the task asks for: instead of one raw prompt-in/text-out
call, every stage is a named function with its own error handling, timing,
and retry policy. The orchestrator strings them together and always returns
a structured PipelineResult, never a raw exception.
"""

import random
import time
import functools

from backend.guardrails.safety import is_unsafe
from backend.guardrails.offtopic import is_offtopic
from backend.guardrails.grounding import is_grounded
from backend.generation.answer import generate_answer_stream
from backend.harness.schemas import PipelineResult, StageTiming


def _is_rate_limit_error(e: Exception) -> bool:
    """
    Best-effort duck-typed check across SDKs (Anthropic/Groq/OpenAI all raise
    their own RateLimitError subclasses, requests/urllib raise on HTTP 429) --
    checking status_code/message text instead of importing every SDK's
    exception type keeps this decorator provider-agnostic.
    """
    status = getattr(e, "status_code", None)
    if status == 429:
        return True
    return "429" in str(e) or "rate limit" in str(e).lower()


def retry(times: int = 2, delay: float = 0.3, backoff: float = 2.0):
    """
    An automated eval loop firing 10+ queries back-to-back is exactly the
    scenario that trips API rate limits -- a flat 0.3s retry delay burns
    through both attempts before a rate limit resets. Rate-limit errors get
    exponential backoff with jitter and an extra attempt; every other
    transient error still gets the fast flat retry (no reason to slow down a
    one-off network blip).
    """
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
                        if _is_rate_limit_error(e):
                            sleep_for = delay * (backoff ** attempt) + random.uniform(0, 0.25)
                        else:
                            sleep_for = delay
                        time.sleep(sleep_for)
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

    @retry(times=3)
    def _generate_with_ttft(self, query: str, chunks: list[dict]) -> tuple[str, float, float]:
        """
        Consumes the streamed answer and splits timing into time-to-first-token
        (how long until the model starts responding -- the number that can
        honestly approach 200ms) and total generation time (which necessarily
        grows with answer length regardless of provider speed). Wrapped by the
        same retry() as the other stages -- streaming doesn't raise until
        iterated, so the retry has to wrap consumption, not just the call that
        creates the generator.
        """
        t0 = time.perf_counter()
        ttft_ms = None
        parts = []
        for piece in generate_answer_stream(query, chunks):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            parts.append(piece)
        total_ms = (time.perf_counter() - t0) * 1000
        return "".join(parts), (ttft_ms if ttft_ms is not None else total_ms), total_ms

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

            # stage 5: generation (streamed -- ttft and total are both recorded)
            answer, ttft_ms, gen_total_ms = self._generate_with_ttft(query, chunks)
            timings.append(StageTiming(stage="generation_ttft", ms=ttft_ms))
            timings.append(StageTiming(stage="generation", ms=gen_total_ms))

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
