from backend.generation.llm_client import generate, generate_stream

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context passages provided below. "
    "If the answer isn't in the context, say clearly that you don't have "
    "enough information -- do not guess or use outside knowledge. "
    "Keep answers short and direct, suitable for reading aloud."
)


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"Context passages:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the passages above. Cite passage numbers like [1] where relevant."
    )


def generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have enough information in the dataset to answer that."
    prompt = build_prompt(query, chunks)
    return generate(prompt, system=SYSTEM_PROMPT, max_tokens=220)


def generate_answer_stream(query: str, chunks: list[dict]):
    """Same prompt as generate_answer, but yields text chunks -- lets the
    orchestrator measure time-to-first-token separately from total time."""
    if not chunks:
        def _empty():
            yield "I don't have enough information in the dataset to answer that."
        return _empty()
    prompt = build_prompt(query, chunks)
    return generate_stream(prompt, system=SYSTEM_PROMPT, max_tokens=180)
