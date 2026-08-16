from backend.generation.llm_client import generate

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
    return generate(prompt, system=SYSTEM_PROMPT, max_tokens=180)
