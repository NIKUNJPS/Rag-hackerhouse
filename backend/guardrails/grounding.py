"""
Cheap but effective hallucination check: after the LLM answers, verify a
reasonable fraction of the answer's content words actually appear in the
retrieved context. This won't catch subtle factual drift, but it reliably
catches the two failure modes that matter most in a demo: the model
ignoring context entirely, or answering from outside knowledge.
"""

import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "to", "for", "and", "or", "it", "this", "that", "as", "by", "with",
    "be", "has", "have", "had", "not", "i", "you", "we", "they", "do",
    "does", "did", "can", "could", "would", "should", "will",
}


def _content_words(text: str) -> set[str]:
    # \w is Unicode-aware in Python 3 str patterns, so this covers Devanagari
    # (the dataset is Hindi) as well as ASCII -- an [a-zA-Z]-only pattern would
    # match zero words in Hindi text and make every answer look ungrounded.
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def grounding_score(answer: str, chunks: list[dict]) -> float:
    """Fraction of the answer's content words that appear somewhere in the retrieved chunks."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words = set()
    for c in chunks:
        context_words |= _content_words(c["text"])
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def is_grounded(answer: str, chunks: list[dict], threshold: float | None = None) -> bool:
    from backend.config import settings
    threshold = threshold if threshold is not None else settings.GROUNDING_OVERLAP_THRESHOLD
    return grounding_score(answer, chunks) >= threshold
