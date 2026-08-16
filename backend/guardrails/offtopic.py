"""
Decides whether a query is even in scope for the dataset before we spend
time retrieving+generating for it. Uses the same embedder as retrieval:
if the query's best match in the index scores below threshold, it's
probably off-topic for this knowledge base.
"""

from backend.config import settings


def is_offtopic(top_retrieval_score: float) -> bool:
    return top_retrieval_score < settings.OFFTOPIC_SIM_THRESHOLD
