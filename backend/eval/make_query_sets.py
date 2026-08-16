"""
Generates real test_queries.json (list of query strings) and labeled_queries.json
(query -> expected doc_id) from the actual built dataset, replacing the placeholder
files.

MSMARCO-XI gives each query several candidate passages plus an `is_selected` flag
marking the one(s) that actually answer it (see backend/ingestion/load_dataset.py).
labeled_queries.json uses that real relevance label as ground truth -- this is not
a synthetic "title matches its own doc" eval, it's the dataset's own judgment of
which passage is correct among several plausible distractors for the same query.

Run: python -m backend.eval.make_query_sets
"""

import json
import random
from collections import defaultdict

from backend.eval.chunking_eval import EVAL_QUERY_GROUPS, _first_n_query_groups

RECORDS_PATH = "backend/data/raw_records.jsonl"
TEST_QUERIES_PATH = "backend/eval/test_queries.json"
LABELED_QUERIES_PATH = "backend/eval/labeled_queries.json"


def _group(records):
    by_query = defaultdict(list)
    for r in records:
        by_query[r.get("query_id", r["title"])].append(r)
    return list(by_query.values())


def run(n_test: int = 40, n_labeled: int = 80, seed: int = 7):
    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]

    # test_queries.json samples from the whole dataset (no coverage constraint --
    # it's just used for the latency benchmark and UI demo chips).
    rng = random.Random(seed)
    all_groups = _group(all_records)
    rng.shuffle(all_groups)
    test_queries = [g[0]["title"] for g in all_groups[:n_test] if g[0]["title"].strip()]

    # labeled_queries.json MUST be drawn from the same first-N-query-groups window
    # backend/eval/chunking_eval.py uses to build its temp index, otherwise the
    # expected_doc_id would point at a passage that isn't even in the index being
    # scored -- see chunking_eval.py's _first_n_query_groups docstring.
    eval_pool = _group(_first_n_query_groups(all_records, EVAL_QUERY_GROUPS))
    rng.shuffle(eval_pool)

    labeled = []
    for g in eval_pool:
        selected = [r for r in g if r.get("is_selected")]
        if not selected or not g[0]["title"].strip():
            continue
        labeled.append({"query": g[0]["title"], "expected_doc_id": selected[0]["doc_id"]})
        if len(labeled) >= n_labeled:
            break

    with open(TEST_QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)
    with open(LABELED_QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(test_queries)} test queries -> {TEST_QUERIES_PATH}")
    print(f"wrote {len(labeled)} labeled queries (real is_selected ground truth) -> {LABELED_QUERIES_PATH}")


if __name__ == "__main__":
    run()
