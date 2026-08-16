"""
Runs the pipeline (text-mode, so STT network time doesn't dominate the
retrieval+generation numbers we actually control) over a batch of test
queries and reports P50/P70/P100 latency, plus a per-stage breakdown.

Run: python -m backend.eval.latency_bench
"""

import json
import statistics as stats

from backend.retrieval.retriever import Retriever
from backend.stt import get_stt_provider
from backend.harness.orchestrator import Orchestrator


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def run_benchmark(queries_path: str = "backend/eval/test_queries.json"):
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    retriever = Retriever()
    orchestrator = Orchestrator(retriever=retriever, stt_provider=None)

    total_times = []
    stage_times = {"retrieval": [], "generation": []}
    statuses = {}

    for q in queries:
        result = orchestrator.run(query=q)
        total_times.append(result.total_ms)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        for t in result.timings:
            if t.stage in stage_times:
                stage_times[t.stage].append(t.ms)

    print(f"\nran {len(queries)} queries")
    print("status breakdown:", statuses)
    print("\nend-to-end latency (ms):")
    print(f"  P50:  {percentile(total_times, 50):.1f}")
    print(f"  P70:  {percentile(total_times, 70):.1f}")
    print(f"  P100: {percentile(total_times, 100):.1f}")
    print(f"  mean: {stats.mean(total_times):.1f}")

    for stage, times in stage_times.items():
        if times:
            print(f"\n{stage} only (ms): P50={percentile(times,50):.1f}  P100={percentile(times,100):.1f}")

    return {
        "n": len(queries),
        "p50": percentile(total_times, 50),
        "p70": percentile(total_times, 70),
        "p100": percentile(total_times, 100),
        "statuses": statuses,
    }


if __name__ == "__main__":
    run_benchmark()
