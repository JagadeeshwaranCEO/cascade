"""
eval.benchmark
~~~~~~~~~~~~~~~
Benchmark suite for Cascade.

Compares three systems on a standard task set:
  1. all_remote  — Fireworks AI only (baseline cost)
  2. all_local   — AMD GPU only (baseline quality ceiling)
  3. cascade     — Adaptive runtime (our system)

Metrics:
  - Token count (primary Track 1 scoring criterion)
  - Accuracy / quality score
  - Cost (USD)
  - Latency (ms)
  - Token waste rate

Run before the hackathon eval to verify routing intelligence.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Task set — extended on kickoff day with revealed tasks
BENCHMARK_TASKS = [
    {
        "id": "math_trivial",
        "prompt": "What is 127 × 43?",
        "domain": "math",
        "expected_local": True,
        "difficulty": "easy",
    },
    {
        "id": "factual_simple",
        "prompt": "What is the capital of Japan?",
        "domain": "factual",
        "expected_local": True,
        "difficulty": "easy",
    },
    {
        "id": "code_medium",
        "prompt": "Write a Python function that performs binary search on a sorted list.",
        "domain": "coding",
        "expected_local": True,
        "difficulty": "medium",
    },
    {
        "id": "code_hard",
        "prompt": "Implement a lock-free concurrent queue in C++ using compare-and-swap.",
        "domain": "coding",
        "expected_local": False,
        "difficulty": "hard",
    },
    {
        "id": "legal_medium",
        "prompt": "Explain force majeure clauses and when they apply in contract law.",
        "domain": "legal",
        "expected_local": False,
        "difficulty": "hard",
    },
    {
        "id": "reasoning_hard",
        "prompt": "If all A are B, and some B are C, what can we conclude about the relationship between A and C? Explain your reasoning.",
        "domain": "reasoning",
        "expected_local": False,
        "difficulty": "hard",
    },
    {
        "id": "creative_simple",
        "prompt": "Write a haiku about machine learning.",
        "domain": "creative",
        "expected_local": True,
        "difficulty": "easy",
    },
    {
        "id": "explain_medium",
        "prompt": "Explain transformer self-attention in simple terms.",
        "domain": "factual",
        "expected_local": True,
        "difficulty": "medium",
    },
    {
        "id": "system_design_hard",
        "prompt": "Design the high-level architecture for a distributed key-value store that supports strong consistency.",
        "domain": "coding",
        "expected_local": False,
        "difficulty": "hard",
    },
    {
        "id": "math_medium",
        "prompt": "Solve: integrate x^2 * sin(x) dx using integration by parts.",
        "domain": "math",
        "expected_local": False,
        "difficulty": "hard",
    },
]


@dataclass
class BenchmarkResult:
    task_id:       str
    system:        str
    difficulty:    str
    route_taken:   str
    latency_ms:    float
    cost_usd:      float
    quality_score: float
    tokens_local:  int
    tokens_remote: int
    tokens_wasted: int
    correct_route: bool   # Did the system route as expected?


async def run_cascade_benchmark(
    controller,
    tasks: list[dict] | None = None,
    n_runs: int = 1,
) -> list[BenchmarkResult]:
    """Run task set through Cascade and collect BenchmarkResult per task."""
    from cascade.core.inference_process import InferenceProcess

    tasks = tasks or BENCHMARK_TASKS
    results: list[BenchmarkResult] = []

    for task in tasks:
        for run in range(n_runs):
            log.info("Benchmarking [%s] run %d/%d", task["id"], run + 1, n_runs)
            process = InferenceProcess(prompt=task["prompt"])
            t0 = time.monotonic()
            result = await controller.run(process)
            latency = (time.monotonic() - t0) * 1_000

            expected_local = task.get("expected_local", True)
            actual_local   = "local" in result.route_taken
            correct_route  = expected_local == actual_local

            results.append(BenchmarkResult(
                task_id=task["id"],
                system="cascade",
                difficulty=task.get("difficulty", "unknown"),
                route_taken=result.route_taken,
                latency_ms=round(latency, 1),
                cost_usd=result.cost_actual_usd,
                quality_score=result.quality_score,
                tokens_local=result.tokens_local,
                tokens_remote=result.tokens_remote,
                tokens_wasted=result.tokens_wasted,
                correct_route=correct_route,
            ))

            log.info(
                "  [%s] route=%s latency=%.0fms quality=%.3f correct=%s",
                task["id"], result.route_taken, latency,
                result.quality_score, "✓" if correct_route else "✗",
            )

    return results


def summarize(results: list[BenchmarkResult], system: str = "cascade") -> dict:
    """Aggregate statistics for a system's results."""
    rs = [r for r in results if r.system == system]
    if not rs:
        return {}

    n          = len(rs)
    local_rs   = [r for r in rs if "local"  in r.route_taken]
    remote_rs  = [r for r in rs if "remote" in r.route_taken]
    correct    = [r for r in rs if r.correct_route]

    total_tokens  = sum(r.tokens_local + r.tokens_remote for r in rs)
    remote_tokens = sum(r.tokens_remote for r in rs)
    wasted_tokens = sum(r.tokens_wasted for r in rs)

    # What all-remote would have cost
    all_remote_cost = (total_tokens / 1_000) * 0.0009
    actual_cost     = sum(r.cost_usd for r in rs)
    savings         = max(0.0, all_remote_cost - actual_cost)

    return {
        "system":              system,
        "n_tasks":             n,
        "routing_accuracy":    round(len(correct) / n * 100, 1),
        "local_route_pct":     round(len(local_rs)  / n * 100, 1),
        "remote_route_pct":    round(len(remote_rs) / n * 100, 1),
        "avg_latency_ms":      round(sum(r.latency_ms    for r in rs) / n, 1),
        "avg_quality_score":   round(sum(r.quality_score for r in rs) / n, 3),
        "total_tokens":        total_tokens,
        "remote_tokens":       remote_tokens,
        "wasted_tokens":       wasted_tokens,
        "waste_pct":           round(wasted_tokens / max(total_tokens, 1) * 100, 1),
        "total_cost_usd":      round(actual_cost, 6),
        "all_remote_cost_usd": round(all_remote_cost, 6),
        "savings_usd":         round(savings, 6),
        "savings_pct":         round(savings / max(all_remote_cost, 1e-9) * 100, 1),
    }


def print_report(results: list[BenchmarkResult]):
    """Print a clean benchmark report to stdout."""
    s = summarize(results)
    sep = "─" * 60
    print(f"\n{sep}")
    print("  CASCADE BENCHMARK REPORT")
    print(sep)
    print(f"  Tasks evaluated       : {s['n_tasks']}")
    print(f"  Routing accuracy      : {s['routing_accuracy']}%")
    print(f"  Local route           : {s['local_route_pct']}%")
    print(f"  Remote route          : {s['remote_route_pct']}%")
    print(f"  Avg latency           : {s['avg_latency_ms']} ms")
    print(f"  Avg quality score     : {s['avg_quality_score']}")
    print(f"  Total tokens          : {s['total_tokens']}")
    print(f"  Wasted tokens         : {s['wasted_tokens']} ({s['waste_pct']}%)")
    print(f"  Actual cost           : ${s['total_cost_usd']:.6f}")
    print(f"  All-remote cost       : ${s['all_remote_cost_usd']:.6f}")
    print(f"  ✓ Savings             : ${s['savings_usd']:.6f} ({s['savings_pct']}%)")
    print(sep)

    print("\n  Per-task breakdown:")
    print(f"  {'Task':<25} {'Route':<18} {'Lat(ms)':>8} {'Quality':>8} {'Correct':>8}")
    print("  " + "─" * 72)
    for r in results:
        c = "✓" if r.correct_route else "✗"
        print(f"  {r.task_id:<25} {r.route_taken:<18} {r.latency_ms:>8.0f} "
              f"{r.quality_score:>8.3f} {c:>8}")
    print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    async def main():
        from cascade.core.runtime_controller import RuntimeController
        controller = RuntimeController()
        results = await run_cascade_benchmark(controller, n_runs=1)
        print_report(results)
        Path("benchmark_results.json").write_text(
            json.dumps([asdict(r) for r in results], indent=2)
        )

    asyncio.run(main())
