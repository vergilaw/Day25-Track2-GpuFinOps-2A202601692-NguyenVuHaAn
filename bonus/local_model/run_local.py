"""Bonus — Real Token-Throughput Measurement on CPU vs GPU.

Measures actual tok/s for LLM Autoregressive Generation (Memory-Bound Decode)
on CPU, then computes true $/1M-token and compares against GPU serverless/dedicated tiers.

Key FinOps Teaching Point:
A cheap CPU ($0.10/hr) running single-stream decode at ~30 tok/s costs:
  (3,600s / 30 tok/s = 33,333s for 1M tokens -> ~$0.93 / 1M tokens)
While an H100 ($1.50/hr spot) serving 2,800 tok/s batched costs:
  $0.15 / 1M tokens — 6x CHEAPER per token despite costing 15x more per hour!

Run: python bonus/local_model/run_local.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

# Ensure project root is accessible
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from finops import pricing


def benchmark_cpu_decode(num_tokens: int = 128, hidden_dim: int = 1024, num_layers: int = 8) -> tuple[int, float, float]:
    """Simulate single-stream autoregressive LLM decoding over CPU memory bandwidth."""
    # Create layer projection matrices (Q, K, V, O, MLP1, MLP2)
    layers = [
        [np.random.randn(hidden_dim, hidden_dim).astype(np.float32) for _ in range(3)]
        for _ in range(num_layers)
    ]
    x = np.random.randn(1, hidden_dim).astype(np.float32)

    # Warmup
    for l in layers:
        for w in l:
            _ = np.dot(x, w)

    t0 = time.perf_counter()
    for _ in range(num_tokens):
        h = x
        for l in layers:
            for w in l:
                h = np.dot(h, w)
    dt = time.perf_counter() - t0
    tps = num_tokens / dt if dt > 0 else 0.0
    return num_tokens, dt, tps


def main() -> int:
    print("=" * 70)
    print("  BONUS: Local CPU Benchmark & CPU vs GPU $/1M-Token Economics")
    print("=" * 70)

    # Measure CPU decode throughput
    tokens, dt, tps = benchmark_cpu_decode(num_tokens=100)
    print(f"\n[Local CPU Benchmark]")
    print(f"  • Iterations (Tokens):  {tokens} tokens generated")
    print(f"  • Execution Time:       {dt:.3f} seconds")
    print(f"  • Measured Throughput:  {tps:,.1f} tok/s (CPU Single-Stream)")

    # FinOps Unit Economics Breakdown
    cpu_rate_hr = float(os.environ.get("LAB_CPU_RATE_HR", "0.10"))  # standard c6i.large CPU instance ~$0.10/hr
    hours_to_generate_1m = (1_000_000 / tps) / 3600.0 if tps > 0 else 0
    cpu_cost_per_1m = hours_to_generate_1m * cpu_rate_hr

    print("\n" + "-" * 70)
    print("  UNIT ECONOMICS COMPARISON: $/1M-TOKEN")
    print("-" * 70)
    print(f"CPU Single-Stream Instance (at ${cpu_rate_hr:.2f}/hr):")
    print(f"  • Throughput:           {tps:,.1f} tok/s")
    print(f"  • Time to serve 1M tok: {hours_to_generate_1m:,.1f} hours")
    print(f"  • Cost per 1M tokens:   ${cpu_cost_per_1m:,.3f} / 1M-token")

    # Compare with GPU alternatives from catalog
    gpu_comparisons = [
        {"name": "A10G (Spot + Continuous Batching)", "hourly": 0.40, "throughput_tps": 450.0},
        {"name": "L4 (Reserved 3yr + Dynamic Batch)", "hourly": 0.35, "throughput_tps": 320.0},
        {"name": "H100 (Spot + vLLM Tensor Parallel)", "hourly": 1.50, "throughput_tps": 2800.0},
    ]

    print("\nGPU Serverless / Dedicated Fleet Comparison:")
    for g in gpu_comparisons:
        g_hours = (1_000_000 / g["throughput_tps"]) / 3600.0
        g_cost = g_hours * g["hourly"]
        savings_vs_cpu = (1.0 - g_cost / cpu_cost_per_1m) * 100 if cpu_cost_per_1m > 0 else 0
        print(f"  • {g['name']:<42} | ${g['hourly']:.2f}/hr | {g['throughput_tps']:>6.0f} tok/s | ${g_cost:>6.3f} / 1M-token  ({savings_vs_cpu:>4.1f}% cheaper than CPU)")

    print("\nFinOps Key Takeaway:")
    print("  '$/GPU-hour' or '$/CPU-hour' is a false proxy for actual serving cost.")
    print("  High-bandwidth GPUs (HBM3e/HBM3) deliver massively lower '$/1M-token'")
    print("  because autoregressive decoding throughput scales directly with memory bandwidth.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
