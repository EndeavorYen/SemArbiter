"""Live physical inference benchmark for Apple Silicon Mac mini using MLX."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

SAMPLE_DECISION = {
    "id": "live-m4-test-1",
    "state": "The user reported an unexpected charge of $49 on their invoice. The account tier is Gold VIP.",
    "question": "Which action should the customer support agent take?",
    "options": [
        {"id": "refund", "description": "Issue an immediate refund for the dispute."},
        {"id": "escalate", "description": "Escalate the case to Tier-2 Billing Investigation."},
        {"id": "deny", "description": "Deny the dispute based on the terms of service."},
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Hugging Face repo id for the MLX quantized model",
    )
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--output", type=Path, default=Path("results/phase2-mac-m4-real-benchmark.json"))
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as exc:
        raise SystemExit(
            "Error: MLX is not installed. Run this benchmark on Apple Silicon macOS with `pip install mlx mlx-lm`."
        ) from exc

    print("=" * 80)
    print("  SemIf Physical Hardware Benchmark: Apple Silicon Mac mini")
    print(f"  Machine: {platform.machine()} | OS: {platform.system()} {platform.release()}")
    try:
        dev_info = mx.device_info()
    except Exception:
        dev_info = mx.metal.device_info()
    print(f"  GPU Device: {dev_info.get('device_name', 'Apple Silicon')} ({dev_info.get('architecture', 'UMA')})")
    print(f"  Total Unified Memory: {dev_info.get('memory_size', 0) / (1024**3):.1f} GB")
    print(f"  Loading Model: {args.model} ...")
    print("=" * 80)

    load_start = time.perf_counter()
    mx.metal.reset_peak_memory()
    model, tokenizer = load(args.model)
    load_seconds = time.perf_counter() - load_start
    model_peak_ram_mb = mx.metal.get_peak_memory() / (1024 * 1024)
    print(f"  Model loaded in {load_seconds:.2f}s | Peak VRAM: {model_peak_ram_mb:.1f} MB")

    from semif_phase1.mlx_engine import score_mlx

    metadata = {"source": args.model, "engine": "mlx-apple-silicon"}

    # Warmup
    print("  Warming up MLX execution pipeline...")
    for _ in range(5):
        _ = score_mlx(model, tokenizer, SAMPLE_DECISION, metadata)
    mx.eval()

    # Live benchmarking
    print(f"  Executing {args.iterations} real decision forward passes...")
    latencies_ms = []
    forward_ms = []
    mx.metal.reset_peak_memory()

    for _ in range(args.iterations):
        res = score_mlx(model, tokenizer, SAMPLE_DECISION, metadata)
        latencies_ms.append(res["total_seconds"] * 1000.0)
        forward_ms.append(res["forward_seconds"] * 1000.0)

    inference_peak_ram_mb = mx.metal.get_peak_memory() / (1024 * 1024)

    def compute_stats(arr):
        arr = sorted(arr)
        return {
            "mean": sum(arr) / len(arr),
            "p50": arr[len(arr) // 2],
            "p90": arr[int(len(arr) * 0.90)],
            "p99": arr[int(len(arr) * 0.99)],
            "min": arr[0],
            "max": arr[-1],
        }

    total_stats = compute_stats(latencies_ms)
    forward_stats = compute_stats(forward_ms)
    throughput_decisions_per_sec = 1000.0 / total_stats["p50"]

    results_data = {
        "hardware": {
            "host": platform.node(),
            "os": platform.system(),
            "release": platform.release(),
            "cpu": platform.processor() or "Apple Silicon",
            "gpu": dev_info.get("device_name", "Apple M4"),
            "architecture": dev_info.get("architecture", "UMA"),
            "total_ram_gb": dev_info.get("memory_size", 0) / (1024**3),
        },
        "model": args.model,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "model_resident_ram_mb": model_peak_ram_mb,
        "inference_peak_ram_mb": inference_peak_ram_mb,
        "total_latency_ms": total_stats,
        "forward_latency_ms": forward_stats,
        "throughput_p50_decisions_per_sec": throughput_decisions_per_sec,
        "last_decision_sample": res,
    }

    print("-" * 80)
    print(f"  Physical Benchmark Results on {dev_info.get('device_name', 'Apple M4')}:")
    print(f"  - End-to-End P50 Latency:  {total_stats['p50']:.3f} ms")
    print(f"  - End-to-End P99 Latency:  {total_stats['p99']:.3f} ms (Jitter: {total_stats['p99'] - total_stats['p50']:.2f} ms)")
    print(f"  - Model Forward P50:      {forward_stats['p50']:.3f} ms")
    print(f"  - Peak Memory Consumption: {inference_peak_ram_mb:.1f} MB (RAM)")
    print(f"  - Decision Throughput:     {throughput_decisions_per_sec:.1f} decisions / sec")
    print("=" * 80)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved benchmark evidence to {args.output}")


if __name__ == "__main__":
    main()
