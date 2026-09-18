"""Benchmark and theoretical profiling of Sliced LM Head projection vs Full Vocabulary projection."""

from __future__ import annotations

import argparse
import json
import time

MODELS = {
    "Qwen/Qwen3.5-4B": {"hidden_dim": 2560, "vocab_size": 151936, "default_dtype_bytes": 2},
    "Qwen/Qwen3-0.6B": {"hidden_dim": 1024, "vocab_size": 151936, "default_dtype_bytes": 2},
    "openbmb/MiniCPM5-2B": {"hidden_dim": 2304, "vocab_size": 122752, "default_dtype_bytes": 2},
    "meta-llama/Meta-Llama-3-8B": {"hidden_dim": 4096, "vocab_size": 128256, "default_dtype_bytes": 2},
}


def compute_projection_savings(hidden_dim: int, vocab_size: int, num_slots: int, dtype_bytes: int = 2):
    """Compute exact theoretical compute and memory savings."""
    full_flops = 2 * hidden_dim * vocab_size
    sliced_flops = 2 * hidden_dim * num_slots
    flop_reduction_pct = 100.0 * (1.0 - sliced_flops / full_flops)

    full_weight_bytes = hidden_dim * vocab_size * dtype_bytes
    sliced_weight_bytes = hidden_dim * num_slots * dtype_bytes
    weight_reduction_pct = 100.0 * (1.0 - sliced_weight_bytes / full_weight_bytes)

    full_output_bytes = vocab_size * dtype_bytes
    sliced_output_bytes = num_slots * dtype_bytes

    return {
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "num_slots": num_slots,
        "full_flops": full_flops,
        "sliced_flops": sliced_flops,
        "flop_reduction_pct": flop_reduction_pct,
        "full_weight_bytes": full_weight_bytes,
        "sliced_weight_bytes": sliced_weight_bytes,
        "weight_reduction_pct": weight_reduction_pct,
        "bandwidth_saved_mb": (full_weight_bytes - sliced_weight_bytes) / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-slots", type=int, default=4, help="Number of candidate decision slots (default: 4)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    results = {}
    for model_name, cfg in MODELS.items():
        results[model_name] = compute_projection_savings(
            cfg["hidden_dim"], cfg["vocab_size"], args.num_slots, cfg["default_dtype_bytes"]
        )

    if args.format == "json":
        print(json.dumps(results, indent=2))
        return

    print("=" * 85)
    print(f"  Theoretical Sliced LM Head Analysis (Candidate Slots K = {args.num_slots})")
    print("=" * 85)
    print(f"{'Model':<28} | {'Full Weights':<12} | {'Sliced Weights':<14} | {'Saved Bandwidth':<15} | {'FLOPs Cut':<10}")
    print("-" * 85)
    for model_name, stats in results.items():
        full_mb = f"{stats['full_weight_bytes'] / (1024*1024):.1f} MB"
        sliced_kb = f"{stats['sliced_weight_bytes'] / 1024:.2f} KB"
        saved_mb = f"{stats['bandwidth_saved_mb']:.1f} MB"
        cut_pct = f"{stats['flop_reduction_pct']:.3f}%"
        print(f"{model_name:<28} | {full_mb:<12} | {sliced_kb:<14} | {saved_mb:<15} | {cut_pct:<10}")
    print("=" * 85)
    print("  Conclusion: Slicing the LM Head reduces projection compute & bandwidth by > 99.99%.")
    print("  On modern GPUs (e.g. RTX 5080 with ~1 TB/s memory bandwidth), reading 778 MB weights")
    print("  takes ~0.7 ms. Slicing drops weight retrieval to 20 KB (L1/L2 cache-resident),")
    print("  eliminating an entire DRAM memory transfer per forward pass.")
    print("=" * 85)


if __name__ == "__main__":
    main()
