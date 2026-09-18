"""Mac mini (16GB Unified Memory) deployment analysis and MLX/MPS benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys

MODELS_CONFIG = {
    "Qwen/Qwen3.5-4B": {
        "params_billion": 4.0,
        "bf16_gb": 8.0,
        "int8_gb": 4.2,
        "int4_gb": 2.3,
        "recommended_mac_tier": "16GB Mac mini (M2 / M4)",
    },
    "Mapika/decider-2b": {
        "params_billion": 2.0,
        "bf16_gb": 4.0,
        "int8_gb": 2.1,
        "int4_gb": 1.2,
        "recommended_mac_tier": "8GB / 16GB Mac mini",
    },
    "Qwen/Qwen3-0.6B": {
        "params_billion": 0.6,
        "bf16_gb": 1.2,
        "int8_gb": 0.65,
        "int4_gb": 0.35,
        "recommended_mac_tier": "All Mac devices (Ultra-lightweight)",
    },
}


def analyze_mac_mini_compatibility(system_ram_gb: float = 16.0):
    """Analyze memory margins on 16GB Unified Memory Mac mini."""
    os_reserved_gb = 3.5  # Typical macOS Sonoma / Sequoia base footprint
    available_gb = system_ram_gb - os_reserved_gb

    analysis = {}
    for model_name, cfg in MODELS_CONFIG.items():
        analysis[model_name] = {
            "bf16_fit": cfg["bf16_gb"] < available_gb,
            "int8_fit": cfg["int8_gb"] < available_gb,
            "int4_fit": cfg["int4_gb"] < available_gb,
            "bf16_remaining_ram": available_gb - cfg["bf16_gb"],
            "int4_remaining_ram": available_gb - cfg["int4_gb"],
            "recommended": cfg["recommended_mac_tier"],
        }
    return analysis


def detect_system_hardware():
    info = {
        "os": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch_version"] = None
        info["mps_available"] = False
        info["cuda_available"] = False

    try:
        import mlx.core as mx
        info["mlx_available"] = True
    except ImportError:
        info["mlx_available"] = False

    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-ram", type=float, default=16.0, help="Target system RAM in GB (default: 16.0)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    hw_info = detect_system_hardware()
    compat = analyze_mac_mini_compatibility(args.system_ram)

    if args.format == "json":
        print(json.dumps({"hardware": hw_info, "compatibility": compat}, indent=2))
        return

    print("=" * 85)
    print(f"  Apple Silicon Mac mini ({args.system_ram:.0f}GB Unified Memory) Deployment Report")
    print(f"  Current Host: {hw_info['os']} ({hw_info['machine']}) | Apple Silicon: {hw_info['is_apple_silicon']}")
    print(f"  Frameworks: PyTorch MPS = {hw_info['mps_available']} | Apple MLX = {hw_info['mlx_available']}")
    print("=" * 85)
    print(f"{'Model':<22} | {'BF16 Footprint':<14} | {'4-Bit MLX':<10} | {'16GB RAM Status':<18} | {'Best Format':<10}")
    print("-" * 85)

    for model_name, stats in compat.items():
        cfg = MODELS_CONFIG[model_name]
        bf16_str = f"{cfg['bf16_gb']:.1f} GB"
        int4_str = f"{cfg['int4_gb']:.1f} GB"
        if stats["bf16_fit"]:
            status_str = f"OK ({stats['bf16_remaining_ram']:.1f}GB free)"
        else:
            status_str = "Tight (Recommend 4-bit)"
        best_fmt = "4-bit / BF16" if cfg["params_billion"] <= 2.0 else "4-bit (2.3GB)"
        print(f"{model_name:<22} | {bf16_str:<14} | {int4_str:<10} | {status_str:<18} | {best_fmt:<10}")

    print("=" * 85)
    print("  Deployment Paths on Mac mini (16GB):")
    print("  1. Native MLX (mlx-lm): Zero-copy Unified Memory, ~120 GB/s bandwidth on M2/M4.")
    print("     Run: pip install mlx mlx-lm && python -m semif_phase1.mlx_engine")
    print("  2. PyTorch MPS: Native Metal Performance Shaders backend with torch.bfloat16.")
    print("  3. Browser WebGPU (Static): Works out-of-the-box in Safari/Chrome via webgpu-demo/.")
    print("=" * 85)


if __name__ == "__main__":
    main()
