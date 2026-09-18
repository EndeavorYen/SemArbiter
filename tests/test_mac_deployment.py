"""Tests for Apple Silicon Mac mini deployment and MLX/MPS resolution."""

import pytest

from benchmarks.benchmark_mac_mini import analyze_mac_mini_compatibility, detect_system_hardware
from semif_phase1.core import resolve_device
from semif_phase1.mlx_engine import score_mlx


def test_resolve_device_explicit_override():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
    assert resolve_device("cuda:0") == "cuda:0"


def test_resolve_device_defaults():
    import torch
    dev = resolve_device(None)
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        assert dev == "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        assert dev == "mps"
    else:
        assert dev == "cpu"


def test_mac_mini_compatibility_analysis():
    analysis = analyze_mac_mini_compatibility(16.0)
    assert "Qwen/Qwen3.5-4B" in analysis
    assert "Mapika/decider-2b" in analysis
    assert "Qwen/Qwen3-0.6B" in analysis

    # On 16GB RAM with 3.5GB OS reserved, 4B (8GB) and 2B (4GB) fit comfortably
    assert analysis["Qwen/Qwen3.5-4B"]["bf16_fit"] is True
    assert analysis["Qwen/Qwen3.5-4B"]["int4_fit"] is True
    assert analysis["Mapika/decider-2b"]["bf16_fit"] is True


def test_detect_system_hardware():
    hw = detect_system_hardware()
    assert "os" in hw
    assert "machine" in hw
    assert "is_apple_silicon" in hw


def test_mlx_raises_on_non_macos():
    # If MLX is not installed (e.g. on Windows/Linux x86), score_mlx should raise informative ImportError
    try:
        import mlx.core
        mlx_installed = True
    except ImportError:
        mlx_installed = False

    if not mlx_installed:
        with pytest.raises(ImportError, match="MLX is only supported on Apple Silicon"):
            score_mlx(None, None, {}, {})
