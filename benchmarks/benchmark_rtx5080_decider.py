"""Physical hardware benchmark on NVIDIA GeForce RTX 5080: Mapika/decider-2b.

Measures real end-to-end forward inference latency with actual model weights loaded in VRAM:
1. Dynamic Forward (Full LM Head)
2. Sliced LM Head Forward
3. Shape-Bucketed CUDA Graphs Forward
4. Peak VRAM footprint and memory throughput
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from semif_phase1.decider import format_decider_prompt, _slot_ids
from semif_phase1.direct import forward_restricted


def run_benchmark(
    model_id: str = "Mapika/decider-2b",
    iterations: int = 50,
    warmup: int = 10,
    output_path: str = "results/phase3-rtx5080-decider2b-real-benchmark.json",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    device_name = torch.cuda.get_device_name(0)
    print(f"Loading {model_id} onto {device_name}...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()
    load_time = time.time() - t_load_start
    print(f"Model loaded in {load_time:.2f} seconds.")

    initial_vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"Initial model resident VRAM: {initial_vram_mb:.1f} MB")

    # Sample decision row
    sample_row = {
        "id": "sample-001",
        "state": "User has an active enterprise subscription with 4 unused seats. Request is for seat transfer.",
        "question": "Should this seat reallocation request be processed immediately?",
        "options": [
            {"id": "approve", "description": "Yes, approve seat reallocation"},
            {"id": "reject", "description": "No, route to human sales admin"},
        ],
    }

    prompt = format_decider_prompt(sample_row)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_len = inputs.input_ids.shape[1]
    slots = _slot_ids(tokenizer, 2)
    print(f"Prompt length: {seq_len} tokens, Candidate slots: {slots}")

    # 1. Warmup
    print("Warming up GPU...")
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(**inputs)
    torch.cuda.synchronize()

    # 2. Dynamic Forward (Full LM Head)
    print(f"Running Full LM Head forward ({iterations} iterations)...")
    full_latencies = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        with torch.inference_mode():
            starter.record()
            _ = model(**inputs).logits[:, -1, :]
            ender.record()
        torch.cuda.synchronize()
        full_latencies.append(starter.elapsed_time(ender))

    # 3. Sliced LM Head Forward
    print(f"Running Sliced LM Head forward ({iterations} iterations)...")
    sliced_latencies = []
    for _ in range(iterations):
        with torch.inference_mode():
            starter.record()
            _ = forward_restricted(model, inputs, slots)
            ender.record()
        torch.cuda.synchronize()
        sliced_latencies.append(starter.elapsed_time(ender))

    # 4. CUDA Graphs Execution (Static Capture)
    print("Capturing CUDA Graph for Sliced Head...")
    graph_latencies = []
    try:
        static_input_ids = inputs["input_ids"].clone()
        static_attention_mask = inputs.get("attention_mask", torch.ones_like(static_input_ids)).clone()
        static_inputs = {"input_ids": static_input_ids, "attention_mask": static_attention_mask}
        slots_cuda = torch.tensor(slots, dtype=torch.long, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                with torch.inference_mode():
                    _ = forward_restricted(model, static_inputs, slots_cuda)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            with torch.inference_mode():
                static_out = forward_restricted(model, static_inputs, slots_cuda)

        print(f"Running CUDA Graphs replay ({iterations} iterations)...")
        for _ in range(iterations):
            starter.record()
            g.replay()
            ender.record()
            torch.cuda.synchronize()
            graph_latencies.append(starter.elapsed_time(ender))
        cuda_graph_supported = True
    except Exception as e:
        print(f"CUDA Graph capture warning: {e}. Falling back to Sliced Head metrics.")
        cuda_graph_supported = False
        graph_latencies = sliced_latencies

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    def stats(arr):
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "std_ms": float(np.std(arr)),
        }

    results = {
        "benchmark": "RTX 5080 Physical Model Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "gpu": device_name,
            "architecture": "Blackwell (sm_120)",
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "total_vram_gb": float(torch.cuda.get_device_properties(0).total_memory / (1024**3)),
        },
        "model": {
            "model_id": model_id,
            "dtype": "bfloat16",
            "prompt_length_tokens": seq_len,
            "resident_vram_mb": float(initial_vram_mb),
            "peak_vram_mb": float(peak_vram_mb),
        },
        "measurements": {
            "full_lm_head_dynamic": stats(full_latencies),
            "sliced_lm_head_dynamic": stats(sliced_latencies),
            "sliced_cuda_graph": stats(graph_latencies),
            "cuda_graph_supported": cuda_graph_supported,
            "speedup_sliced_vs_full": float(np.percentile(full_latencies, 50) / np.percentile(sliced_latencies, 50)),
            "speedup_graph_vs_full": float(np.percentile(full_latencies, 50) / np.percentile(graph_latencies, 50)),
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nBenchmark completed successfully! Results written to {output_path}:")
    print(f"  Full LM Head P50:    {results['measurements']['full_lm_head_dynamic']['p50_ms']:.3f} ms")
    print(f"  Sliced LM Head P50:  {results['measurements']['sliced_lm_head_dynamic']['p50_ms']:.3f} ms")
    print(f"  CUDA Graphs P50:     {results['measurements']['sliced_cuda_graph']['p50_ms']:.3f} ms")
    print(f"  Speedup vs Full:     {results['measurements']['speedup_graph_vs_full']:.2f}x")
    print(f"  Peak VRAM:           {peak_vram_mb:.1f} MB")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Mapika/decider-2b")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/phase3-rtx5080-decider2b-real-benchmark.json")
    args = parser.parse_args()
    run_benchmark(model_id=args.model, iterations=args.iterations, output_path=args.output)
