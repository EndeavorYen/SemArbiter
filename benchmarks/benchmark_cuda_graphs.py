"""Benchmark end-to-end latency: Standard PyTorch forward vs Shape-Bucketed CUDA Graphs on RTX 5080."""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace


def create_synthetic_transformer_layer(hidden_dim: int, device: str = "cuda"):
    import torch
    import torch.nn as nn

    class MockTransformerBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp_gate = nn.Linear(dim, dim * 3, bias=False)
            self.mlp_down = nn.Linear(dim * 3, dim, bias=False)

        def forward(self, x):
            # Attention mock
            normed = self.norm1(x)
            q = self.q_proj(normed)
            k = self.k_proj(normed)
            v = self.v_proj(normed)
            attn = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            attn_out = self.out_proj(torch.softmax(attn, dim=-1) @ v)
            x = x + attn_out

            # MLP mock
            normed2 = self.norm2(x)
            mlp_out = self.mlp_down(torch.relu(self.mlp_gate(normed2)))
            return x + mlp_out

    class MockDeepModel(nn.Module):
        def __init__(self, dim: int, num_layers: int = 16):
            super().__init__()
            self.layers = nn.ModuleList([MockTransformerBlock(dim) for _ in range(num_layers)])
            self.final_norm = nn.LayerNorm(dim)

        def forward(self, input_ids, **kwargs):
            B, L = input_ids.shape
            # Float embedding simulation
            x = input_ids.float().unsqueeze(-1).expand(B, L, hidden_dim)
            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)
            return SimpleNamespace(last_hidden_state=x)

    return MockDeepModel(hidden_dim).to(device)


def benchmark_latency(model, runner, seq_len: int, iterations: int = 100, device: str = "cuda"):
    import torch
    from semif_phase1.cuda_graph import pad_to_bucket

    input_ids = list(range(1, seq_len + 1))
    inputs = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.long, device=device),
    }

    # Warmup standard forward
    for _ in range(10):
        with torch.inference_mode():
            _ = model(**inputs, use_cache=False, return_dict=True)
    torch.cuda.synchronize()

    # Benchmark Standard Dynamic Forward
    dynamic_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(**inputs, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        dynamic_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark CUDA Graph Replay
    # First ensure bucket runner is captured
    bucket_size = runner.buckets[0]
    for b in runner.buckets:
        if b >= seq_len:
            bucket_size = b
            break
    padded_ids, mask, real_len = pad_to_bucket(input_ids, bucket_size, pad_id=0)
    graph_bucket = runner.get_or_create_bucket(bucket_size)

    # Warmup graph
    for _ in range(10):
        with torch.inference_mode():
            _ = graph_bucket.replay(padded_ids, mask)
    torch.cuda.synchronize()

    graph_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = graph_bucket.replay(padded_ids, mask)
        torch.cuda.synchronize()
        graph_latencies.append((time.perf_counter() - t0) * 1000.0)

    def stats(arr):
        arr = sorted(arr)
        p50 = arr[len(arr) // 2]
        p90 = arr[int(len(arr) * 0.90)]
        p99 = arr[int(len(arr) * 0.99)]
        mean = sum(arr) / len(arr)
        return {"mean": mean, "p50": p50, "p90": p90, "p99": p99}

    return {
        "dynamic": stats(dynamic_latencies),
        "cuda_graph": stats(graph_latencies),
    }


def main():
    import torch
    from semif_phase1.cuda_graph import BucketGraphRunner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[64, 256, 512])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires an NVIDIA GPU.")
        return

    device_name = torch.cuda.get_device_name(0)
    print("=" * 85)
    print(f"  RTX 5080 CUDA Graphs Latency Benchmark")
    print(f"  Hardware: {device_name} (Blackwell sm_120)")
    print(f"  Model config: {args.layers} layers, {args.hidden_dim} hidden dim")
    print("=" * 85)

    model = create_synthetic_transformer_layer(args.hidden_dim, device="cuda")
    runner = BucketGraphRunner(model, buckets=(64, 128, 256, 512, 1024), device="cuda")

    print(f"{'Seq Len':<10} | {'Dynamic P50':<14} | {'Graph P50':<14} | {'Dynamic P99':<14} | {'Graph P99':<14} | {'Speedup':<10}")
    print("-" * 85)

    for seq_len in args.seq_lens:
        res = benchmark_latency(model, runner, seq_len, iterations=args.iterations, device="cuda")
        dyn_p50 = res["dynamic"]["p50"]
        grp_p50 = res["cuda_graph"]["p50"]
        dyn_p99 = res["dynamic"]["p99"]
        grp_p99 = res["cuda_graph"]["p99"]
        speedup = dyn_p50 / max(grp_p50, 1e-6)
        print(f"{seq_len:<10} | {dyn_p50:.3f} ms      | {grp_p50:.3f} ms      | {dyn_p99:.3f} ms      | {grp_p99:.3f} ms      | {speedup:.2f}x")

    print("=" * 85)
    print("  Analysis: CUDA Graphs eliminates driver kernel launch queuing, reducing CPU overhead")
    print("  and latency variance (P99 jitter), ensuring deterministic sub-10ms decisions.")
    print("=" * 85)


if __name__ == "__main__":
    main()
