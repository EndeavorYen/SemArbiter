"""Shape-bucketed CUDA Graphs execution engine for sub-10ms semantic decisions."""

from __future__ import annotations

from typing import Any
import time

DEFAULT_BUCKETS = (64, 128, 256, 512, 1024, 2048)


def find_bucket(seq_len: int, buckets: tuple[int, ...] = DEFAULT_BUCKETS) -> int | None:
    """Find the smallest bucket size >= seq_len. Returns None if exceeding all buckets."""
    for b in sorted(buckets):
        if b >= seq_len:
            return b
    return None


def pad_to_bucket(
    input_ids: list[int], bucket_len: int, pad_id: int = 0
) -> tuple[list[int], list[int], int]:
    """Pad input_ids with pad_id to bucket_len. Returns (padded_ids, attention_mask, real_length)."""
    real_len = len(input_ids)
    if real_len > bucket_len:
        raise ValueError(f"Sequence length {real_len} exceeds bucket size {bucket_len}")
    pad_count = bucket_len - real_len
    padded_ids = input_ids + [pad_id] * pad_count
    attention_mask = [1] * real_len + [0] * pad_count
    return padded_ids, attention_mask, real_len


class CUDAGraphBucket:
    """Captures and replays a static CUDA Graph for one specific sequence length bucket."""

    def __init__(self, model: Any, bucket_len: int, device: Any = "cuda"):
        import torch

        self.bucket_len = bucket_len
        self.device = torch.device(device)
        self.model = model
        self.graph = None

        self.static_input_ids = torch.zeros((1, bucket_len), dtype=torch.long, device=self.device)
        self.static_attention_mask = torch.ones((1, bucket_len), dtype=torch.long, device=self.device)
        self.static_last_hidden_state = None

    def capture(self, warmup_runs: int = 3) -> None:
        """Warm up the model and capture the CUDA Graph for this bucket length."""
        import torch

        if self.device.type != "cuda":
            return  # No-op on CPU / mock devices

        base_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None) or self.model
        stream = torch.cuda.Stream(device=self.device)

        # 1. Warm up in separate CUDA stream to initialize allocations & caches
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(warmup_runs):
                out = base_model(
                    input_ids=self.static_input_ids,
                    attention_mask=self.static_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None and isinstance(out, (tuple, list)):
                    hidden = out[0]
                elif hasattr(out, "logits"):
                    hidden = out.logits
        torch.cuda.synchronize(self.device)

        # Retain a clone for output shape buffer
        self.static_last_hidden_state = hidden.clone()

        # 2. Graph Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
            out = base_model(
                input_ids=self.static_input_ids,
                attention_mask=self.static_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None and isinstance(out, (tuple, list)):
                hidden = out[0]
            elif hasattr(out, "logits"):
                hidden = out.logits
            self.static_last_hidden_state.copy_(hidden)

        torch.cuda.synchronize(self.device)

    def replay(self, input_ids: list[int], attention_mask: list[int]) -> Any:
        """Copy input into static tensors and replay the pre-captured CUDA Graph."""
        import torch

        if self.graph is None or self.device.type != "cuda":
            # Dynamic execution fallback (e.g. CPU or uncaptured graph)
            base_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None) or self.model
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=self.device),
            }
            out = base_model(**inputs, use_cache=False, return_dict=True)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None and isinstance(out, (tuple, list)):
                hidden = out[0]
            elif hasattr(out, "logits"):
                hidden = out.logits
            return hidden

        # Copy data into fixed memory locations
        self.static_input_ids[0].copy_(torch.as_tensor(input_ids, dtype=torch.long, device=self.device))
        self.static_attention_mask[0].copy_(torch.as_tensor(attention_mask, dtype=torch.long, device=self.device))
        self.graph.replay()
        return self.static_last_hidden_state


class BucketGraphRunner:
    """Manages shape-bucketed CUDA Graphs for variable sequence-length inputs."""

    def __init__(
        self,
        model: Any,
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
        device: Any = "cuda",
        pad_id: int = 0,
        warmup_on_init: bool = False,
    ):
        self.model = model
        self.buckets = tuple(sorted(buckets))
        self.device = device
        self.pad_id = pad_id
        self._bucket_runners: dict[int, CUDAGraphBucket] = {}

        if warmup_on_init and getattr(device, "type", str(device)) == "cuda":
            self.warmup_all()

    def get_or_create_bucket(self, bucket_len: int) -> CUDAGraphBucket:
        if bucket_len not in self._bucket_runners:
            runner = CUDAGraphBucket(self.model, bucket_len, device=self.device)
            runner.capture()
            self._bucket_runners[bucket_len] = runner
        return self._bucket_runners[bucket_len]

    def warmup_all(self) -> None:
        """Pre-capture graphs for all configured buckets."""
        for b in self.buckets:
            self.get_or_create_bucket(b)

    def forward_hidden(self, input_ids: list[int]) -> tuple[Any, int, bool]:
        """Compute the last-real-token hidden state using CUDA Graphs if bucketed.

        Returns (last_token_hidden_state, bucket_used, was_graph_replayed).
        """
        import torch

        seq_len = len(input_ids)
        bucket_size = find_bucket(seq_len, self.buckets)

        if bucket_size is not None and getattr(self.device, "type", str(self.device)) == "cuda":
            padded_ids, attention_mask, real_len = pad_to_bucket(input_ids, bucket_size, self.pad_id)
            runner = self.get_or_create_bucket(bucket_size)
            hidden_all = runner.replay(padded_ids, attention_mask)
            # In causal attention with right-padding, hidden state at (real_len - 1)
            # is identical to unpadded sequence.
            last_token_hidden = hidden_all[:, real_len - 1, :]
            return last_token_hidden, bucket_size, True

        # Fallback: unpadded dynamic forward
        base_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None) or self.model
        device = next(self.model.parameters()).device
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, seq_len), dtype=torch.long, device=device),
        }
        out = base_model(**inputs, use_cache=False, return_dict=True)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None and isinstance(out, (tuple, list)):
            hidden = out[0]
        elif hasattr(out, "logits"):
            hidden = out.logits
        return hidden[:, -1, :], seq_len, False


def compile_model(model: Any, mode: str = "reduce-overhead") -> Any:
    """Optionally compile the model using torch.compile."""
    import torch

    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode=mode, fullgraph=False)
        except Exception:
            return model
    return model
