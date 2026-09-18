"""Tests for shape-bucketed CUDA Graphs execution engine."""

from types import SimpleNamespace
import pytest

from semif_phase1.cuda_graph import BucketGraphRunner, CUDAGraphBucket, find_bucket, pad_to_bucket


def test_find_bucket_boundaries():
    buckets = (64, 128, 256)
    assert find_bucket(10, buckets) == 64
    assert find_bucket(64, buckets) == 64
    assert find_bucket(65, buckets) == 128
    assert find_bucket(256, buckets) == 256
    assert find_bucket(257, buckets) is None


def test_pad_to_bucket():
    ids = [101, 102, 103]
    padded_ids, mask, real_len = pad_to_bucket(ids, bucket_len=6, pad_id=0)
    assert padded_ids == [101, 102, 103, 0, 0, 0]
    assert mask == [1, 1, 1, 0, 0, 0]
    assert real_len == 3


def test_pad_to_bucket_overflow_raises():
    with pytest.raises(ValueError, match="exceeds bucket size"):
        pad_to_bucket([1, 2, 3, 4], bucket_len=3)


class MockModelForGraph:
    def __init__(self, hidden_dim=16):
        import torch
        self.hidden_dim = hidden_dim
        self.p = torch.zeros(1)

    def parameters(self):
        yield self.p

    def forward(self, input_ids, **kwargs):
        import torch
        B, L = input_ids.shape
        # Return synthetic hidden state [B, L, hidden_dim]
        hidden = torch.ones((B, L, self.hidden_dim), dtype=torch.float32) * 2.5
        return SimpleNamespace(last_hidden_state=hidden)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


def test_bucket_graph_runner_cpu_fallback():
    torch = pytest.importorskip("torch")
    model = MockModelForGraph(hidden_dim=16)
    runner = BucketGraphRunner(model, buckets=(32, 64), device="cpu")

    input_ids = [1, 2, 3, 4, 5]
    hidden, bucket, was_replayed = runner.forward_hidden(input_ids)

    # On CPU, was_replayed is False (dynamic forward)
    assert was_replayed is False
    assert hidden.shape == (1, 16)
    assert hidden[0, 0].item() == pytest.approx(2.5)


def test_cuda_graph_capture_and_replay_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for hardware CUDA graph capture test")

    class SimpleGPUModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)

        def forward(self, input_ids, **kwargs):
            B, L = input_ids.shape
            x = input_ids.float().unsqueeze(-1).expand(B, L, 32)
            out = self.linear(x)
            return SimpleNamespace(last_hidden_state=out)

    device = torch.device("cuda")
    model = SimpleGPUModule().to(device)

    runner = BucketGraphRunner(model, buckets=(16, 32), device=device)
    input_ids = [1, 2, 3, 4, 5, 6, 7]

    hidden, bucket, was_replayed = runner.forward_hidden(input_ids)
    assert was_replayed is True
    assert bucket == 16
    assert hidden.shape == (1, 32)
    assert hidden.device.type == "cuda"
