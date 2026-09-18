"""Tests for direct categorical decision readout and sliced LM head optimization."""

from types import SimpleNamespace
import pytest

from semif_phase1.direct import _slot_ids, encode_prompt, forward_restricted, score

ROW = {
    "id": "direct-test-1",
    "state": "The customer requested a refund after 45 days. Return policy is 30 days.",
    "question": "Can the refund be approved?",
    "options": [
        {"id": "yes", "description": "Approve refund"},
        {"id": "no", "description": "Deny refund"},
    ],
}


class MockTokenizer:
    def __init__(self):
        self.vocab = {"A": 65, "B": 66, "C": 67, "D": 68}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return f"CHAT:{messages[0]['content']}:END"

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        tokens = [ord(c) for c in text]
        return tokens

    def decode(self, tokens):
        if len(tokens) == 1 and tokens[0] in self.inv_vocab:
            return self.inv_vocab[tokens[0]]
        return "".join(chr(t) for t in tokens)


class MockLinear:
    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        import torch
        import torch.nn.functional as F
        return F.linear(x, self.weight, self.bias)


class MockBaseModel:
    def __init__(self, hidden_state):
        self.hidden_state = hidden_state

    def __call__(self, **kwargs):
        return SimpleNamespace(last_hidden_state=self.hidden_state)


class MockCausalLM:
    def __init__(self, hidden_dim, vocab_size, seed=42):
        torch = pytest.importorskip("torch")
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Fixed random hidden states: batch=1, seq=10, hidden_dim
        self.hidden_tensor = torch.randn(1, 10, hidden_dim)
        self.model = MockBaseModel(self.hidden_tensor)

        # LM head projection: vocab_size x hidden_dim
        weight = torch.randn(vocab_size, hidden_dim)
        bias = torch.randn(vocab_size)
        self.lm_head = MockLinear(weight, bias)

    def parameters(self):
        yield self.lm_head.weight

    def forward(self, **kwargs):
        import torch
        out = self.model(**kwargs)
        logits = self.lm_head(out.last_hidden_state)
        return SimpleNamespace(logits=logits)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


def test_slot_ids_valid():
    tok = MockTokenizer()
    slots = _slot_ids(tok, 2)
    assert slots == [65, 66]


def test_encode_prompt_produces_consistent_hash():
    tok = MockTokenizer()
    ids, slots, prompt_hash = encode_prompt(tok, ROW, max_tokens=4096)
    assert len(ids) > 0
    assert slots == [65, 66]
    assert len(prompt_hash) == 64


def test_encode_prompt_raises_on_overflow():
    tok = MockTokenizer()
    with pytest.raises(ValueError, match="exceed limit"):
        encode_prompt(tok, ROW, max_tokens=5)


def test_sliced_head_mathematical_equivalence():
    torch = pytest.importorskip("torch")
    causal_lm = MockCausalLM(hidden_dim=32, vocab_size=128)
    slots = [65, 66]  # 'A', 'B'

    dummy_inputs = {
        "input_ids": torch.randint(0, 100, (1, 10)),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }

    # Full forward
    full_vocab_logits = causal_lm.forward(**dummy_inputs).logits[:, -1, :]
    expected_slot_logits = full_vocab_logits[0, slots].float().tolist()

    # Sliced forward
    sliced_slot_logits_t, readout = forward_restricted(causal_lm, dummy_inputs, slots)
    sliced_slot_logits = sliced_slot_logits_t.float().tolist()

    assert readout == "native restricted lm_head projection to declared answer slots"
    for exp, act in zip(expected_slot_logits, sliced_slot_logits):
        assert act == pytest.approx(exp, abs=1e-5)


def test_sliced_head_score_matches_full_vocab_score():
    pytest.importorskip("torch")
    tok = MockTokenizer()
    causal_lm = MockCausalLM(hidden_dim=32, vocab_size=128)
    meta = {"source": "mock-qwen", "revision": "test"}

    res_sliced = score(causal_lm, tok, ROW, meta, sliced_head=True)
    res_full = score(causal_lm, tok, ROW, meta, sliced_head=False)

    assert res_sliced["sliced_head"] is True
    assert res_full["sliced_head"] is False

    assert "native restricted lm_head" in res_sliced["readout"]
    assert "native full-vocabulary" in res_full["readout"]

    # The probabilities and logits must match exactly!
    assert res_sliced["probabilities"] == pytest.approx(res_full["probabilities"], abs=1e-5)
    assert res_sliced["option_logits"] == pytest.approx(res_full["option_logits"], abs=1e-5)


def test_sliced_head_fallback_when_base_model_missing():
    pytest.importorskip("torch")
    tok = MockTokenizer()
    # Simple model without .model attribute
    class BareModel:
        def __init__(self):
            import torch
            self.p = torch.zeros(1)
        def parameters(self):
            yield self.p
        def forward(self, **kwargs):
            import torch
            # Mock return with logits of vocab 128
            logits = torch.ones(1, 1, 128)
            return SimpleNamespace(logits=logits)
        def __call__(self, **kwargs):
            return self.forward(**kwargs)

    bare_model = BareModel()
    meta = {"source": "bare-model", "revision": "test"}

    res = score(bare_model, tok, ROW, meta, sliced_head=True)
    assert res["readout"] == "native full-vocabulary last-position logits restricted to declared answer slots"
    assert len(res["probabilities"]) == 2
    assert res["probabilities"][0] == pytest.approx(0.5)
