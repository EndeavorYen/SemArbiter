import pytest

from semif_phase1.decider import (
    DEFAULT_DECIDER_TEMPERATURE,
    encode_decider_prompt,
    format_decider_prompt,
    score,
)


ROW = {
    "id": "test-decider-1",
    "state": "The user reported that order #12345 has not arrived after 10 days.",
    "question": "Which department should handle this ticket?",
    "options": [
        {"id": "logistics", "description": "Delivery and tracking inquiries."},
        {"id": "billing", "description": "Invoices and refund processing."},
    ],
}


class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")


class MockModel:
    def __init__(self, logits_dict):
        self.logits_dict = logits_dict

    def parameters(self):
        import torch
        class Param:
            device = torch.device("cpu")

        yield Param()

    def forward(self, **kwargs):
        import torch

        # Vocabulary size 256 for mock ascii bytes
        vocab = torch.zeros(256)
        for letter, logit_val in self.logits_dict.items():
            token_id = ord(letter)
            vocab[token_id] = logit_val

        class Output:
            pass

        out = Output()
        out.logits = vocab.unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
        return out

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


def test_format_decider_prompt_layout():
    prompt = format_decider_prompt(ROW)
    assert prompt.startswith("Context:\nThe user reported that order #12345")
    assert "Question: Which department should handle this ticket?" in prompt
    assert "Options:\n(A) Delivery and tracking inquiries.\n(B) Invoices and refund processing." in prompt
    assert prompt.endswith("Answer: (")


def test_decider_prompt_does_not_leak_forbidden_fields():
    leaky_row = dict(ROW, label="logistics", secret_notes="private database credentials")
    prompt = format_decider_prompt(leaky_row)
    assert "private database credentials" not in prompt
    assert "label" not in prompt


def test_structured_json_state_rendered_cleanly():
    struct_row = dict(ROW, state={"ticket_id": 99, "severity": "urgent"})
    prompt = format_decider_prompt(struct_row)
    assert '"ticket_id": 99' in prompt
    assert '"severity": "urgent"' in prompt


def test_encode_decider_prompt_checks_token_limits():
    tok = MockTokenizer()
    ids, slots, prompt_hash = encode_decider_prompt(tok, ROW, max_tokens=4096)
    assert len(ids) > 0
    assert slots == [ord("A"), ord("B")]
    assert len(prompt_hash) == 64

    # Test truncation error
    with pytest.raises(ValueError, match="exceed limit"):
        encode_decider_prompt(tok, ROW, max_tokens=10)


def test_decider_score_applies_temperature():
    torch = pytest.importorskip("torch")
    tok = MockTokenizer()
    # Let A have logit 10.0 and B have logit 5.0
    model = MockModel({"A": 10.0, "B": 5.0})
    metadata = {"source": "Mapika/decider-2b", "revision": "test"}

    # Run with T=1.0
    res_t1 = score(model, tok, ROW, metadata, temperature=1.0)
    p_t1 = res_t1["probabilities"]
    # With T=1, e^(10-10) / (e^0 + e^-5) = 1 / (1 + 0.0067) = 0.9933
    assert p_t1[0] > 0.99

    # Run with T=2.0
    res_t2 = score(model, tok, ROW, metadata, temperature=2.0)
    p_t2 = res_t2["probabilities"]
    # With T=2, logits are 5.0 and 2.5 -> diff is 2.5
    # e^0 / (e^0 + e^-2.5) = 1 / (1 + 0.082) = 0.924
    assert p_t2[0] < p_t1[0]  # Temperature smoothed the probabilities!
    assert sum(p_t2) == pytest.approx(1.0)
    assert res_t2["temperature"] == 2.0
    assert "calibrated decision distribution (T=2.0)" in res_t2["probability_status"]


def test_pure_temperature_scaling_logic():
    import math
    from semif_phase1.core import softmax

    raw_logits = [10.0, 5.0]
    p_t1 = softmax([val / 1.0 for val in raw_logits])
    p_t2 = softmax([val / 2.0 for val in raw_logits])
    assert p_t1[0] > 0.99
    assert p_t2[0] < p_t1[0]
    assert p_t2[0] == pytest.approx(math.exp(2.5) / (math.exp(2.5) + 1.0))
