"""Direct categorical decision readout from native next-token logits."""

from __future__ import annotations

import inspect
import time

from .core import LETTERS, apply_prior_calibration, digest, direct_messages, softmax

PROMPT_VERSION = "direct-options-v1"


def _slot_ids(tokenizer, count: int) -> list[int]:
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    return result


def _forward(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :]


def forward_restricted(model, inputs, slots: list[int]):
    """Compute logits restricted strictly to slot token IDs, avoiding full-vocabulary projection.

    If model exposes a base transformer (e.g., model.model or model.transformer)
    and an lm_head Linear layer, extracts the final hidden state of the last token
    and projects exclusively against the weight rows corresponding to candidate slots.
    Otherwise, falls back cleanly to standard forward pass.
    """
    import torch
    import torch.nn.functional as F

    base_model = getattr(model, "model", None) or getattr(model, "transformer", None)
    lm_head = getattr(model, "lm_head", None)

    if base_model is not None and lm_head is not None and hasattr(lm_head, "weight"):
        try:
            base_out = base_model(**inputs, use_cache=False, return_dict=True)
            last_hidden = getattr(base_out, "last_hidden_state", None)
            if last_hidden is None and isinstance(base_out, (tuple, list)):
                last_hidden = base_out[0]
            if last_hidden is not None:
                rep = last_hidden[:, -1, :]
                slots_tensor = torch.as_tensor(slots, dtype=torch.long, device=rep.device)
                sliced_weight = lm_head.weight[slots_tensor]
                sliced_bias = None
                if getattr(lm_head, "bias", None) is not None:
                    sliced_bias = lm_head.bias[slots_tensor]
                slot_logits = F.linear(rep, sliced_weight, sliced_bias)
                return slot_logits[0], "native restricted lm_head projection to declared answer slots"
        except Exception:
            pass

    vocabulary = _forward(model, inputs)[0]
    return vocabulary[slots], "native full-vocabulary last-position logits restricted to declared answer slots"


def encode_prompt(tokenizer, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    """Encode one decision and verify its single-token answer slots."""
    prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = _slot_ids(tokenizer, len(row["options"]))
    for letter, token in zip(LETTERS, slots):
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    return ids, slots, digest(prompt)


def score(
    model,
    tokenizer,
    row: dict,
    metadata: dict,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    prior_logits: list[float] | None = None,
    sliced_head: bool = True,
    graph_runner: Any = None,
) -> dict:
    import torch
    import torch.nn.functional as F

    started = time.perf_counter()
    ids, slots, prompt_hash = encode_prompt(tokenizer, row, max_tokens)
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        if graph_runner is not None and getattr(device, "type", str(device)) == "cuda":
            last_hidden, bucket_used, was_replayed = graph_runner.forward_hidden(ids)
            slots_tensor = torch.as_tensor(slots, dtype=torch.long, device=last_hidden.device)
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight"):
                sliced_weight = lm_head.weight[slots_tensor]
                sliced_bias = lm_head.bias[slots_tensor] if getattr(lm_head, "bias", None) is not None else None
                slot_logits = F.linear(last_hidden, sliced_weight, sliced_bias)
                selected = slot_logits[0].float().cpu().tolist()
                readout = f"cuda-graph-bucket-{bucket_used} restricted lm_head projection" if was_replayed else "native restricted lm_head projection to declared answer slots"
            else:
                vocabulary = _forward(model, inputs)[0].float()
                selected = vocabulary[slots].cpu().tolist()
                readout = "native full-vocabulary last-position logits restricted to declared answer slots"
        elif sliced_head:
            slot_tensor, readout = forward_restricted(model, inputs, slots)
            selected = slot_tensor.float().cpu().tolist()
        else:
            vocabulary = _forward(model, inputs)[0].float()
            selected = vocabulary[slots].cpu().tolist()
            readout = "native full-vocabulary last-position logits restricted to declared answer slots"
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if prior_logits is not None:
        calibrated_logits = apply_prior_calibration(selected, prior_logits[:len(selected)])
    else:
        calibrated_logits = selected

    probs = softmax(calibrated_logits, temperature=temperature)
    status = (
        "conditional option score; uncalibrated as decision confidence"
        if (temperature == 1.0 and prior_logits is None)
        else f"calibrated decision distribution (T={temperature}, prior_debiased={prior_logits is not None})"
    )
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": probs,
        "option_logits": selected,
        "calibrated_logits": calibrated_logits,
        "temperature": temperature,
        "prior_debiased": prior_logits is not None,
        "sliced_head": sliced_head,
        "cuda_graph": graph_runner is not None,
        "input_tokens": len(ids),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": readout,
        "probability_status": status,
    }
