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
) -> dict:
    import torch

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
        vocabulary = _forward(model, inputs)[0].float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    selected = vocabulary[slots].cpu().tolist()
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
        "input_tokens": len(ids),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": status,
    }
