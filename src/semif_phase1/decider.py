"""Mapika/decider-2b decision model adapter and scorer.

Formats SemIf decision rows into decider's native prompt layout:
Context:
...
Question: ...
Options:
(A) ...
(B) ...
Answer: (

Reads the declared option logits at the last position and applies
temperature-scaled softmax.
"""

from __future__ import annotations

import inspect
import json
import time

from .core import LETTERS, digest, softmax, validate_row

PROMPT_VERSION = "decider-native-v1"
DEFAULT_DECIDER_TEMPERATURE = 1.05


def format_decider_prompt(row: dict) -> str:
    """Render a SemIf row into decider-2b prompt layout."""
    validate_row(row)
    state = row["state"]
    if isinstance(state, (dict, list)):
        state_text = json.dumps(state, ensure_ascii=False, indent=2)
    else:
        state_text = str(state).strip()

    options_lines = []
    for index, option in enumerate(row["options"]):
        letter = LETTERS[index]
        desc = option["description"].strip()
        options_lines.append(f"({letter}) {desc}")

    options_block = "\n".join(options_lines)
    question = row["question"].strip()

    return f"Context:\n{state_text}\n\nQuestion: {question}\nOptions:\n{options_block}\nAnswer: ("


def _slot_ids(tokenizer, count: int) -> list[int]:
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or (hasattr(tokenizer, "decode") and tokenizer.decode(encoded) != letter):
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


def encode_decider_prompt(tokenizer, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    """Encode one decider decision prompt and verify its single-token answer slots."""
    prompt = format_decider_prompt(row)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = _slot_ids(tokenizer, len(row["options"]))
    return ids, slots, digest(prompt)


def score(
    model,
    tokenizer,
    row: dict,
    metadata: dict,
    max_tokens: int = 4096,
    temperature: float = DEFAULT_DECIDER_TEMPERATURE,
) -> dict:
    """Score one row using Mapika/decider-2b layout and temperature-scaled readout."""
    import torch

    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    started = time.perf_counter()
    ids, slots, prompt_hash = encode_decider_prompt(tokenizer, row, max_tokens)
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
    scaled_logits = [val / temperature for val in selected]

    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(scaled_logits),
        "option_logits": selected,
        "temperature": temperature,
        "input_tokens": len(ids),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "decider-2b single-slot option logits with temperature scaling",
        "probability_status": f"calibrated decision distribution (T={temperature})",
    }
