"""Apple Silicon MLX execution engine for SemIf decisions with zero-copy Unified Memory."""

from __future__ import annotations

import time
from typing import Any

from .core import LETTERS, apply_prior_calibration, digest, direct_messages, softmax

PROMPT_VERSION = "direct-options-v1"


def load_mlx_model(source: str, revision: str | None = None) -> tuple[Any, Any, dict]:
    """Load model and tokenizer using mlx_lm on Apple Silicon."""
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as exc:
        raise ImportError(
            "MLX is only supported on Apple Silicon macOS. Install with `pip install mlx mlx-lm`."
        ) from exc

    model, tokenizer = load(source)
    metadata = {
        "source": source,
        "revision": revision or "mlx-native",
        "engine": "mlx-apple-silicon",
        "dtype": "mlx-unified-memory",
    }
    return model, tokenizer, metadata


def score_mlx(
    model: Any,
    tokenizer: Any,
    row: dict,
    metadata: dict,
    temperature: float = 1.0,
    prior_logits: list[float] | None = None,
) -> dict:
    """Execute direct semantic decision readout using Apple Silicon MLX."""
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ImportError(
            "MLX is only supported on Apple Silicon macOS. Install with `pip install mlx mlx-lm`."
        ) from exc

    started = time.perf_counter()
    prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    tokens = tokenizer.encode(prompt)
    slots = []
    for letter in LETTERS[: len(row["options"])]:
        enc = tokenizer.encode(letter)
        if len(enc) != 1:
            enc_space = tokenizer.encode(f" {letter}")
            if len(enc_space) == 1:
                slots.append(enc_space[0])
                continue
        slots.append(enc[0])

    mx_tokens = mx.array([tokens])
    forward_start = time.perf_counter()

    # MLX forward
    out = model(mx_tokens)
    mx.eval(out)  # Force computation in unified memory
    forward_seconds = time.perf_counter() - forward_start

    # Readout slot logits
    logits_last = out[0, -1]
    selected = [float(logits_last[s].item()) for s in slots]

    if prior_logits is not None:
        calibrated_logits = apply_prior_calibration(selected, prior_logits[: len(selected)])
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
        "engine": "mlx",
        "input_tokens": len(tokens),
        "forward_seconds": forward_seconds,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": digest(prompt),
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "mlx-apple-silicon unified memory zero-copy slot logits",
        "probability_status": status,
    }
