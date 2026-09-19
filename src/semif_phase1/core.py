"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be a positive finite number")
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    weights = [math.exp(val - maximum) for val in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def expected_value(probabilities: list[float], option_values: list[float]) -> float:
    """Compute expected value over continuous values associated with categorical options.

    E[X] = sum_i (p_i * x_i)
    Useful for continuous steering, probability-weighted regression, risk scoring,
    and auction bid sizing without discrete argmax thrashing.
    """
    if len(probabilities) != len(option_values):
        raise ValueError("Probabilities and option_values must have identical length")
    return float(sum(p * v for p, v in zip(probabilities, option_values)))


def sanitize_state(state: Any) -> Any:
    """Recursively sanitize state data to replace NaN/Inf with string descriptors."""
    if isinstance(state, float):
        if math.isnan(state):
            return "NaN"
        if math.isinf(state):
            return "Infinity" if state > 0 else "-Infinity"
        return state
    elif isinstance(state, dict):
        return {k: sanitize_state(v) for k, v in state.items()}
    elif isinstance(state, list):
        return [sanitize_state(v) for v in state]
    return state


def apply_prior_calibration(logits: list[float], prior_logits: list[float]) -> list[float]:
    """Subtract unconditional context-free prior logits: z_calib = z_raw - z_null."""
    if len(logits) != len(prior_logits):
        raise ValueError("Logits and prior_logits must have identical length")
    return [z - z_null for z, z_null in zip(logits, prior_logits)]


def null_prompt_row(options_count: int = 2) -> dict:
    """Construct an information-free null state row for prior estimation."""
    return {
        "id": "null_prior_anchor",
        "state": "N/A",
        "question": "Which option follows?",
        "options": [
            {"id": f"opt_{LETTERS[i]}", "description": f"Option {LETTERS[i]}."}
            for i in range(options_count)
        ],
    }


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def resolve_device(requested_device: str | None = None) -> str:
    """Resolve the optimal execution device (CUDA, MPS for Apple Silicon, or CPU)."""
    import torch

    if requested_device is not None:
        return requested_device
    if torch.cuda.is_available():
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_causal_model(source: str, revision: str, device: str | None = None):
    """Load one pinned causal model on CUDA, Apple Silicon MPS, or CPU."""
    import torch
    import transformers

    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    
    target_device = resolve_device(device)
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()

    load_dtype = torch.bfloat16 if (target_device.startswith("cuda") or target_device == "mps") else torch.float32
    device_map = {"": target_device} if target_device != "cpu" else None

    model_kwargs = {
        "config": config,
        "dtype": load_dtype,
        "low_cpu_mem_usage": True,
        "output_loading_info": True,
        **common,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model, loading = cls.from_pretrained(source, **model_kwargs)
    if target_device == "cpu":
        model.to("cpu")
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": str(load_dtype).replace("torch.", ""),
        "device": target_device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata
