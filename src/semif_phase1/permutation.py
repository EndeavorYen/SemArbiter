"""Permutation ensembling over option orderings to eliminate position bias."""

from __future__ import annotations

import time

from .core import softmax, validate_row
from .direct import PROMPT_VERSION, score as direct_score


def generate_permutations(count: int, max_perms: int = 2) -> list[list[int]]:
    """Generate option order permutations.

    - For binary (count=2): returns [[0, 1], [1, 0]] (full reversal).
    - For multiclass (count>2): returns cyclic rotations to ensure balanced slot positions.
    """
    if count < 2:
        return [[0]]
    if count == 2:
        return [[0, 1], [1, 0]][:max_perms]

    # Cyclic permutations
    num_perms = min(count, max_perms)
    perms = []
    for shift in range(num_perms):
        perms.append([(i + shift) % count for i in range(count)])
    return perms


def permute_row(row: dict, perm: list[int]) -> dict:
    """Create a row variant with options ordered by perm."""
    validate_row(row)
    options = row["options"]
    if len(perm) != len(options):
        raise ValueError("Permutation length must match options count")
    return {
        **row,
        "options": [options[idx] for idx in perm],
    }


def align_logits(permuted_logits: list[float], perm: list[int]) -> list[float]:
    """Map slot logits back to original canonical option indices.

    If perm[slot_idx] = original_idx, then aligned[original_idx] = permuted_logits[slot_idx].
    """
    if len(permuted_logits) != len(perm):
        raise ValueError("Logits and permutation length must match")
    aligned = [0.0] * len(perm)
    for slot_idx, orig_idx in enumerate(perm):
        aligned[orig_idx] = permuted_logits[slot_idx]
    return aligned


def aggregate_logits(aligned_runs: list[list[float]]) -> list[float]:
    """Compute arithmetic mean of aligned logits across permutation runs."""
    if not aligned_runs:
        raise ValueError("Need at least one run to aggregate")
    n_runs = len(aligned_runs)
    n_opts = len(aligned_runs[0])
    mean_logits = [0.0] * n_opts
    for run in aligned_runs:
        for idx in range(n_opts):
            mean_logits[idx] += run[idx] / n_runs
    return mean_logits


def score_permuted(
    model,
    tokenizer,
    row: dict,
    metadata: dict,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    prior_logits: list[float] | None = None,
    max_perms: int = 2,
    sliced_head: bool = True,
) -> dict:
    """Evaluate row over multiple option permutations and ensemble the results."""
    validate_row(row)
    started = time.perf_counter()
    options_count = len(row["options"])
    perms = generate_permutations(options_count, max_perms=max_perms)

    aligned_runs = []
    runs_details = []
    total_forward_seconds = 0.0

    for perm in perms:
        variant_row = permute_row(row, perm)
        # Note: prior_logits is slot-based, so it applies to the slots of variant_row
        run_res = direct_score(
            model,
            tokenizer,
            variant_row,
            metadata,
            max_tokens=max_tokens,
            temperature=temperature,
            prior_logits=prior_logits,
            sliced_head=sliced_head,
        )
        total_forward_seconds += run_res["forward_seconds"]
        # run_res["option_logits"] is in variant_row slot order
        aligned = align_logits(run_res["option_logits"], perm)
        aligned_runs.append(aligned)
        runs_details.append({
            "perm": perm,
            "slot_logits": run_res["option_logits"],
            "aligned_logits": aligned,
            "slot_probabilities": run_res["probabilities"],
        })

    # Aggregate aligned logits
    mean_aligned_logits = aggregate_logits(aligned_runs)
    final_probs = softmax(mean_aligned_logits, temperature=temperature)

    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": final_probs,
        "option_logits": mean_aligned_logits,
        "temperature": temperature,
        "permutations_evaluated": len(perms),
        "forward_seconds": total_forward_seconds,
        "total_seconds": time.perf_counter() - started,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": f"permutation-ensembled ({len(perms)} perms) native option logits",
        "probability_status": f"calibrated ensembled distribution (perms={len(perms)}, T={temperature})",
        "ensemble_runs": runs_details,
    }
