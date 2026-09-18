"""Open-world decision boundary gating: independent sigmoid, free energy OOD detection, and abstention."""

from __future__ import annotations

import math


def sigmoid(z: float, temperature: float = 1.0) -> float:
    """Numerically stable sigmoid function."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be positive and finite")
    scaled = z / temperature
    if scaled >= 40.0:
        return 1.0
    elif scaled <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-scaled))


def independent_sigmoid_scores(
    logits: list[float],
    temperature: float = 1.0,
    threshold_shift: float = 0.0,
) -> list[float]:
    """Compute independent sigmoid probability for each option hypothesis.

    Args:
        logits: Raw option logits.
        temperature: Softening temperature.
        threshold_shift: Shift subtracted from logits before sigmoid (z_i - shift).
                         If shift=0, logit > 0 yields p > 0.5.
    """
    return [sigmoid(z - threshold_shift, temperature=temperature) for z in logits]


def compute_free_energy(logits: list[float], temperature: float = 1.0) -> float:
    """Compute Helmholtz Free Energy as an Out-of-Distribution (OOD) score:

        E(x; T) = -T * log( sum_i exp(z_i / T) )

    In-distribution samples with high positive evidence have LOW (negative) energy.
    Unfamiliar, missing-evidence, or off-topic samples have HIGH (less negative/positive) energy.
    """
    if not logits or any(not math.isfinite(z) for z in logits):
        raise ValueError("Logits must be nonempty and finite")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be positive and finite")

    scaled = [z / temperature for z in logits]
    max_val = max(scaled)
    # Stable Log-Sum-Exp
    lse = max_val + math.log(sum(math.exp(v - max_val) for v in scaled))
    return -temperature * lse


def gate_decision(
    probabilities: list[float],
    logits: list[float],
    option_ids: list[str],
    min_confidence: float = 0.5,
    energy_threshold: float | None = None,
    temperature: float = 1.0,
    abstain_option_id: str = "none_of_the_above",
) -> dict:
    """Comprehensive open-world decision gate combining confidence thresholding and energy OOD detection.

    Returns:
        dict with:
            disposition: 'automatic_decision' or 'abstain'
            selected_id: Best option ID, or abstain_option_id if rejected
            top_confidence: Argmax probability
            free_energy: Calculated Helmholtz free energy
            rejection_reasons: List of triggered rejection triggers (if any)
            qualifying_options: Options whose independent probability exceeds min_confidence (multi-label support)
    """
    if len(probabilities) != len(option_ids) or len(logits) != len(option_ids):
        raise ValueError("Probabilities, logits, and option_ids must have identical length")

    chosen_idx = max(range(len(probabilities)), key=probabilities.__getitem__)
    top_option_id = option_ids[chosen_idx]
    top_conf = probabilities[chosen_idx]
    energy = compute_free_energy(logits, temperature=temperature)

    rejection_reasons = []
    if top_conf < min_confidence:
        rejection_reasons.append("low_confidence")

    if energy_threshold is not None and energy > energy_threshold:
        rejection_reasons.append("high_free_energy_ood")

    # Multi-label detection
    qualifying = [option_ids[i] for i, p in enumerate(probabilities) if p >= min_confidence]

    if rejection_reasons:
        disposition = "abstain"
        final_choice = abstain_option_id
    else:
        disposition = "automatic_decision"
        final_choice = top_option_id

    return {
        "disposition": disposition,
        "selected_id": final_choice,
        "raw_argmax_id": top_option_id,
        "top_confidence": top_conf,
        "free_energy": energy,
        "rejection_reasons": rejection_reasons,
        "qualifying_options": qualifying,
        "is_multilabel": len(qualifying) > 1,
    }
