import math
import pytest

from semif_phase1.gating import (
    compute_free_energy,
    gate_decision,
    independent_sigmoid_scores,
    sigmoid,
)


def test_sigmoid_numerical_stability():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(100.0) == 1.0
    assert sigmoid(-100.0) == 0.0
    assert 0.7 < sigmoid(1.0) < 0.8

    with pytest.raises(ValueError, match="positive and finite"):
        sigmoid(1.0, temperature=-1.0)


def test_independent_sigmoid_scores():
    logits = [2.0, -2.0, 0.0]
    scores = independent_sigmoid_scores(logits)
    assert scores[0] > 0.8
    assert scores[1] < 0.2
    assert scores[2] == pytest.approx(0.5)

    # With positive threshold shift (demanding higher bar)
    shifted = independent_sigmoid_scores(logits, threshold_shift=2.0)
    assert shifted[0] == pytest.approx(0.5)  # 2.0 - 2.0 = 0 -> p = 0.5


def test_compute_free_energy_ood_contrast():
    # In-Distribution (ID): clear strong match (one high logit)
    id_logits = [20.0, 5.0, 2.0]
    energy_id = compute_free_energy(id_logits, temperature=1.0)
    # Energy should be approximately -max(logits) = -20
    assert energy_id < -19.0

    # Out-of-Distribution (OOD) / Missing evidence: diffuse, low logits
    ood_logits = [-2.0, -3.0, -2.5]
    energy_ood = compute_free_energy(ood_logits, temperature=1.0)
    # With diffuse low logits, energy is significantly higher (less negative / positive)
    assert energy_ood > 0.0

    # ID energy must be much lower than OOD energy
    assert energy_id < energy_ood


def test_gate_decision_automatic_acceptance():
    probs = [0.88, 0.12]
    logits = [15.0, 10.0]
    options = ["refund", "exchange"]

    gate = gate_decision(
        probabilities=probs,
        logits=logits,
        option_ids=options,
        min_confidence=0.7,
        energy_threshold=-10.0,
    )

    assert gate["disposition"] == "automatic_decision"
    assert gate["selected_id"] == "refund"
    assert gate["rejection_reasons"] == []
    assert gate["is_multilabel"] is False


def test_gate_decision_low_confidence_abstention():
    # Ambiguous choice: 51% vs 49%
    probs = [0.51, 0.49]
    logits = [12.0, 11.9]
    options = ["refund", "exchange"]

    gate = gate_decision(
        probabilities=probs,
        logits=logits,
        option_ids=options,
        min_confidence=0.8,  # Demands 80% confidence
    )

    assert gate["disposition"] == "abstain"
    assert gate["selected_id"] == "none_of_the_above"
    assert "low_confidence" in gate["rejection_reasons"]


def test_gate_decision_energy_ood_rejection():
    # Model assigns 85% to an option, but logits are overall very low (missing context / off-topic)
    probs = [0.85, 0.15]
    logits = [1.0, -1.0]  # Very low absolute logits
    options = ["technical_support", "billing"]

    # Energy is -log(e^1 + e^-1) = -log(2.718 + 0.368) = -1.127
    gate = gate_decision(
        probabilities=probs,
        logits=logits,
        option_ids=options,
        min_confidence=0.8,
        energy_threshold=-5.0,  # Demands energy <= -5.0 for familiar context
    )

    assert gate["disposition"] == "abstain"
    assert "high_free_energy_ood" in gate["rejection_reasons"]


def test_gate_decision_multilabel():
    # Both options have high independent support
    probs = [0.75, 0.65]
    logits = [12.0, 11.5]
    options = ["billing", "account_access"]

    gate = gate_decision(
        probabilities=probs,
        logits=logits,
        option_ids=options,
        min_confidence=0.6,
    )

    assert gate["disposition"] == "automatic_decision"
    assert gate["is_multilabel"] is True
    assert set(gate["qualifying_options"]) == {"billing", "account_access"}
