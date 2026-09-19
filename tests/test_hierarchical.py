"""Unit tests for Hierarchical Decision Trees and continuous expected value mapping."""

from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock

from semif_phase1.core import expected_value, sanitize_state
from semif_phase1.hierarchical import (
    DecisionNode,
    DecisionOption,
    HierarchicalDecisionTree,
    HierarchicalResult,
)


def test_expected_value():
    probs = [0.1, 0.8, 0.1]
    values = [-1.0, 0.0, 1.0]
    ev = expected_value(probs, values)
    assert math.isclose(ev, 0.0, abs_tol=1e-5)

    probs2 = [0.0, 0.2, 0.8]
    ev2 = expected_value(probs2, values)
    assert math.isclose(ev2, 0.8, abs_tol=1e-5)

    with pytest.raises(ValueError):
        expected_value([0.5, 0.5], [1.0])


def test_sanitize_state():
    bad_state = {
        "speed": float("nan"),
        "ratio": float("inf"),
        "nested": {"loss": float("-inf"), "count": 42},
        "items": [1.0, float("nan"), "valid"],
    }
    cleaned = sanitize_state(bad_state)
    assert cleaned["speed"] == "NaN"
    assert cleaned["ratio"] == "Infinity"
    assert cleaned["nested"]["loss"] == "-Infinity"
    assert cleaned["nested"]["count"] == 42
    assert cleaned["items"] == [1.0, "NaN", "valid"]


def test_hierarchical_tree_traversal():
    # Construct a 2-tier tree:
    # Root: Combat Stance (Aggressive vs Defensive)
    # Child 1: Aggressive -> (Melee Slash vs Fireball)
    # Child 2: Defensive -> (Shield Block vs Roll Dodge)

    node_aggressive = DecisionNode(
        id="aggressive_sub",
        question="Which attack action to execute?",
        options=[
            DecisionOption("melee_slash", "Execute heavy sword strike."),
            DecisionOption("fireball", "Cast fire magic spell."),
        ],
        energy_threshold=-10.0,
    )

    node_defensive = DecisionNode(
        id="defensive_sub",
        question="Which defensive reaction to take?",
        options=[
            DecisionOption("shield_block", "Raise shield to absorb blow."),
            DecisionOption("roll_dodge", "Evade sideways."),
        ],
        energy_threshold=-10.0,
    )

    root_node = DecisionNode(
        id="root_stance",
        question="What is the general combat tactical posture?",
        options=[
            DecisionOption("attack", "Take offensive stance.", child_node=node_aggressive),
            DecisionOption("defend", "Take defensive stance.", child_node=node_defensive),
        ],
        energy_threshold=-10.0,
    )

    tree = HierarchicalDecisionTree(root_node)

    # Mock model and tokenizer
    mock_model = MagicMock()
    param_mock = MagicMock()
    param_mock.device.type = "cpu"
    mock_model.parameters.return_value = [param_mock]

    # Mock score return values
    def mock_score_side_effect(model, tokenizer, row, metadata, **kwargs):
        if "root_stance" in row["id"]:
            # Pick attack
            return {
                "id": row["id"],
                "probabilities": [0.85, 0.15],
                "calibrated_logits": [25.0, 10.0],
            }
        else:
            # Pick fireball
            return {
                "id": row["id"],
                "probabilities": [0.2, 0.8],
                "calibrated_logits": [12.0, 28.0],
            }

    import semif_phase1.hierarchical as h_mod
    original_score = h_mod.score
    try:
        h_mod.score = mock_score_side_effect
        state = {"enemy_health": 40, "distance_m": 8.0}
        res = tree.evaluate(mock_model, MagicMock(), state)

        assert res.depth_reached == 2
        assert res.path == ["attack", "fireball"]
        assert res.final_action == "fireball"
        assert res.is_ood is False
        assert "root_stance" in res.node_decisions
        assert "aggressive_sub" in res.node_decisions
    finally:
        h_mod.score = original_score
