"""Tests for automated multi-model evaluation pipeline and matrix generator."""

import pytest

from benchmarks.evaluate_matrix import evaluate_predictions, generate_markdown_table

GOLD_DATA = [
    {
        "id": "item-1",
        "family": "customer_support",
        "group_id": "grp1",
        "options": [{"id": "yes"}, {"id": "no"}],
        "label": 0,
    },
    {
        "id": "item-2",
        "family": "customer_support",
        "group_id": "grp1",
        "options": [{"id": "yes"}, {"id": "no"}],
        "label": 1,
    },
]


def test_evaluate_predictions_perfect():
    preds = [
        {"id": "item-1", "probabilities": [0.95, 0.05], "forward_seconds": 0.012},
        {"id": "item-2", "probabilities": [0.05, 0.95], "forward_seconds": 0.014},
    ]
    stats = evaluate_predictions(GOLD_DATA, preds)
    assert stats["valid"] == 2
    assert stats["accuracy"] == 1.0
    assert stats["balanced_accuracy"] == 1.0
    assert stats["ece_15"] is not None
    assert stats["p50_forward_ms"] == pytest.approx(13.0, abs=1.0)


def test_evaluate_predictions_empty_or_missing():
    stats = evaluate_predictions(GOLD_DATA, [])
    assert stats["valid"] == 0
    assert stats["accuracy"] == 0.0
    assert stats["p50_forward_ms"] is None


def test_generate_markdown_table():
    sample_matrix = {
        "model-a": {
            "balanced_accuracy": 0.85,
            "accuracy": 0.84,
            "ece_15": 0.0512,
            "mean_brier": 0.21,
            "mean_confidence": 0.88,
            "p50_forward_ms": 4.5,
        }
    }
    table = generate_markdown_table(sample_matrix)
    assert "| Experiment / Model" in table
    assert "model-a" in table
    assert "0.850" in table
    assert "4.50" in table
