import math
import pytest
import sys
from pathlib import Path

# Add benchmarks directory to sys.path so evaluate can be imported
BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import evaluate


def test_compute_ece_empty():
    res = evaluate.compute_ece([])
    assert res["ece"] is None
    assert res["mce"] is None
    assert res["bins"] == []


def test_compute_ece_perfectly_calibrated():
    # 10 samples with 1.0 confidence and 100% correct
    rows = [{"confidence": 1.0, "correct": True} for _ in range(10)]
    res = evaluate.compute_ece(rows, n_bins=10)
    assert res["ece"] == pytest.approx(0.0)
    assert res["mce"] == pytest.approx(0.0)

    # 10 samples in [0.5, 0.6) with confidence 0.55, 50% accuracy
    rows_half = [{"confidence": 0.55, "correct": (i < 5)} for i in range(10)]
    res_half = evaluate.compute_ece(rows_half, n_bins=10)
    # accuracy = 0.5, mean_confidence = 0.55 -> gap = 0.05
    assert res_half["ece"] == pytest.approx(0.05)
    assert res_half["mce"] == pytest.approx(0.05)


def test_compute_ece_completely_overconfident():
    # 10 samples with 0.95 confidence but 0% accuracy
    rows = [{"confidence": 0.95, "correct": False} for _ in range(10)]
    res = evaluate.compute_ece(rows, n_bins=10)
    assert res["ece"] == pytest.approx(0.95)
    assert res["mce"] == pytest.approx(0.95)


def test_compute_ece_multi_bin_weighted():
    # 6 samples in bin 9 (conf 0.95, 5 correct -> acc 5/6=0.8333, gap = |5/6 - 0.95| = 0.116666...)
    # 4 samples in bin 4 (conf 0.45, 2 correct -> acc 2/4=0.5, gap = |0.5 - 0.45| = 0.05)
    rows = [
        {"confidence": 0.95, "correct": True},
        {"confidence": 0.95, "correct": True},
        {"confidence": 0.95, "correct": True},
        {"confidence": 0.95, "correct": True},
        {"confidence": 0.95, "correct": True},
        {"confidence": 0.95, "correct": False},
        {"confidence": 0.45, "correct": True},
        {"confidence": 0.45, "correct": True},
        {"confidence": 0.45, "correct": False},
        {"confidence": 0.45, "correct": False},
    ]
    res = evaluate.compute_ece(rows, n_bins=10)
    expected_ece = (6 / 10) * abs((5 / 6) - 0.95) + (4 / 10) * abs(0.5 - 0.45)
    assert res["ece"] == pytest.approx(expected_ece)
    assert res["mce"] == pytest.approx(abs((5 / 6) - 0.95))
    assert len(res["bins"]) == 2


def test_summarize_includes_ece_metrics():
    gold = [
        {
            "id": "item-1",
            "group_id": "g1",
            "family": "classification",
            "label": 0,
            "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
        },
        {
            "id": "item-2",
            "group_id": "g1",
            "family": "classification",
            "label": 1,
            "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
        },
    ]
    predictions = [
        {"id": "item-1", "probabilities": [0.9, 0.1], "option_ids": ["yes", "no"]},
        {"id": "item-2", "probabilities": [0.2, 0.8], "option_ids": ["yes", "no"]},
    ]
    eval_result = evaluate.evaluate(gold, predictions)
    family_res = eval_result["family_results"]["classification"]
    assert "ece" in family_res
    assert "ece_15" in family_res
    assert "ece_10" in family_res
    assert "mce" in family_res
    assert "calibration_report_15" in family_res
    # Both items are correct with confidence ~0.85 average:
    # item-1: correct=True, conf=0.9
    # item-2: correct=True, conf=0.8
    # Mean confidence = 0.85, accuracy = 1.0 -> gap = 0.15
    assert family_res["ece"] == pytest.approx(0.15)
