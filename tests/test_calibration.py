import math
import pytest
import sys
from pathlib import Path

from semif_phase1.core import apply_prior_calibration, null_prompt_row, softmax, validate_row

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from calibrate_temperature import compute_nll_at_temp, fit_temperature, golden_section_search


def test_softmax_temperature_behavior():
    raw = [4.0, 2.0]
    # At T=1: diff is 2 -> e^2 / (e^2 + 1) = 0.8808
    p_t1 = softmax(raw, temperature=1.0)
    # At T=2: diff is 1 -> e^1 / (e^1 + 1) = 0.7310
    p_t2 = softmax(raw, temperature=2.0)
    # At T=0.5: diff is 4 -> e^4 / (e^4 + 1) = 0.9820
    p_t05 = softmax(raw, temperature=0.5)

    assert p_t05[0] > p_t1[0] > p_t2[0]
    assert sum(p_t2) == pytest.approx(1.0)
    assert sum(p_t05) == pytest.approx(1.0)

    # Invalid temperatures
    with pytest.raises(ValueError, match="positive finite"):
        softmax(raw, temperature=0.0)
    with pytest.raises(ValueError, match="positive finite"):
        softmax(raw, temperature=-1.5)


def test_apply_prior_calibration():
    # Model has a natural bias towards Option A (logit +3.0 higher on null prompt)
    raw_logits = [8.0, 5.0]
    null_prior = [3.0, 0.0]

    debiased = apply_prior_calibration(raw_logits, null_prior)
    assert debiased == [5.0, 5.0]

    # After debiasing, the two options have equal probability
    probs = softmax(debiased)
    assert probs[0] == pytest.approx(0.5)
    assert probs[1] == pytest.approx(0.5)

    # Length mismatch
    with pytest.raises(ValueError, match="identical length"):
        apply_prior_calibration([1.0, 2.0], [1.0])


def test_null_prompt_row_structure():
    row = null_prompt_row(options_count=4)
    validate_row(row)
    assert row["state"] == "N/A"
    assert len(row["options"]) == 4
    assert [opt["id"] for opt in row["options"]] == ["opt_A", "opt_B", "opt_C", "opt_D"]


def test_golden_section_search_simple_parabola():
    # Minimum of (x - 2.5)^2 is at x=2.5
    f = lambda x: (x - 2.5) ** 2
    res = golden_section_search(f, a=0.0, b=5.0)
    assert res == pytest.approx(2.5, abs=1e-4)


def test_fit_temperature_on_overconfident_data():
    # 10 samples: model outputs very confident logits [10.0, 0.0]
    # But only 7 out of 10 are actually label 0 (70% accuracy)
    # The raw confidence is ~99.99%, but true accuracy is 70%.
    # Therefore, optimal T must be > 1.0 to soften confidence to ~70%!
    data = []
    for i in range(10):
        label = 0 if i < 7 else 1
        data.append(([10.0, 0.0], label))

    optimal_t = fit_temperature(data, lower=0.5, upper=8.0)
    assert optimal_t > 2.0  # Optimal T must significantly soften the distribution

    nll_t1 = compute_nll_at_temp(data, 1.0)
    nll_opt = compute_nll_at_temp(data, optimal_t)
    assert nll_opt < nll_t1  # NLL improved
