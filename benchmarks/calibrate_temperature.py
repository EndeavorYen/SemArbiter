"""Optimize scalar temperature T on validation predictions to minimize NLL and ECE.

Uses a dependency-free 1D golden-section search to find the unique global
optimum of Negative Log-Likelihood over T > 0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import evaluate


def compute_nll_at_temp(data: list[tuple[list[float], int]], temperature: float) -> float:
    """Compute average Negative Log-Likelihood across samples given temperature T."""
    if temperature <= 0:
        return float("inf")
    total_nll = 0.0
    for logits, label_idx in data:
        scaled = [z / temperature for z in logits]
        max_val = max(scaled)
        # log-sum-exp
        lse = max_val + math.log(sum(math.exp(v - max_val) for v in scaled))
        nll = lse - scaled[label_idx]
        total_nll += nll
    return total_nll / len(data)


def golden_section_search(f, a: float = 0.1, b: float = 10.0, tol: float = 1e-5, max_iter: int = 100) -> float:
    """Find scalar minimizer of 1D unimodal function f over [a, b]."""
    phi = (math.sqrt(5) - 1) / 2  # ~0.618
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2


def fit_temperature(data: list[tuple[list[float], int]], lower: float = 0.1, upper: float = 10.0) -> float:
    """Fit optimal temperature scalar minimizing NLL."""
    if not data:
        raise ValueError("Need at least one sample to fit temperature")
    return golden_section_search(lambda t: compute_nll_at_temp(data, t), a=lower, b=upper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold manifest JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to raw predictions JSONL with option_logits")
    parser.add_argument("--output", type=Path, required=True, help="Path to save calibration report JSON")
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output path must be new")

    gold_rows = evaluate.read_jsonl(args.gold)
    pred_rows = evaluate.read_jsonl(args.predictions)

    truth_map = evaluate.indexed(gold_rows, "gold")
    aligned_data = []

    for pred in pred_rows:
        pid = pred["id"]
        if pid not in truth_map:
            continue
        item = truth_map[pid]
        option_ids = [opt["id"] for opt in item["options"]]
        label_idx = item["label"]
        logits = pred.get("option_logits")
        if logits is None or len(logits) != len(option_ids):
            continue
        aligned_data.append((logits, label_idx))

    if not aligned_data:
        raise ValueError("No valid aligned rows with option_logits found")

    # 1. Evaluate baseline at T=1.0
    nll_t1 = compute_nll_at_temp(aligned_data, 1.0)
    
    # 2. Fit optimal T*
    optimal_t = fit_temperature(aligned_data, lower=0.2, upper=5.0)
    nll_opt = compute_nll_at_temp(aligned_data, optimal_t)

    # 3. Compute ECE and Brier before and after scaling
    def get_eval_rows(temp: float):
        rows = []
        for logits, label_idx in aligned_data:
            scaled = [z / temp for z in logits]
            max_v = max(scaled)
            weights = [math.exp(v - max_v) for v in scaled]
            total = sum(weights)
            probs = [w / total for w in weights]
            chosen_idx = max(range(len(probs)), key=probs.__getitem__)
            conf = probs[chosen_idx]
            correct = (chosen_idx == label_idx)
            brier = sum((p - (k == label_idx)) ** 2 for k, p in enumerate(probs))
            rows.append({"confidence": conf, "correct": correct, "brier": brier})
        return rows

    rows_t1 = get_eval_rows(1.0)
    rows_opt = get_eval_rows(optimal_t)

    ece_t1 = evaluate.compute_ece(rows_t1, n_bins=15)
    ece_opt = evaluate.compute_ece(rows_opt, n_bins=15)
    brier_t1 = sum(r["brier"] for r in rows_t1) / len(rows_t1)
    brier_opt = sum(r["brier"] for r in rows_opt) / len(rows_opt)

    report = {
        "samples_evaluated": len(aligned_data),
        "baseline_temperature": 1.0,
        "optimal_temperature": round(optimal_t, 4),
        "baseline_metrics": {
            "nll": nll_t1,
            "brier": brier_t1,
            "ece_15": ece_t1["ece"],
            "mce": ece_t1["mce"],
        },
        "calibrated_metrics": {
            "nll": nll_opt,
            "brier": brier_opt,
            "ece_15": ece_opt["ece"],
            "mce": ece_opt["mce"],
        },
        "improvements": {
            "nll_delta": nll_opt - nll_t1,
            "ece_reduction": (ece_t1["ece"] - ece_opt["ece"]) if (ece_t1["ece"] and ece_opt["ece"]) else None,
            "brier_delta": brier_opt - brier_t1,
        },
        "reliability_bins_baseline": ece_t1["bins"],
        "reliability_bins_calibrated": ece_opt["bins"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as dest:
        dest.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print(json.dumps({
        "optimal_temperature": report["optimal_temperature"],
        "baseline_ece": report["baseline_metrics"]["ece_15"],
        "calibrated_ece": report["calibrated_metrics"]["ece_15"],
        "ece_reduction": report["improvements"]["ece_reduction"],
    }, indent=2))


if __name__ == "__main__":
    main()
