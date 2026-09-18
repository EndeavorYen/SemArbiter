"""Evaluation helper and comparison reporting for Mapika/decider-2b versus SemIf baselines."""

import argparse
import json
from pathlib import Path

import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold manifest JSONL")
    parser.add_argument("--decider-predictions", type=Path, required=True, help="Path to decider predictions JSONL")
    parser.add_argument("--baseline-predictions", type=Path, default=None, help="Optional baseline predictions JSONL (e.g. Qwen3.5-4B)")
    parser.add_argument("--output", type=Path, required=True, help="Path to output evaluation JSON")
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output path must be new")

    gold = evaluate.read_jsonl(args.gold)
    decider_preds = evaluate.read_jsonl(args.decider_predictions)
    baseline_preds = evaluate.read_jsonl(args.baseline_predictions) if args.baseline_predictions else None

    report = evaluate.evaluate(gold, decider_preds, comparison=baseline_preds)
    
    # Extract high-level summary metrics
    summary_table = {
        "gold_samples": len(gold),
        "scored_decider_samples": len(decider_preds),
        "mean_family_balanced_accuracy": report.get("mean_family_balanced_accuracy"),
        "mean_family_macro_f1": report.get("mean_family_macro_f1"),
        "coverage": report.get("coverage"),
    }

    # Extract ECE and Brier across families
    all_briers = []
    all_eces = []
    for fam_name, fam_data in report.get("family_results", {}).items():
        if fam_data.get("brier") is not None:
            all_briers.append(fam_data["brier"])
        if fam_data.get("ece") is not None:
            all_eces.append(fam_data["ece"])

    if all_briers:
        summary_table["mean_brier_score"] = sum(all_briers) / len(all_briers)
    if all_eces:
        summary_table["mean_ece_15"] = sum(all_eces) / len(all_eces)

    report["comparison_summary"] = summary_table

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as dest:
        dest.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print(json.dumps(summary_table, indent=2))


if __name__ == "__main__":
    main()
