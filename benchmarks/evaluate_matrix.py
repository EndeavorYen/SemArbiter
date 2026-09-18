"""Automated multi-model evaluation pipeline and matrix generator."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

from .evaluate import align, balanced_metric, basic, compute_ece, read_jsonl


def evaluate_predictions(gold_rows: list[dict], pred_rows: list[dict]) -> dict:
    """Compute comprehensive evaluation metrics for a single prediction run."""
    aligned = align(gold_rows, pred_rows)
    valid_rows = [r for r in aligned if r["status"] not in ("invalid", "missing")]

    if not valid_rows:
        return {
            "total_gold": len(gold_rows),
            "scored": len(pred_rows),
            "valid": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "ece_15": None,
            "mean_brier": None,
            "mean_confidence": None,
            "p50_forward_ms": None,
            "p99_forward_ms": None,
        }

    # Accuracy metrics
    basic_stats = basic(valid_rows)
    balanced_acc = balanced_metric(valid_rows)

    # Calibration metrics
    ece_data = compute_ece(valid_rows, n_bins=15)
    brier_scores = [r["brier"] for r in valid_rows if r.get("brier") is not None]
    confidences = [r["confidence"] for r in valid_rows if r.get("confidence") is not None]

    # Timing metrics from raw predictions
    forward_times = [
        p["forward_seconds"] * 1000.0
        for p in pred_rows
        if isinstance(p, dict) and "forward_seconds" in p and isinstance(p["forward_seconds"], (int, float))
    ]
    p50_fwd = None
    p99_fwd = None
    if forward_times:
        sorted_times = sorted(forward_times)
        p50_fwd = sorted_times[len(sorted_times) // 2]
        p99_fwd = sorted_times[int(len(sorted_times) * 0.99)]

    # Model metadata detection
    sample_p = pred_rows[0] if pred_rows else {}
    model_meta = sample_p.get("model", {})
    model_source = model_meta.get("source") if isinstance(model_meta, dict) else str(model_meta)
    readout = sample_p.get("readout", "standard")

    return {
        "model_source": model_source,
        "readout": readout,
        "total_gold": len(gold_rows),
        "scored": len(pred_rows),
        "valid": len(valid_rows),
        "accuracy": basic_stats["accuracy"],
        "balanced_accuracy": balanced_acc,
        "ece_15": ece_data["ece"] if ece_data else None,
        "mean_brier": sum(brier_scores) / len(brier_scores) if brier_scores else None,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "p50_forward_ms": p50_fwd,
        "p99_forward_ms": p99_fwd,
    }


def generate_markdown_table(matrix_results: dict[str, dict]) -> str:
    """Format matrix results into a GitHub-flavored markdown table."""
    headers = [
        "Experiment / Model",
        "Balanced Acc",
        "Overall Acc",
        "ECE (15 bins)",
        "Brier Score",
        "Avg Conf",
        "Forward P50 (ms)",
    ]
    rows = []
    for exp_name, stats in matrix_results.items():
        b_acc = f"{stats['balanced_accuracy']:.3f}" if stats["balanced_accuracy"] is not None else "—"
        o_acc = f"{stats['accuracy']:.3f}" if stats["accuracy"] is not None else "—"
        ece = f"{stats['ece_15']:.4f}" if stats["ece_15"] is not None else "—"
        brier = f"{stats['mean_brier']:.4f}" if stats["mean_brier"] is not None else "—"
        conf = f"{stats['mean_confidence']:.3f}" if stats["mean_confidence"] is not None else "—"
        fwd = f"{stats['p50_forward_ms']:.2f}" if stats["p50_forward_ms"] is not None else "—"
        rows.append([exp_name, b_acc, o_acc, ece, brier, conf, fwd])

    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]

    def fmt_row(items):
        return "| " + " | ".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(items)) + " |"

    def fmt_sep():
        return "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"

    lines = [fmt_row(headers), fmt_sep()]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path, help="Path to ground truth JSONL dataset")
    parser.add_argument(
        "--predictions-glob",
        nargs="+",
        required=True,
        help="Glob pattern or list of paths to prediction JSONL files",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (optional)")
    parser.add_argument("--format", choices=["json", "table", "both"], default="both")
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)

    # Expand globs
    pred_files = []
    for pattern in args.predictions_glob:
        matched = glob.glob(pattern)
        if matched:
            pred_files.extend(matched)
        else:
            p = Path(pattern)
            if p.exists():
                pred_files.append(str(p))

    pred_files = sorted(set(pred_files))
    if not pred_files:
        raise SystemExit(f"No prediction files found matching: {args.predictions_glob}")

    matrix = {}
    for pred_path in pred_files:
        name = Path(pred_path).stem
        try:
            p_rows = read_jsonl(pred_path)
            stats = evaluate_predictions(gold_rows, p_rows)
            matrix[name] = stats
        except Exception as exc:
            matrix[name] = {"error": str(exc)}

    if args.format in ("table", "both"):
        table = generate_markdown_table({k: v for k, v in matrix.items() if "error" not in v})
        print("\n" + "=" * 90)
        print("  SemIf Automated Evaluation Matrix")
        print(f"  Ground Truth: {args.gold} ({len(gold_rows)} rows)")
        print("=" * 90)
        print(table)
        print("=" * 90 + "\n")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2, ensure_ascii=False)
        print(f"Saved matrix evaluation to {args.output}")


if __name__ == "__main__":
    main()
