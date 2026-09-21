"""Telemetry-only vs synthetic-vision closed loop. Same sampler; vision is evidence only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.benchmark_jevpilot_hierarchical import evaluate_jevpilot2_mode
from benchmarks.driving_quality import compare_driving
from demo.server import DecisionEngine
from semif_phase1.latency_telemetry import align_web_export, summarize_latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/phase5-jevpilot-vision-mock.json")
    parser.add_argument(
        "--web-telemetry",
        default=None,
        help="Path to SEMIF_TELEMETRY.exportJSON() dump for Web vs local latency alignment",
    )
    args = parser.parse_args()

    engine = DecisionEngine(use_mock=args.mock, device=None if args.mock else "cuda")
    vision_backend = "synthetic" if args.mock else "unknown"
    if not args.mock:
        from semif_phase1.vision import get_vision_encoder

        enc = get_vision_encoder()
        if enc.backend == "stub":
            raise SystemExit("vision encoder required for official scores (got stub)")
        vision_backend = enc.backend
    arms = {}
    loop_latencies: dict = {}
    vision_arm = "synthetic" if args.mock else "clip"
    for vision_mode in ("off", vision_arm):
        eps = {}
        compact = {}
        for mode in ("heuristic", "flat"):
            summary = evaluate_jevpilot2_mode(
                engine,
                mode,
                episodes_per_sc=args.episodes,
                base_seed=args.seed,
                raw_mode=True,
                vision_mode=vision_mode,
            )
            eps[mode] = summary["episodes"]
            compact[mode] = {k: v for k, v in summary.items() if k != "episodes"}
            samples = []
            for ep in summary["episodes"]:
                samples.extend(ep["latencies"])
            loop_latencies[(mode, vision_mode)] = samples
        arms[vision_mode] = {
            "modes": compact,
            "driving_quality": compare_driving(eps),
        }

    report = {
        "benchmark": "JevPilot vision vs telemetry (same sampler)",
        "seed": args.seed,
        "mock": args.mock,
        "model": engine.model_name if not args.mock else "MockDecisionEngine",
        "device": engine.device,
        "vision_backend": vision_backend,
        "visual_prefix": os.environ.get("SEMIF_VISION_PREFIX", "0"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arms": arms,
    }
    if args.web_telemetry:
        web = json.loads(Path(args.web_telemetry).read_text(encoding="utf-8"))
        local_samples = loop_latencies.get(("flat", vision_arm), [])
        report["web_alignment"] = align_web_export(
            web,
            {
                "classifier_ms": summarize_latency(local_samples),
                "mode": "flat",
                "vision_mode": vision_arm,
            },
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "ranking": {arm: data["driving_quality"]["ranking"] for arm, data in arms.items()},
        "clean": {
            arm: {m: data["driving_quality"]["modes"][m]["clean_completion_rate"] for m in data["driving_quality"]["modes"]}
            for arm, data in arms.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
