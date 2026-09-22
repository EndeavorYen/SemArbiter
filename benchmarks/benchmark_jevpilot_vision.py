"""Mock loop stays on synthetic labels. Official scores are two web-city laps."""

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
from benchmarks.web_city_vision import OFFICIAL_REPORT, run_official_web_city
from demo.server import DecisionEngine
from semif_phase1.latency_telemetry import align_web_export, summarize_latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--lap-timeout-s", type=float, default=180)
    parser.add_argument(
        "--web-telemetry",
        default=None,
        help="Path to SEMIF_TELEMETRY.exportJSON() dump for Web vs local latency alignment",
    )
    args = parser.parse_args()

    if not args.mock:
        output = Path(args.output) if args.output else OFFICIAL_REPORT
        report = run_official_web_city(seed=args.seed, output=output, lap_timeout_s=args.lap_timeout_s)
        print(json.dumps({
            "output": report["official_report"],
            "seed": report["seed"],
            "device": report["device"],
            "mock": report["mock"],
            "world": report["world"],
            "laps": report["laps"],
        }, indent=2))
        return

    output = args.output or "results/phase5-jevpilot-vision-mock.json"
    engine = DecisionEngine(use_mock=True)
    vision_backend = "synthetic"
    arms = {}
    loop_latencies: dict = {}
    vision_arm = "synthetic"
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
        "model": "MockDecisionEngine",
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
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "ranking": {arm: data["driving_quality"]["ranking"] for arm, data in arms.items()},
        "clean": {
            arm: {m: data["driving_quality"]["modes"][m]["clean_completion_rate"] for m in data["driving_quality"]["modes"]}
            for arm, data in arms.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
