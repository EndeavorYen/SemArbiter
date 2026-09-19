"""Ablate option-tag wording vs latency and closed-loop quality. CUDA only."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

import semif_phase1.trajectory_sampler as ts
from benchmarks.benchmark_jevpilot_hierarchical import JevPilot2Simulator, evaluate_jevpilot2_mode
from demo.server import DecisionEngine
from semif_phase1.cuda_graph import find_bucket
from semif_phase1.direct import encode_prompt
from semif_phase1.trajectory_sampler import compact_jev_state, vector_option_tag


def prompt_stats(engine: DecisionEngine, obs: dict, style: str) -> dict:
    options = [{"id": k, "description": vector_option_tag(v, style=style)} for k, v in obs["candidates"].items()]
    row = {
        "id": f"tag-{style}",
        "state": compact_jev_state(obs),
        "question": "Choose a safe driving path.",
        "options": options,
    }
    ids, _, _ = encode_prompt(engine.tokenizer, row, 4096)
    buckets = getattr(engine.graph_runner, "buckets", (256, 512, 1024))
    return {
        "style": style,
        "tokens": len(ids),
        "bucket": find_bucket(len(ids), buckets),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    engine = DecisionEngine(model_name="Qwen/Qwen2.5-3B-Instruct", device="cuda", use_mock=False)
    sim = JevPilot2Simulator("traffic_light_red", seed=42, raw_mode=True)
    sim.z = 20.0
    obs = sim.get_observation()
    report = {"prompt": [], "closed_loop": []}
    for style in ("csv", "words", "verbose"):
        report["prompt"].append(prompt_stats(engine, obs, style))
    print(json.dumps({"prompt": report["prompt"]}, indent=2), flush=True)
    for style in ("csv", "words", "verbose"):
        ts.OPTION_TAG_STYLE = style
        t0 = time.perf_counter()
        summary = evaluate_jevpilot2_mode(
            engine,
            "flat",
            episodes_per_sc=2,
            base_seed=42,
            raw_mode=True,
        )
        elapsed = time.perf_counter() - t0
        quality = summary["driving_quality"]
        report["closed_loop"].append(
            {
                "style": style,
                "clean_completion_rate": quality["clean_completion_rate"],
                "incidents": quality["incidents"],
                "latency_p50_ms": summary["latency_p50_ms"],
                "latency_p99_ms": summary["latency_p99_ms"],
                "scenario_completed": {
                    sc: row["completed"] for sc, row in summary["scenario_breakdown"].items()
                },
                "wall_s": round(elapsed, 1),
            }
        )
        print(json.dumps(report["closed_loop"][-1], indent=2), flush=True)
    Path("results/phase5-jevpilot-option-tag-ablation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    ranked = sorted(
        report["closed_loop"],
        key=lambda row: (row["clean_completion_rate"], -row["latency_p50_ms"]),
        reverse=True,
    )
    print("RANKING", json.dumps(ranked, indent=2), flush=True)


if __name__ == "__main__":
    main()
