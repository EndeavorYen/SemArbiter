"""Break down classify_jev latency. CUDA only. Not a mock stand-in for 12ms."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

import torch

from benchmarks.benchmark_jevpilot_hierarchical import JevPilot2Simulator
from demo.server import DecisionEngine
from semif_phase1.cuda_graph import find_bucket
from semif_phase1.direct import encode_prompt, score
from semif_phase1.trajectory_sampler import compact_jev_state, vector_option_tag


def _median_ms(xs: list[float]) -> float:
    return round(statistics.median(xs) * 1000.0, 2)


def time_score(engine: DecisionEngine, row: dict, repeats: int = 8) -> dict:
    tokenizer = engine.tokenizer
    t0 = time.perf_counter()
    ids, slots, _ = encode_prompt(tokenizer, row, 4096)
    encode_s = time.perf_counter() - t0
    bucket = find_bucket(len(ids))
    prior = engine._prior_for(len(row["options"]))
    # warmup / capture
    score(engine.model, tokenizer, row, {}, sliced_head=True, prior_logits=prior, graph_runner=engine.graph_runner)
    torch.cuda.synchronize()
    totals, forwards, reads = [], [], []
    for _ in range(repeats):
        out = score(
            engine.model,
            tokenizer,
            row,
            {},
            sliced_head=True,
            prior_logits=prior,
            graph_runner=engine.graph_runner,
        )
        totals.append(out["total_seconds"])
        forwards.append(out["forward_seconds"])
        reads.append(out["readout"])
    return {
        "n_options": len(row["options"]),
        "input_tokens": len(ids),
        "bucket": bucket,
        "encode_ms": round(encode_s * 1000.0, 2),
        "forward_p50_ms": _median_ms(forwards),
        "total_p50_ms": _median_ms(totals),
        "readout": reads[-1],
        "graph_replay": "cuda-graph-bucket" in (reads[-1] or ""),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    engine = DecisionEngine(model_name="Qwen/Qwen2.5-3B-Instruct", device="cuda", use_mock=False, enable_graph=True)
    sim = JevPilot2Simulator("traffic_light_red", seed=42, raw_mode=True)
    sim.z = 20.0
    obs = sim.get_observation()
    cands = obs["candidates"]
    full_opts = [{"id": k, "description": (obs.get("candidate_meta") or {}).get(k, {}).get("description") or k} for k in cands]
    short_opts = [
        {"id": "A", "description": "keep lane"},
        {"id": "B", "description": "slow"},
        {"id": "C", "description": "stop"},
        {"id": "D", "description": "nudge left"},
        {"id": "E", "description": "nudge right"},
    ]
    compact_opts = [{"id": k, "description": vector_option_tag(vec)} for k, vec in cands.items()]
    full_row = {
        "id": "full",
        "state": obs,
        "question": "Choose a safe driving path.",
        "options": full_opts,
    }
    compact_row = {
        "id": "compact",
        "state": compact_jev_state(obs),
        "question": "Choose a safe driving path.",
        "options": compact_opts,
    }
    short_row = {
        "id": "short",
        "state": {"speed_mps": 16.0, "signal": "red"},
        "question": "Choose a safe driving path.",
        "options": short_opts,
    }
    tiny_full = {
        "id": "ids_only",
        "state": {"speed_mps": obs["speed_mps"], "intersection": obs["intersection"]},
        "question": "Choose a safe driving path.",
        "options": [{"id": k, "description": k} for k in cands],
    }
    report = {
        "device": str(engine.device),
        "model": engine.model_name,
        "full_closed_loop_bloated": time_score(engine, full_row),
        "compact_state": time_score(engine, compact_row),
        "short_5opt": time_score(engine, short_row),
        "sixteen_ids_tiny_state": time_score(engine, tiny_full),
    }
    t0 = time.perf_counter()
    engine.classify_jev({
        "model": engine.model_name,
        "mode": "flat",
        "state": obs,
        "questions": {"vector": {"type": "choice", "criteria": {k: None for k in cands}}},
    })
    torch.cuda.synchronize()
    report["classify_jev_one_shot_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
