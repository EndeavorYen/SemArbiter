"""Headless SIL control-quality diagnostic (#117).

Tier 1 (--fast): CPU heuristic sweep, no GPU, no LLM load.
Tier 2 (--cuda): Qwen2.5-3B-Instruct closed loop on one CUDA GPU.

The web demo remains the high-fidelity check. This script is the fast
observer: it fails on weave, reverse chatter, deadband escape, and stall
even when the episode never collides.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    _p = str(_path)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from demo.server import DecisionEngine
from semif_phase1.control_diagnostics import (
    Diagnosis,
    diagnose_trace,
    format_issue_body,
    format_issue_title,
    format_report,
    make_sample,
)
from semif_phase1.trajectory_sampler import vector_option_tag
from benchmarks.benchmark_jevpilot_hierarchical import (
    ALL_SCENARIOS,
    run_jevpilot2_episode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("semif.diagnose.control")

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SCENARIO = "speed_zone_city"
ISSUE_CAP = 3
REPO = "EndeavorYen/SemIf"


def sample_from_env(env: Any, chosen_id: str, chosen_vec: Sequence[Any]) -> Dict[str, Any]:
    signal = None
    if getattr(env, "intersection", None):
        signal = env.intersection.get("signal")
    tags = vector_option_tag(list(chosen_vec), ego_x=env.x, signal=signal) if chosen_vec else str(chosen_id)
    chosen_speed = float(chosen_vec[0]) if chosen_vec else float(env.speed_mps)
    return make_sample(
        t=float(env.t),
        x=float(env.x),
        z=float(env.z),
        speed=float(env.speed_mps),
        steer=float(env.steer_angle),
        lane_offset=float(env.x),
        chosen_id=str(chosen_id),
        chosen_tags=tags,
        signal=signal,
        chosen_speed=chosen_speed,
    )


def collect_episode(
    engine: DecisionEngine,
    *,
    mode: str,
    scenario: str,
    seed: int,
    raw_mode: bool = False,
    vision_mode: str = "off",
    use_camera_obstacles: bool = True,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    trace: List[Dict[str, Any]] = []

    def on_step(env: Any, chosen_id: str, chosen_vec: Sequence[Any]) -> None:
        trace.append(sample_from_env(env, chosen_id, chosen_vec))

    episode = run_jevpilot2_episode(
        engine,
        mode,
        scenario,
        seed,
        raw_mode=raw_mode,
        vision_mode=vision_mode,
        on_step=on_step,
        use_camera_obstacles=use_camera_obstacles,
    )
    return episode, trace


def diagnose_episode(
    engine: DecisionEngine,
    *,
    mode: str,
    scenario: str,
    seed: int,
    raw_mode: bool = False,
    vision_mode: str = "off",
    use_camera_obstacles: bool = True,
) -> tuple[Dict[str, Any], Diagnosis]:
    episode, trace = collect_episode(
        engine,
        mode=mode,
        scenario=scenario,
        seed=seed,
        raw_mode=raw_mode,
        vision_mode=vision_mode,
        use_camera_obstacles=use_camera_obstacles,
    )
    diagnosis = diagnose_trace(trace, seed=seed, scenario=scenario, mode=mode)
    return episode, diagnosis


def file_issue(
    diagnosis: Diagnosis,
    *,
    runner: Optional[Callable[..., Any]] = None,
    repo: str = REPO,
) -> Dict[str, Any]:
    """Create a GitHub issue. Tests inject `runner`; live path uses `gh`."""
    if diagnosis.ok:
        return {"filed": False, "reason": "clean"}
    title = format_issue_title(diagnosis)
    body = format_issue_body(diagnosis)
    run = runner or subprocess.run
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".md", delete=False
    ) as handle:
        handle.write(body)
        body_path = handle.name
    try:
        proc = run(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                repo,
                "--title",
                title,
                "--body-file",
                body_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        Path(body_path).unlink(missing_ok=True)
    stdout = getattr(proc, "stdout", "") or ""
    stderr = getattr(proc, "stderr", "") or ""
    code = int(getattr(proc, "returncode", 1) or 0)
    return {
        "filed": code == 0,
        "title": title,
        "url": stdout.strip(),
        "stderr": stderr.strip(),
        "returncode": code,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless SIL observer for weave, reverse chatter, deadband escape, and stall (#117)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Tier 1: CPU heuristic mock sweep (no GPU, no LLM).",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Tier 2: neural closed loop on one CUDA GPU.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Single-seed repro (default 42 if neither sweep flag).")
    parser.add_argument("--seeds", type=int, default=None, help="Episodes per scenario for a sweep.")
    parser.add_argument(
        "--scenario",
        default=None,
        help=f"One scenario name. Default {DEFAULT_SCENARIO} for --seed, all scenarios for sweeps.",
    )
    parser.add_argument("--mode", default=None, help="heuristic or flat. Default follows --fast/--cuda.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mock", action="store_true", help="Force MockDecisionEngine.")
    parser.add_argument(
        "--file-issues",
        action="store_true",
        help=f"Call `gh issue create` for failing episodes (cap {ISSUE_CAP}). Off by default.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path. Create-only.")
    return parser


def _engine_for(args: argparse.Namespace) -> tuple[DecisionEngine, str]:
    if args.fast and args.cuda:
        raise SystemExit("use --fast or --cuda, not both")
    mock = bool(args.mock or args.fast or not args.cuda)
    mode = args.mode or ("flat" if args.cuda else "heuristic")
    device = args.device or ("cuda" if args.cuda else None)
    if args.cuda and mock:
        raise SystemExit("--cuda refuses MockDecisionEngine")
    engine = DecisionEngine(
        model_name=args.model,
        device=device,
        use_mock=mock,
        enable_graph=bool(args.cuda),
    )
    if args.cuda and str(engine.device) != "cuda":
        raise SystemExit(f"--cuda requested but engine.device={engine.device}")
    return engine, mode


def _jobs(args: argparse.Namespace) -> List[tuple[str, int]]:
    if args.fast:
        scenarios = [args.scenario] if args.scenario else list(ALL_SCENARIOS)
        n = args.seeds if args.seeds is not None else 10
        base = args.seed if args.seed is not None else 42
        return [(sc, base + i) for sc in scenarios for i in range(n)]
    if args.cuda:
        scenarios = [args.scenario] if args.scenario else list(ALL_SCENARIOS)
        n = args.seeds if args.seeds is not None else 2
        base = args.seed if args.seed is not None else 42
        return [(sc, base + i) for sc in scenarios for i in range(n)]
    seed = args.seed if args.seed is not None else 42
    scenario = args.scenario or DEFAULT_SCENARIO
    return [(scenario, seed)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    engine, mode = _engine_for(args)
    jobs = _jobs(args)
    logger.info(
        "diagnose mode=%s mock=%s device=%s jobs=%s",
        mode,
        engine.use_mock,
        engine.device,
        len(jobs),
    )
    rows: List[Dict[str, Any]] = []
    filed = 0
    t0 = time.perf_counter()
    for scenario, seed in jobs:
        episode, diagnosis = diagnose_episode(
            engine,
            mode=mode,
            scenario=scenario,
            seed=seed,
            use_camera_obstacles=not args.fast,
        )
        text = format_report(diagnosis)
        print(text)
        print()
        row = {
            "scenario": scenario,
            "seed": seed,
            "mode": mode,
            "ok": diagnosis.ok,
            "violations": [
                {"kind": v.kind, "t0": v.t0, "metrics": v.metrics}
                for v in diagnosis.violations
            ],
            "completed": episode.get("completed"),
            "collision": episode.get("collision"),
            "red_light_violation": episode.get("red_light_violation"),
            "off_track": episode.get("off_track"),
        }
        if args.file_issues and not diagnosis.ok and filed < ISSUE_CAP:
            result = file_issue(diagnosis)
            row["issue"] = result
            filed += int(bool(result.get("filed")))
            logger.info("issue create: %s", result)
        rows.append(row)
    elapsed = time.perf_counter() - t0
    n_fail = sum(1 for r in rows if not r["ok"])
    summary = {
        "benchmark": "control_quality_observer",
        "mode": mode,
        "n_episodes": len(rows),
        "n_fail": n_fail,
        "elapsed_s": round(elapsed, 2),
        "file_issues": bool(args.file_issues),
        "episodes": rows,
    }
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "episodes"},
            indent=2,
        )
    )
    if args.output:
        out = Path(args.output)
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("wrote %s", out)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
