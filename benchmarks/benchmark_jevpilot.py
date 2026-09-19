"""Closed-Loop Autonomous Driving Benchmark: JevPilot evaluation suite.

Evaluates Heuristic, Raw LLM, and SemIf Enhanced Qwen2.5-3B-Instruct across
simulated highway scenarios (sharp curves, sudden obstacles, high-speed cruising,
and sensor corruption anomalies).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repository root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.server import DecisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("semif.benchmark.jevpilot")


class SimulatorEnvironment:
    """Lightweight 2D kinematic bicycle/point-mass vehicle simulator for closed-loop testing."""

    def __init__(self, scenario_type: str, seed: int = 42):
        self.scenario_type = scenario_type
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.dt = 0.05  # 20 Hz simulation step
        self.x = 0.0   # Lateral position (m)
        self.z = 0.0   # Longitudinal distance (m)
        self.speed_kmh = 65.0
        self.steer_angle = 0.0

        # Scenario setup
        if self.scenario_type == "sharp_curve":
            self.track_curvature = 0.65 if self.rng.random() > 0.5 else -0.65
            self.obstacle_z = 999.0
            self.has_anomaly = False
        elif self.scenario_type == "sudden_obstacle":
            self.track_curvature = 0.0
            self.obstacle_z = 55.0  # Obstacle appears ahead
            self.has_anomaly = False
        elif self.scenario_type == "sensor_anomaly":
            self.track_curvature = 0.1
            self.obstacle_z = 999.0
            self.has_anomaly = True  # Corrupted sensor readings
        else:  # cruising
            self.track_curvature = 0.15 * (self.rng.random() - 0.5)
            self.obstacle_z = 999.0
            self.has_anomaly = False

        self.collision = False
        self.off_track = False
        self.completed = False
        self.fail_safe_triggered = False

    def step(self, target_steer: float, target_throttle: float, is_ood: bool) -> Tuple[bool, Dict[str, Any]]:
        self.t += self.dt

        if is_ood:
            self.fail_safe_triggered = True

        # Throttle / Braking dynamics
        accel = target_throttle * 30.0 if target_throttle > 0 else target_throttle * 60.0
        self.speed_kmh = max(0.0, min(130.0, self.speed_kmh + accel * self.dt))
        speed_mps = (self.speed_kmh * 1000.0) / 3600.0

        # Steering dynamics
        self.steer_angle += (target_steer - self.steer_angle) * 5.0 * self.dt
        self.z += speed_mps * self.dt
        self.x += self.steer_angle * (speed_mps * 0.08) * self.dt * 25.0
        self.x -= self.track_curvature * (speed_mps * 0.03) * self.dt * 25.0

        # Boundary checks
        if abs(self.x) > 5.5:
            self.off_track = True

        # Obstacle collision check
        dist_to_obs = self.obstacle_z - self.z
        if 0.0 <= dist_to_obs < 3.5 and abs(self.x) < 1.8:
            self.collision = True

        # Termination conditions
        terminated = False
        if self.collision or self.off_track:
            terminated = True
        elif self.scenario_type == "sensor_anomaly" and self.fail_safe_triggered and self.speed_kmh < 2.0:
            self.completed = True  # Successfully stopped on anomaly
            terminated = True
        elif self.z >= 300.0 or self.t >= 15.0:
            self.completed = not (self.collision or self.off_track)
            terminated = True

        telemetry = {
            "speed_kmh": self.speed_kmh,
            "lateral_offset_m": self.x,
            "track_curvature": self.track_curvature,
            "obstacle_distance_m": max(0.0, self.obstacle_z - self.z),
            "obstacle_lane": "ego" if self.obstacle_z < 900.0 else "none",
            "weather_condition": "clear",
        }
        if self.has_anomaly and self.t > 1.0:
            telemetry["anomaly"] = "SENSOR_PACKET_CHECKSUM_FAILURE_NAN"
            telemetry["speed_kmh"] = float("nan")

        return terminated, telemetry


def run_episode(engine: DecisionEngine, mode: str, scenario: str, seed: int) -> Dict[str, Any]:
    env = SimulatorEnvironment(scenario, seed=seed)
    telemetry = {
        "speed_kmh": env.speed_kmh,
        "lateral_offset_m": env.x,
        "track_curvature": env.track_curvature,
        "obstacle_distance_m": env.obstacle_z,
        "obstacle_lane": "ego" if env.obstacle_z < 900 else "none",
        "weather_condition": "clear",
    }

    latencies = []
    actions = []
    ood_flags = []

    step_count = 0
    decision_step = 2  # Run decision every 2 physics steps (10 Hz decision loop)

    while True:
        if step_count % decision_step == 0:
            dec = engine.decide(telemetry, mode=mode)
            steer = dec.get("target_steering", 0.0)
            throttle = dec.get("target_throttle", 0.3)
            is_ood = dec.get("is_ood", False)
            latencies.append(dec.get("latency_ms", 0.0))
            actions.append(dec.get("action", "maintain"))
            ood_flags.append(is_ood)

        terminated, telemetry = env.step(steer, throttle, is_ood)
        step_count += 1
        if terminated:
            break

    return {
        "scenario": scenario,
        "completed": env.completed,
        "collision": env.collision,
        "off_track": env.off_track,
        "fail_safe_triggered": env.fail_safe_triggered,
        "has_anomaly": env.has_anomaly,
        "final_z": env.z,
        "final_speed": env.speed_kmh,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "latencies": latencies,
        "actions": actions,
    }


def evaluate_mode(engine: DecisionEngine, mode: str, episodes_per_scenario: int = 10) -> Dict[str, Any]:
    scenarios = ["sharp_curve", "sudden_obstacle", "cruising", "sensor_anomaly"]
    results = []
    all_latencies = []

    for sc in scenarios:
        for ep in range(episodes_per_scenario):
            seed = ep * 100 + hash(sc) % 10000
            res = run_episode(engine, mode, sc, seed)
            results.append(res)
            all_latencies.extend(res["latencies"])

    total = len(results)
    completed = sum(1 for r in results if r["completed"])
    collisions = sum(1 for r in results if r["collision"])
    off_track = sum(1 for r in results if r["off_track"])

    anomaly_episodes = [r for r in results if r["has_anomaly"]]
    ood_detected = sum(1 for r in anomaly_episodes if r["fail_safe_triggered"])
    ood_recall = (ood_detected / len(anomaly_episodes)) if anomaly_episodes else 1.0

    all_latencies.sort()
    p50 = all_latencies[len(all_latencies) // 2] if all_latencies else 0.0
    p99 = all_latencies[int(len(all_latencies) * 0.99)] if all_latencies else 0.0

    return {
        "mode": mode,
        "episodes": total,
        "completion_rate": round(completed / total, 3),
        "collision_rate": round(collisions / total, 3),
        "off_track_rate": round(off_track / total, 3),
        "ood_anomaly_recall": round(ood_recall, 3),
        "latency_p50_ms": round(p50, 2),
        "latency_p99_ms": round(p99, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="SemIf JevPilot Closed-Loop Driving Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    parser.add_argument("--mock", action="store_true", help="Run with mock heuristic model")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per scenario")
    parser.add_argument("--output", default="results/phase5-jevpilot-benchmark.json", help="Output path")
    args = parser.parse_args()

    engine = DecisionEngine(
        model_name=args.model,
        device=args.device,
        use_mock=args.mock,
        enable_graph=True,
    )

    modes = ["heuristic", "raw", "semif"]
    benchmark_report: Dict[str, Any] = {
        "benchmark": "JevPilot Closed-Loop Highway Navigation",
        "model": args.model if not args.mock else "MockEngine",
        "device": engine.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modes": {},
    }

    for mode in modes:
        logger.info(f"--- Running Benchmark for Mode: {mode.upper()} ---")
        summary = evaluate_mode(engine, mode, episodes_per_scenario=args.episodes)
        benchmark_report["modes"][mode] = summary
        logger.info(f"[{mode.upper()}] Completion: {summary['completion_rate']*100:.1f}%, Collisions: {summary['collision_rate']*100:.1f}%, Off-Track: {summary['off_track_rate']*100:.1f}%, P50: {summary['latency_p50_ms']}ms")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    logger.info(f"Benchmark results successfully saved to: {out_path}")
    print(json.dumps(benchmark_report, indent=2))


if __name__ == "__main__":
    main()
