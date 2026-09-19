"""JevPilot 2.0 Hierarchical Closed-Loop Autonomous Driving Benchmark.

Evaluates Heuristics, Raw Flat LLM, and SemIf Hierarchical Decider across
7 comprehensive realistic traffic scenarios with standardized action space:
1. traffic_light_red: Signal intersection requiring stopping smoothly before line.
2. speed_zone_city: Segment speed limit enforcement (30mph city vs 65mph highway).
3. pedestrian_jaywalking: Pedestrian crossing mid-block, requiring yielding/stopping (tests casualty prevention).
4. roadside_parked_hazard: Parked hazard vehicle intruding into lane, requiring lateral nudge clearance.
5. cut_in_vehicle: Aggressive cut-in lead vehicle, testing collision avoidance.
6. sharp_curve: High-curvature mountain/highway curve, testing trajectory tracking.
7. sensor_anomaly: Corrupted sensor telemetry (NaN/noise) testing Helmholtz fail-safe.
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
logger = logging.getLogger("semif.benchmark.jevpilot2")


class JevPilot2Simulator:
    """Standardized 2D kinematic vehicle simulator for JevPilot 2.0 multi-scenario benchmark."""

    def __init__(self, scenario_type: str, seed: int = 42):
        self.scenario_type = scenario_type
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.dt = 0.05  # 20 Hz simulation step
        self.x = 0.0    # Lateral position (m)
        self.z = 0.0    # Longitudinal position (m)
        self.speed_mps = 16.0  # ~35 mph starting speed
        self.prev_accel = 0.0
        self.steer_angle = 0.0
        self.prev_steer = 0.0

        # Scenario parameters
        self.speed_ceiling_mps = 29.0  # Default ~65 mph
        self.intersection = None
        self.pedestrian = None
        self.roadside_obstacle = None
        self.cut_in_vehicle = None
        self.obstacle_z = 999.0
        self.has_anomaly = False

        if self.scenario_type == "sharp_curve":
            self.track_curvature = 0.65 if self.rng.random() > 0.5 else -0.65
        elif self.scenario_type == "traffic_light_red":
            self.track_curvature = 0.0
            self.intersection = {
                "control": "traffic_light",
                "stop_line_ahead_m": 55.0,
                "signal": "red",
                "stop_completed": False,
                "already_entered": False,
            }
        elif self.scenario_type == "speed_zone_city":
            self.track_curvature = 0.05
            self.speed_ceiling_mps = 13.4  # 30 mph city zone limit
        elif self.scenario_type == "pedestrian_jaywalking":
            self.track_curvature = 0.0
            self.pedestrian = {
                "z": 45.0,
                "x": 2.2,
                "speed_x": -1.0,
                "crossing": True,
            }
        elif self.scenario_type == "roadside_parked_hazard":
            self.track_curvature = 0.0
            self.roadside_obstacle = {
                "z": 50.0,
                "x": 1.2,
                "width": 1.9,
                "type": "parked_car_door_ajar",
            }
        elif self.scenario_type == "cut_in_vehicle":
            self.track_curvature = 0.0
            self.cut_in_vehicle = {
                "z": 40.0,
                "x": 2.5,
                "speed_mps": 10.0,
                "cut_in_time": 0.8,
            }
        elif self.scenario_type == "sensor_anomaly":
            self.track_curvature = 0.1
            self.has_anomaly = True
        else:
            self.track_curvature = 0.1 * (self.rng.random() - 0.5)

        # Quantitative Metrics Tracking
        self.collision = False
        self.pedestrian_casualty = False
        self.vehicle_collision = False
        self.off_track = False
        self.red_light_violation = False
        self.speeding_violation = False
        self.speeding_time_s = 0.0
        self.completed = False
        self.fail_safe_triggered = False

        self.jerks: List[float] = []
        self.steering_deltas: List[float] = []
        self.speeds: List[float] = []

    def get_observation(self) -> Dict[str, Any]:
        dist_to_line = None
        if self.intersection:
            dist_to_line = max(0.0, self.intersection["stop_line_ahead_m"] - self.z)

        # Pedestrian observation
        ped_obs = None
        if self.pedestrian:
            p_dist = max(0.0, self.pedestrian["z"] - self.z)
            ped_obs = {
                "distance_m": round(p_dist, 1),
                "lateral_offset_m": round(self.pedestrian["x"], 1),
                "is_crossing": True,
            }

        # Roadside obstacle observation
        roadside_obs = None
        if self.roadside_obstacle:
            r_dist = max(0.0, self.roadside_obstacle["z"] - self.z)
            roadside_obs = {
                "distance_m": round(r_dist, 1),
                "lateral_offset_m": round(self.roadside_obstacle["x"], 1),
                "type": self.roadside_obstacle["type"],
            }

        # Standard candidate vectors across all strategies
        # [speed, steer, route_error, offroad_fraction, collision_predicted, stop_at_line]
        candidates = {
            "v0": [round(min(self.speed_mps + 2.5, 30.0), 1), 0.0, round(self.x, 2), 0.0, False, False],
            "v1": [round(self.speed_mps, 1), 0.0, round(self.x, 2), 0.0, False, False],
            "v2": [round(max(0.0, self.speed_mps - 4.0), 1), 0.0, round(self.x, 2), 0.0, False, dist_to_line is not None and dist_to_line < 25.0],
            "v3": [round(max(0.0, self.speed_mps - 1.0), 1), -0.20, round(self.x - 0.8, 2), 0.0, False, False],  # Nudge left (dodge right hazard)
            "v4": [round(max(0.0, self.speed_mps - 1.0), 1), 0.20, round(self.x + 0.8, 2), 0.0, False, False],   # Nudge right
            "v5": [0.0, 0.0, round(self.x, 2), 0.0, False, dist_to_line is not None and dist_to_line < 15.0],    # Full stop at line
        }

        # Flag collision predictions for forward hazards
        if self.pedestrian and 0.0 < (self.pedestrian["z"] - self.z) < 25.0 and abs(self.pedestrian["x"]) < 1.6:
            candidates["v0"][4] = True
            candidates["v1"][4] = True

        if self.roadside_obstacle and 0.0 < (self.roadside_obstacle["z"] - self.z) < 30.0:
            if candidates["v0"][1] >= 0.0:
                candidates["v0"][4] = True
            if candidates["v1"][1] >= 0.0:
                candidates["v1"][4] = True

        if self.cut_in_vehicle and 0.0 < (self.cut_in_vehicle["z"] - self.z) < 20.0 and abs(self.cut_in_vehicle["x"]) < 1.2:
            candidates["v0"][4] = True
            candidates["v1"][4] = True

        obs = {
            "speed_mps": round(self.speed_mps, 2),
            "speed_ceiling_mps": round(self.speed_ceiling_mps, 2),
            "on_road": abs(self.x) <= 4.0,
            "lateral_offset_m": round(self.x, 2),
            "track_curvature": round(self.track_curvature, 3),
            "intersection": {
                "control": self.intersection["control"],
                "distance_to_line_m": round(dist_to_line, 1),
                "signal": self.intersection["signal"],
                "stop_completed": dist_to_line <= 1.0 and self.speed_mps < 1.0,
                "already_entered": dist_to_line == 0.0,
            } if self.intersection else None,
            "pedestrian": ped_obs,
            "roadside_obstacle": roadside_obs,
            "candidates": candidates,
        }

        if self.has_anomaly and self.t > 1.0:
            obs["anomaly"] = "SENSOR_PACKET_CORRUPTED_NAN"
            obs["speed_mps"] = float("nan")

        return obs

    def step(self, chosen_vector: List[float], is_ood: bool) -> Tuple[bool, Dict[str, Any]]:
        self.t += self.dt

        if is_ood:
            self.fail_safe_triggered = True

        target_speed = chosen_vector[0] if len(chosen_vector) > 0 else self.speed_mps
        target_steer = chosen_vector[1] if len(chosen_vector) > 1 else 0.0

        if is_ood:
            target_speed = 0.0
            target_steer = 0.0

        # Dynamics
        accel = (target_speed - self.speed_mps) * 4.0
        accel = max(-12.0, min(6.0, accel))
        jerk = (accel - self.prev_accel) / self.dt
        self.jerks.append(jerk)
        self.prev_accel = accel

        self.speed_mps = max(0.0, self.speed_mps + accel * self.dt)
        self.speeds.append(self.speed_mps)

        steer_delta = abs(target_steer - self.steer_angle)
        self.steering_deltas.append(steer_delta)
        self.steer_angle += (target_steer - self.steer_angle) * 6.0 * self.dt

        self.z += self.speed_mps * self.dt
        self.x += self.steer_angle * self.speed_mps * self.dt * 2.0
        self.x -= self.track_curvature * self.speed_mps * self.dt * 1.5

        # Dynamic Agents Progression
        if self.pedestrian and self.pedestrian["crossing"]:
            self.pedestrian["x"] += self.pedestrian["speed_x"] * self.dt

        if self.cut_in_vehicle and self.t >= self.cut_in_vehicle["cut_in_time"]:
            self.cut_in_vehicle["x"] = max(0.0, self.cut_in_vehicle["x"] - 1.5 * self.dt)
            self.cut_in_vehicle["z"] += self.cut_in_vehicle["speed_mps"] * self.dt

        # Safety & Boundary Checks
        if abs(self.x) > 4.5:
            self.off_track = True

        # Speed limit violation (after 1.2s adaptation buffer)
        if self.t > 1.2 and self.speed_mps > self.speed_ceiling_mps * 1.05:
            self.speeding_violation = True
            self.speeding_time_s += self.dt

        # Red light check
        if self.intersection:
            dist_to_line = self.intersection["stop_line_ahead_m"] - self.z
            if dist_to_line < 0.0 and self.intersection["signal"] == "red" and self.speed_mps > 2.0:
                self.red_light_violation = True

        # Pedestrian Casualty Check
        if self.pedestrian:
            dist_p = abs(self.pedestrian["z"] - self.z)
            lat_p = abs(self.pedestrian["x"] - self.x)
            if dist_p < 3.0 and lat_p < 1.6 and self.speed_mps > 2.0:
                self.collision = True
                self.pedestrian_casualty = True

        # Roadside Obstacle Collision Check
        if self.roadside_obstacle:
            dist_r = abs(self.roadside_obstacle["z"] - self.z)
            if dist_r < 3.5 and (self.x + 0.95) > (self.roadside_obstacle["x"] - 0.70):
                self.collision = True
                self.vehicle_collision = True

        # Cut-in Vehicle Collision Check
        if self.cut_in_vehicle:
            dist_c = self.cut_in_vehicle["z"] - self.z
            if 0.0 <= dist_c < 3.5 and abs(self.cut_in_vehicle["x"] - self.x) < 1.8:
                self.collision = True
                self.vehicle_collision = True

        terminated = False
        if self.collision or self.off_track or self.red_light_violation:
            terminated = True
        elif self.scenario_type == "sensor_anomaly" and self.fail_safe_triggered and self.speed_mps < 0.5:
            self.completed = True
            terminated = True
        elif self.scenario_type == "traffic_light_red" and self.intersection and (self.intersection["stop_line_ahead_m"] - self.z) <= 5.0 and self.speed_mps < 1.0:
            self.completed = True
            terminated = True
        elif self.scenario_type == "pedestrian_jaywalking" and self.pedestrian and (self.pedestrian["z"] - self.z) <= 5.0 and self.speed_mps < 1.0:
            self.completed = True
            terminated = True
        elif self.z >= 250.0 or self.t >= 15.0:
            self.completed = not (self.collision or self.off_track or self.red_light_violation)
            terminated = True

        return terminated, self.get_observation()


def run_jevpilot2_episode(engine: DecisionEngine, mode: str, scenario: str, seed: int) -> Dict[str, Any]:
    env = JevPilot2Simulator(scenario, seed=seed)
    latencies = []
    step_count = 0
    decision_interval = 2

    chosen_vec = [env.speed_mps, 0.0]
    is_ood = False

    while True:
        if step_count % decision_interval == 0:
            obs = env.get_observation()
            req = {
                "model": engine.model_name,
                "mode": mode,
                "state": obs,
                "questions": {
                    "motion": {
                        "type": "choice",
                        "instructions": "Decide drive or stop.",
                        "criteria": {"drive": None, "stop": None},
                    },
                    "vector": {
                        "type": "choice",
                        "instructions": "Choose a safe driving path.",
                        "criteria": {k: None for k in obs["candidates"].keys()},
                    }
                }
            }

            t0 = time.perf_counter()
            resp = engine.classify_jev(req)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            answers = resp.get("answers", {})
            chosen_id = answers.get("vector", {}).get("choice", "v1")
            chosen_vec = obs["candidates"].get(chosen_id, [env.speed_mps, 0.0])
            is_ood = resp.get("meta", {}).get("tier1_maneuver") == "HAZARD_AVOID" and "anomaly" in obs

        terminated, _ = env.step(chosen_vec, is_ood)
        step_count += 1
        if terminated:
            break

    # Calculate Jerk RMS
    jerk_rms = math.sqrt(sum(j ** 2 for j in env.jerks) / max(1, len(env.jerks))) if env.jerks else 0.0
    steering_oscillation = sum(env.steering_deltas)
    avg_speed = sum(env.speeds) / max(1, len(env.speeds)) if env.speeds else 0.0

    return {
        "scenario": scenario,
        "completed": env.completed,
        "collision": env.collision,
        "pedestrian_casualty": env.pedestrian_casualty,
        "vehicle_collision": env.vehicle_collision,
        "off_track": env.off_track,
        "red_light_violation": env.red_light_violation,
        "speeding_violation": env.speeding_violation,
        "speeding_time_s": round(env.speeding_time_s, 2),
        "jerk_rms": round(jerk_rms, 2),
        "steering_oscillation": round(steering_oscillation, 2),
        "avg_speed_mps": round(avg_speed, 2),
        "fail_safe_triggered": env.fail_safe_triggered,
        "latencies": latencies,
    }


def evaluate_jevpilot2_mode(engine: DecisionEngine, mode: str, episodes_per_sc: int = 10) -> Dict[str, Any]:
    scenarios = [
        "traffic_light_red",
        "speed_zone_city",
        "pedestrian_jaywalking",
        "roadside_parked_hazard",
        "cut_in_vehicle",
        "sharp_curve",
        "sensor_anomaly",
    ]
    results = []
    all_latencies = []

    for sc in scenarios:
        for ep in range(episodes_per_sc):
            seed = ep * 200 + hash(sc) % 10000
            res = run_jevpilot2_episode(engine, mode, sc, seed)
            results.append(res)
            all_latencies.extend(res["latencies"])

    total = len(results)
    completed = sum(1 for r in results if r["completed"])
    collisions = sum(1 for r in results if r["collision"])
    ped_casualties = sum(1 for r in results if r["pedestrian_casualty"])
    veh_collisions = sum(1 for r in results if r["vehicle_collision"])
    off_track = sum(1 for r in results if r["off_track"])
    red_lights = sum(1 for r in results if r["red_light_violation"])
    speeding = sum(1 for r in results if r["speeding_violation"])
    avg_jerk = sum(r["jerk_rms"] for r in results) / total
    avg_speed = sum(r["avg_speed_mps"] for r in results) / total

    all_latencies.sort()
    p50 = all_latencies[len(all_latencies) // 2] if all_latencies else 0.0
    p99 = all_latencies[int(len(all_latencies) * 0.99)] if all_latencies else 0.0

    return {
        "mode": mode,
        "total_episodes": total,
        "task_completion_rate": round(completed / total, 3),
        "collision_rate": round(collisions / total, 3),
        "pedestrian_casualties": ped_casualties,
        "pedestrian_casualty_rate": round(ped_casualties / total, 3),
        "vehicle_collisions": veh_collisions,
        "off_track_rate": round(off_track / total, 3),
        "red_light_violations": red_lights,
        "red_light_violation_rate": round(red_lights / total, 3),
        "speeding_violations": speeding,
        "speeding_violation_rate": round(speeding / total, 3),
        "jerk_rms": round(avg_jerk, 2),
        "avg_speed_mps": round(avg_speed, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p99_ms": round(p99, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="JevPilot 2.0 Hierarchical Closed-Loop Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--device", default=None, help="Compute device")
    parser.add_argument("--mock", action="store_true", help="Run with mock engine")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per scenario")
    parser.add_argument("--output", default="results/phase5-jevpilot-2.0-hierarchical-benchmark.json", help="Output path")
    args = parser.parse_args()

    engine = DecisionEngine(
        model_name=args.model,
        device=args.device,
        use_mock=args.mock,
        enable_graph=True,
    )

    modes = ["heuristic", "flat", "semif_hierarchical"]
    scenarios = [
        "traffic_light_red",
        "speed_zone_city",
        "pedestrian_jaywalking",
        "roadside_parked_hazard",
        "cut_in_vehicle",
        "sharp_curve",
        "sensor_anomaly",
    ]

    report: Dict[str, Any] = {
        "benchmark": "JevPilot 2.0 Hierarchical Closed-Loop Driving Benchmark",
        "scenarios": scenarios,
        "model": args.model if not args.mock else "MockDecisionEngine",
        "device": engine.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modes": {},
    }

    for mode in modes:
        logger.info(f"Evaluating strategy: {mode.upper()}...")
        summary = evaluate_jevpilot2_mode(engine, mode, episodes_per_sc=args.episodes)
        report["modes"][mode] = summary
        logger.info(
            f"[{mode.upper()}] Success: {summary['task_completion_rate']*100:.1f}%, "
            f"RedLights: {summary['red_light_violations']}, PedCasualties: {summary['pedestrian_casualties']}, "
            f"VehCollisions: {summary['vehicle_collisions']}, Speeding: {summary['speeding_violations']}, "
            f"Jerk: {summary['jerk_rms']} m/s^3"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
