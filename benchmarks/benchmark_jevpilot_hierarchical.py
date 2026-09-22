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
8. ambiguous_priority: Unmarked junction with a simultaneous arrival (semantic exclusive).
9. construction_detour: Temporary lane-closed sign requiring a lateral detour.
10. emergency_vehicle: Siren vehicle approaching from behind requiring a pull-over.
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
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.server import DecisionEngine
from semif_phase1.lateral import DETOUR_STEER, lateral_pd
from semif_phase1.trajectory_sampler import (
    VECTOR_INSTRUCTIONS,
    candidates_as_vecs,
    candidates_meta,
    sample_trajectories,
)
from benchmarks.driving_quality import compare_driving, driving_quality
from benchmarks.sdi import (
    SEMANTIC_EXCLUSIVE_SCENARIOS,
    scenario_seed,
    semantic_driving_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("semif.benchmark.jevpilot2")


class JevPilot2Simulator:
    """Standardized 2D kinematic vehicle simulator for JevPilot 2.0 multi-scenario benchmark."""

    def __init__(self, scenario_type: str, seed: int = 42, raw_mode: bool = False):
        self.scenario_type = scenario_type
        self.seed = int(seed)
        self.raw_mode = bool(raw_mode)
        self.rng = random.Random(self.seed)
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
        self.other_vehicle = None
        self.construction = None
        self.emergency_vehicle = None
        self.obstacle_z = 999.0
        self.has_anomaly = False
        self.priority_violation = False
        self.detour_violation = False
        self.emergency_violation = False

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
            self.track_curvature = 0.0
            self.has_anomaly = True
        elif self.scenario_type == "ambiguous_priority":
            self.track_curvature = 0.0
            self.intersection = {
                "control": "unmarked",
                "stop_line_ahead_m": 48.0,
                "signal": "none",
                "stop_completed": False,
                "already_entered": False,
            }
            self.other_vehicle = {
                "z": 46.0,
                "x": 3.4,
                "speed_mps": 11.0,
                "arriving": True,
            }
        elif self.scenario_type == "construction_detour":
            self.track_curvature = 0.0
            self.construction = {
                "z": 42.0,
                "x": 0.15,
                "width": 2.4,
                "sign": "LANE_CLOSED_FOLLOW_DETOUR",
            }
        elif self.scenario_type == "emergency_vehicle":
            self.track_curvature = 0.0
            self.emergency_vehicle = {
                "z": -8.0,
                "x": 0.0,
                "speed_mps": 24.0,
                "siren": True,
            }
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

    def ground_truth_obstacles(self) -> List[Dict[str, Any]]:
        """Ego-frame boxes from the simulator. --fast uses this instead of empty IPM."""
        out: List[Dict[str, Any]] = []

        def add(kind: str, x: float, z: float) -> None:
            out.append(
                {
                    "kind": kind,
                    "rel_x": round(float(x) - self.x, 2),
                    "rel_z": round(float(z) - self.z, 2),
                }
            )

        if self.pedestrian:
            add("pedestrian", self.pedestrian["x"], self.pedestrian["z"])
        if self.roadside_obstacle:
            add("roadside", self.roadside_obstacle["x"], self.roadside_obstacle["z"])
        if self.cut_in_vehicle:
            add("vehicle", self.cut_in_vehicle["x"], self.cut_in_vehicle["z"])
        if self.other_vehicle:
            add("vehicle", self.other_vehicle["x"], self.other_vehicle["z"])
        if self.construction:
            add("roadside", self.construction["x"], self.construction["z"])
        return out

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

        construction_obs = None
        if self.construction:
            c_dist = max(0.0, self.construction["z"] - self.z)
            construction_obs = {
                "distance_m": round(c_dist, 1),
                "lateral_offset_m": round(self.construction["x"], 1),
                "sign": self.construction["sign"],
                "lane": "ego",
            }

        other_obs = None
        if self.other_vehicle:
            other_obs = {
                "distance_m": round(self.other_vehicle["z"] - self.z, 1),
                "lateral_offset_m": round(self.other_vehicle["x"] - self.x, 1),
                "speed_mps": round(self.other_vehicle["speed_mps"], 1),
                "arriving": True,
            }

        emergency_obs = None
        if self.emergency_vehicle:
            emergency_obs = {
                "distance_m": round(abs(self.z - self.emergency_vehicle["z"]), 1),
                "behind": self.emergency_vehicle["z"] < self.z,
                "speed_mps": round(self.emergency_vehicle["speed_mps"], 1),
                "siren": True,
            }

        from semif_phase1.ipm import camera_obstacles_from_blobs
        from semif_phase1.vision import blobs_from_frame, render_scenario_frame

        if getattr(self, "use_camera_obstacles", True):
            frame = render_scenario_frame(self.scenario_type, self)
            obstacles = camera_obstacles_from_blobs(blobs_from_frame(frame))
        else:
            obstacles = self.ground_truth_obstacles()

        stop_line_z = None
        if self.intersection:
            stop_line_z = float(self.intersection["stop_line_ahead_m"])

        samples = sample_trajectories(
            ego_x=self.x,
            ego_z=self.z,
            speed=self.speed_mps if math.isfinite(self.speed_mps) else 0.0,
            curvature=self.track_curvature,
            stop_line_z=stop_line_z,
            obstacles=obstacles,
            seed=self.seed + int(self.t * 20),
            speed_ceiling=self.speed_ceiling_mps,
            current_steer=self.steer_angle,
        )
        candidates = candidates_as_vecs(samples)
        candidate_meta = candidates_meta(samples)

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
            "construction": construction_obs,
            "other_vehicle": other_obs,
            "emergency_vehicle": emergency_obs,
            "raw_mode": self.raw_mode,
            "seed": self.seed,
            "candidates": candidates,
            "candidate_meta": candidate_meta,
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

        # Dynamics
        accel = (target_speed - self.speed_mps) * 4.0
        accel = max(-12.0, min(6.0, accel))
        jerk = (accel - self.prev_accel) / self.dt
        self.jerks.append(jerk)
        self.prev_accel = accel

        self.speed_mps = max(0.0, self.speed_mps + accel * self.dt)
        self.speeds.append(self.speed_mps)

        offset_dot = (
            self.steer_angle * self.speed_mps * 2.0
            - self.track_curvature * self.speed_mps * 1.5
        )
        u_sel = float(target_steer)
        if abs(u_sel) <= DETOUR_STEER:
            # Plant: x += steer * v * dt * 2.0; x -= kappa * v * dt * 1.5. Hold x with steer = (1.5/2) * kappa.
            u_sel = (1.5 / 2.0) * float(self.track_curvature)
        target_steer = lateral_pd(u_sel, self.x, offset_dot)
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

        if self.other_vehicle and self.other_vehicle["arriving"]:
            self.other_vehicle["z"] += self.other_vehicle["speed_mps"] * self.dt * 0.35
            self.other_vehicle["x"] = max(0.4, self.other_vehicle["x"] - 1.1 * self.dt)

        if self.emergency_vehicle:
            self.emergency_vehicle["z"] += self.emergency_vehicle["speed_mps"] * self.dt

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

        if self.construction:
            dist_k = self.construction["z"] - self.z
            if 0.0 <= dist_k < 3.2 and abs(self.x - self.construction["x"]) < 1.35:
                self.collision = True
                self.vehicle_collision = True
                self.detour_violation = True

        if self.other_vehicle:
            dist_o = self.other_vehicle["z"] - self.z
            if 0.0 <= dist_o < 3.2 and abs(self.other_vehicle["x"] - self.x) < 1.7:
                self.collision = True
                self.vehicle_collision = True
                self.priority_violation = True

        if self.emergency_vehicle:
            catching = self.emergency_vehicle["z"] - self.z
            if abs(self.x) < 0.55 and -4.0 < catching < 6.0 and self.t > 1.4:
                self.emergency_violation = True
                self.collision = True
                self.vehicle_collision = True

        terminated = False
        if self.scenario_type == "sensor_anomaly" and self.fail_safe_triggered and self.speed_mps < 0.5:
            self.completed = True
            terminated = True
        elif self.collision or self.off_track or self.red_light_violation:
            terminated = True
        elif self.scenario_type == "traffic_light_red" and self.intersection and (self.intersection["stop_line_ahead_m"] - self.z) <= 5.0 and self.speed_mps < 1.0:
            self.completed = True
            terminated = True
        elif self.scenario_type == "pedestrian_jaywalking" and self.pedestrian and (self.pedestrian["z"] - self.z) <= 5.0 and self.speed_mps < 1.0:
            self.completed = True
            terminated = True
        elif self.scenario_type == "ambiguous_priority" and self.other_vehicle and (self.other_vehicle["z"] - self.z) <= 6.0 and self.speed_mps < 1.2:
            self.completed = True
            terminated = True
        elif self.scenario_type == "construction_detour" and self.construction and (self.construction["z"] - self.z) < -4.0 and abs(self.x) >= 0.55:
            self.completed = True
            terminated = True
        elif self.scenario_type == "emergency_vehicle" and abs(self.x) >= 0.7 and self.t >= 2.0:
            self.completed = True
            terminated = True
        elif self.z >= 250.0 or self.t >= 15.0:
            self.completed = not (self.collision or self.off_track or self.red_light_violation)
            terminated = True

        return terminated, self.get_observation()


def run_jevpilot2_episode(
    engine: DecisionEngine,
    mode: str,
    scenario: str,
    seed: int,
    raw_mode: bool = False,
    vision_mode: str = "off",
    on_step: Optional[Any] = None,
    use_camera_obstacles: bool = True,
) -> Dict[str, Any]:
    if vision_mode == "clip":
        raise RuntimeError(
            "JevPilot2 vision_mode=clip is refused; official vision scores use the web city onboard camera"
        )
    env = JevPilot2Simulator(scenario, seed=seed, raw_mode=raw_mode)
    env.use_camera_obstacles = use_camera_obstacles
    latencies = []
    step_count = 0
    decision_interval = 2

    chosen_vec = [env.speed_mps, 0.0]
    chosen_id = "v1"
    last_obs: Optional[Dict[str, Any]] = None
    is_ood = False

    while True:
        if step_count % decision_interval == 0:
            obs = env.get_observation()
            if vision_mode == "synthetic":
                from semif_phase1.vision import vision_from_scenario

                obs = dict(obs)
                obs["vision"] = vision_from_scenario(scenario)
            req = {
                "model": engine.model_name,
                "mode": mode,
                "raw_mode": raw_mode,
                "state": obs,
                "questions": {
                    "vector": {
                        "type": "choice",
                        "instructions": VECTOR_INSTRUCTIONS,
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
            chosen_vec = list(obs["candidates"].get(chosen_id, [env.speed_mps, 0.0]))
            is_ood = bool(resp.get("meta", {}).get("true_ood"))
            last_obs = obs

        terminated, _ = env.step(chosen_vec, is_ood)
        if on_step is not None:
            on_step(env, chosen_id, chosen_vec, last_obs)
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
        "true_ood": env.has_anomaly,
        "priority_violation": env.priority_violation,
        "detour_violation": env.detour_violation,
        "emergency_violation": env.emergency_violation,
        "seed": seed,
        "raw_mode": raw_mode,
        "vision_mode": vision_mode,
        "latencies": latencies,
    }


CORE_SCENARIOS = [
    "traffic_light_red",
    "speed_zone_city",
    "pedestrian_jaywalking",
    "roadside_parked_hazard",
    "cut_in_vehicle",
    "sharp_curve",
    "sensor_anomaly",
]

ALL_SCENARIOS = CORE_SCENARIOS + list(SEMANTIC_EXCLUSIVE_SCENARIOS[:-1])  # sensor_anomaly already in core


def evaluate_jevpilot2_mode(
    engine: DecisionEngine,
    mode: str,
    episodes_per_sc: int = 10,
    base_seed: int = 42,
    raw_mode: bool = False,
    scenarios: Optional[List[str]] = None,
    vision_mode: str = "off",
) -> Dict[str, Any]:
    scenarios = list(scenarios or ALL_SCENARIOS)
    results = []
    all_latencies = []

    for sc in scenarios:
        for ep in range(episodes_per_sc):
            seed = scenario_seed(base_seed, sc, ep)
            res = run_jevpilot2_episode(
                engine, mode, sc, seed, raw_mode=raw_mode, vision_mode=vision_mode
            )
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

    scenario_breakdown: Dict[str, Dict[str, Any]] = {}
    for sc in scenarios:
        chunk = [r for r in results if r["scenario"] == sc]
        n = max(1, len(chunk))
        scenario_breakdown[sc] = {
            "episodes": len(chunk),
            "completed": sum(1 for r in chunk if r["completed"]),
            "collisions": sum(1 for r in chunk if r["collision"]),
            "red_light_violations": sum(1 for r in chunk if r["red_light_violation"]),
            "pedestrian_casualties": sum(1 for r in chunk if r["pedestrian_casualty"]),
            "vehicle_collisions": sum(1 for r in chunk if r["vehicle_collision"]),
            "fail_safe": sum(1 for r in chunk if r["fail_safe_triggered"]),
        }

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
        "driving_quality": driving_quality(results),
        "deprecated_sdi": semantic_driving_index(results),
        "base_seed": base_seed,
        "raw_mode": raw_mode,
        "scenario_breakdown": scenario_breakdown,
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser(description="JevPilot 2.0 Hierarchical Closed-Loop Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--device", default=None, help="Compute device")
    parser.add_argument("--mock", action="store_true", help="Run with mock engine")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Base simulation seed (default 42)")
    parser.add_argument(
        "--seeds",
        default="42,123,2026",
        help="Comma-separated seed matrix. Empty string uses only --seed.",
    )
    parser.add_argument("--raw-mode", action="store_true", help="Disable stop-at-line candidate injection")
    parser.add_argument("--output", default="results/phase5-jevpilot-2.0-hierarchical-benchmark.json", help="Output path")
    args = parser.parse_args()

    engine = DecisionEngine(
        model_name=args.model,
        device=args.device,
        use_mock=args.mock,
        enable_graph=True,
    )

    modes = ["heuristic", "flat"]
    scenarios = list(ALL_SCENARIOS)
    seed_text = (args.seeds or "").strip()
    seeds = [int(s) for s in seed_text.split(",") if s.strip()] if seed_text else [args.seed]
    if args.seed not in seeds:
        seeds = [args.seed] + seeds

    report: Dict[str, Any] = {
        "benchmark": "JevPilot 2.0 Hierarchical Closed-Loop Driving Benchmark",
        "scenarios": scenarios,
        "model": args.model if not args.mock else "MockDecisionEngine",
        "device": engine.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "raw_mode": args.raw_mode,
        "modes": {},
        "quality_by_seed": {},
    }

    mode_episodes: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in modes}

    for mode in modes:
        logger.info(f"Evaluating strategy: {mode.upper()}...")
        summary = evaluate_jevpilot2_mode(
            engine,
            mode,
            episodes_per_sc=args.episodes,
            base_seed=seeds[0],
            raw_mode=args.raw_mode,
            scenarios=scenarios,
        )
        compact = {k: v for k, v in summary.items() if k != "episodes"}
        report["modes"][mode] = compact
        mode_episodes[mode].extend(summary["episodes"])
        quality = summary["driving_quality"]
        logger.info(
            f"[{mode.upper()}] Clean: {quality['clean_completion_rate']*100:.1f}%, "
            f"Incidents: {quality['incident_rate']*100:.1f}%, "
            f"RedLights: {summary['red_light_violations']}, "
            f"PedCasualties: {summary['pedestrian_casualties']}, "
            f"VehCollisions: {summary['vehicle_collisions']}, Speeding: {summary['speeding_violations']}, "
            f"OOD-FP: {quality['ood_false_positive_rate']:.2f}"
        )

    if len(seeds) > 1:
        for seed in seeds[1:]:
            seed_eps: Dict[str, List[Dict[str, Any]]] = {}
            for mode in modes:
                extra = evaluate_jevpilot2_mode(
                    engine,
                    mode,
                    episodes_per_sc=max(1, args.episodes // 2),
                    base_seed=seed,
                    raw_mode=args.raw_mode,
                    scenarios=scenarios,
                )
                seed_eps[mode] = extra["episodes"]
                mode_episodes[mode].extend(extra["episodes"])
            report["quality_by_seed"][str(seed)] = {
                mode: driving_quality(eps) for mode, eps in seed_eps.items()
            }

    report["driving_quality"] = compare_driving(mode_episodes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
