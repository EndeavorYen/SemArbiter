"""SemIf Live Decision Server: Real-time inference streaming for JevPilot 3D simulator.

Provides high-throughput, low-latency decision endpoints via WebSockets and HTTP.
Serves Qwen2.5-3B-Instruct with Sliced LM Head projection, Prior Logit Debiasing,
and Helmholtz Free Energy OOD detection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from semif_phase1.action_tree import Branch, Leaf, Node, sensors_corrupt, walk_action_tree
from semif_phase1.trajectory_sampler import compact_jev_state, partition_ids, vector_option_tag
from semif_phase1.core import LETTERS, apply_prior_calibration, null_prompt_row, softmax
from semif_phase1.direct import score
from semif_phase1.gating import compute_free_energy, gate_decision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("semif.server")

# Driving action schema
DRIVING_ACTIONS = [
    {"id": "steer_left", "description": "Steer left to follow leftward road curvature or change lane left to bypass obstacle."},
    {"id": "maintain", "description": "Maintain current cruising heading and lane alignment."},
    {"id": "steer_right", "description": "Steer right to follow rightward road curvature or change lane right to bypass obstacle."},
    {"id": "brake", "description": "Apply vehicle brakes immediately to decelerate or perform emergency stop."},
    {"id": "accelerate", "description": "Accelerate vehicle forward while road and forward distance are clear."},
]

ACTION_IDS = [a["id"] for a in DRIVING_ACTIONS]

# Precomputed empirical null prior for 5-option Qwen2.5-3B-Instruct
DEFAULT_5OPT_PRIOR = [37.25, 26.375, 24.5, 23.25, 28.75]

MANEUVERS = [
    {"id": "proceed", "description": "Keep the current lane and a useful speed."},
    {"id": "stop", "description": "Come to a halt for a red or yellow signal, a person in the path, or an unmarked yield."},
    {"id": "go_around", "description": "Move laterally to pass a blockage, follow a detour, or pull aside."},
    {"id": "slow", "description": "Reduce speed to the posted limit without a full stop."},
    {"id": "fail_safe", "description": "Sensor data is corrupted or out of distribution; refuse the observation."},
]
MANEUVER_IDS = [m["id"] for m in MANEUVERS]
MANEUVER_BY_ID = {m["id"]: m for m in MANEUVERS}

def build_drive_tree(candidates: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Node:
    """Bucket this frame's sampled trajectories. Leaves are sampled ids."""
    halt, lateral, lane = partition_ids(candidates, meta)
    if not lane:
        lane = list(candidates.keys()) or ["t00"]
    slow_ids = tuple(halt) if halt else tuple(lane)
    lat_ids = tuple(lateral) if lateral else tuple(lane)
    lane_ids = tuple(lane)
    speed_node = Node(
        id="speed",
        question="Is current speed above the posted ceiling? Choose slow only if speed_mps > speed_ceiling_mps.",
        evidence_keys=("speed_mps", "speed_ceiling_mps"),
        skip_if_empty=False,
        default_branch="cruise",
        branches=(
            Branch("slow", "speed_mps is greater than speed_ceiling_mps.", Leaf(slow_ids)),
            Branch("cruise", "speed_mps is at or under the posted ceiling.", Leaf(lane_ids)),
        ),
    )
    around_node = Node(
        id="around",
        question="Is a lateral move required? Choose go_around only for a detour, parked hazard, or emergency vehicle.",
        evidence_keys=("construction", "roadside_obstacle", "emergency_vehicle"),
        skip_if_empty=True,
        default_branch="stay_in_lane",
        branches=(
            Branch("go_around", "A detour, parked hazard, or siren requires moving aside.", Leaf(lat_ids)),
            Branch("stay_in_lane", "No blockage requiring a lateral move.", speed_node),
        ),
    )
    halt_next: Any = Leaf(tuple(halt)) if halt else around_node
    return Node(
        id="halt",
        question="Must the vehicle halt now? Choose must_stop only for a red or yellow signal not yet cleared, a person in the path, or an unmarked yield.",
        evidence_keys=("intersection", "pedestrian", "other_vehicle"),
        skip_if_empty=True,
        default_branch="keep_moving",
        branches=(
            Branch("must_stop", "Red or yellow signal, a person in the path, or an unmarked yield.", halt_next),
            Branch("keep_moving", "No halt required for a signal or a person.", around_node),
        ),
    )


DRIVE_TREE = build_drive_tree(
    {"t00": [10.0, 0.0, 0.0, 0.0, False, False], "t01": [0.0, 0.0, 0.0, 0.0, False, True]},
    {"t01": {"end_speed": 0.0, "steer": 0.0, "stop_at_line": True}},
)

INTENT_TO_MANEUVER = {
    "OOD_FAIL_SAFE": "fail_safe",
    "YIELD_RED_LIGHT": "stop",
    "YIELD_PEDESTRIAN": "stop",
    "YIELD_UNMARKED_PRIORITY": "stop",
    "FOLLOW_DETOUR": "go_around",
    "GIVE_WAY_EMERGENCY": "go_around",
    "AVOID_ROADSIDE_OBSTACLE": "go_around",
    "HAZARD_AVOID": "go_around",
    "GOVERN_SPEED": "slow",
    "SAFE_CRUISE": "proceed",
    "CRUISE": "proceed",
}


def coarse_maneuver(intent: str) -> str:
    return INTENT_TO_MANEUVER.get(intent, "proceed")


def _finite_or_corrupt(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.upper() in {"ANOMALY_CORRUPTED", "NAN", "INFINITY", "-INFINITY"}:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isnan(number) or math.isinf(number)


def rule_maneuver_tree(state: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Oracle walk for tests and mock. Same splits as the neural tree."""
    path: List[str] = []
    if state.get("anomaly") is not None or _finite_or_corrupt(state.get("speed_mps")):
        path.append("sensor_bad")
        return "fail_safe", path
    path.append("sensor_ok")

    intersection = state.get("intersection") if isinstance(state.get("intersection"), dict) else {}
    signal = str(intersection.get("signal") or "").lower()
    entered = bool(intersection.get("already_entered"))
    stopped = bool(intersection.get("stop_completed"))
    pedestrian = state.get("pedestrian") if isinstance(state.get("pedestrian"), dict) else None
    other = state.get("other_vehicle") if isinstance(state.get("other_vehicle"), dict) else None
    ped_near = bool(pedestrian and pedestrian.get("distance_m") is not None and pedestrian["distance_m"] <= 40)
    unmarked = str(intersection.get("control") or "").lower() in {"unmarked", "none", ""}
    other_arriving = bool(other and other.get("arriving") and (other.get("distance_m") or 99) <= 30)
    if (signal in {"red", "yellow"} and not entered and not stopped) or ped_near or (unmarked and other_arriving):
        path.append("must_stop")
        return "stop", path
    path.append("keep_moving")

    construction = state.get("construction") if isinstance(state.get("construction"), dict) else None
    roadside = state.get("roadside_obstacle") if isinstance(state.get("roadside_obstacle"), dict) else None
    emergency = state.get("emergency_vehicle") if isinstance(state.get("emergency_vehicle"), dict) else None
    around = False
    if construction and construction.get("distance_m") is not None and construction["distance_m"] <= 45:
        around = True
    if roadside and roadside.get("distance_m") is not None and roadside["distance_m"] <= 45:
        around = True
    if emergency and emergency.get("siren") and emergency.get("behind", True):
        around = True
    if around:
        path.append("go_around")
        return "go_around", path
    path.append("stay_in_lane")

    speed = state.get("speed_mps")
    ceiling = state.get("speed_ceiling_mps")
    try:
        speeding = float(speed) > float(ceiling) * 1.05
    except (TypeError, ValueError):
        speeding = False
    if speeding:
        path.append("slow")
        return "slow", path
    path.append("proceed")
    return "proceed", path


class DecisionEngine:
    """Manages model loading, CUDA graphs, prior calibration, and real-time execution."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        use_mock: bool = False,
        enable_graph: bool = True,
    ):
        self.model_name = model_name
        self.use_mock = use_mock
        self.enable_graph = enable_graph
        self.model = None
        self.tokenizer = None
        self.device = None
        self.graph_runner = None
        self._vis_proj = None
        self._vis_null_priors: Dict[int, List[float]] = {}
        self.prior_logits: List[float] = DEFAULT_5OPT_PRIOR
        self.priors_by_n: Dict[int, List[float]] = {len(DEFAULT_5OPT_PRIOR): list(DEFAULT_5OPT_PRIOR)}
        self.stats = {
            "total_decisions": 0,
            "total_latency_ms": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
        }

        if not self.use_mock:
            self._load_model(device)
        else:
            logger.info("DecisionEngine initialized in MOCK mode (zero GPU required).")

    def _load_model(self, requested_device: Optional[str]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if requested_device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device

        logger.info(f"Loading model '{self.model_name}' on device '{self.device}'...")
        t0 = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

        load_sec = time.perf_counter() - t0
        logger.info(f"Model loaded successfully in {load_sec:.2f} seconds.")

        # Compute empirical null prior
        try:
            logger.info("Calibrating context-free unconditional null prior for 5 options...")
            null_row = null_prompt_row(options_count=len(DRIVING_ACTIONS))
            res = score(self.model, self.tokenizer, null_row, {}, sliced_head=True)
            self.prior_logits = res["option_logits"]
            self.priors_by_n[len(self.prior_logits)] = list(self.prior_logits)
            logger.info(f"5-option null prior calibrated: {self.prior_logits}")
        except Exception as e:
            logger.warning(f"Failed to auto-compute null prior: {e}. Using precomputed prior: {self.prior_logits}")

        # CUDA Graph initialization if on CUDA
        if self.device == "cuda" and self.enable_graph:
            try:
                from semif_phase1.cuda_graph import BucketGraphRunner
                logger.info("Initializing BucketGraphRunner for shape-bucketed inference...")
                self.graph_runner = BucketGraphRunner(
                    self.model, buckets=(256, 512, 1024), device="cuda", warmup_on_init=True
                )
                logger.info("CUDA Graph buckets captured successfully.")
            except Exception as e:
                logger.warning(f"CUDA Graph warmup skipped or failed: {e}. Falling back to dynamic sliced LM head.")
                self.graph_runner = None

    def _prior_for(self, n: int) -> Optional[List[float]]:
        """Return an n-dimensional null prior. Never slice a mismatched 5-opt prior."""
        if n < 2:
            return None
        cached = self.priors_by_n.get(n)
        if cached is not None and len(cached) == n:
            return cached
        if self.use_mock or self.model is None or self.tokenizer is None:
            return None
        try:
            null_row = null_prompt_row(options_count=n)
            res = score(self.model, self.tokenizer, null_row, {}, sliced_head=True)
            logits = list(res["option_logits"])
            if len(logits) != n:
                return None
            self.priors_by_n[n] = logits
            logger.info(f"{n}-option null prior calibrated: {logits}")
            return logits
        except Exception as e:
            logger.warning(f"Failed to calibrate {n}-option null prior: {e}")
            return None

    def _visual_prefix_from_state(self, state: Any):
        from semif_phase1.visual_prefix import VisualPrefixProjector, uses_visual_prefix
        from semif_phase1.vision import get_vision_encoder

        vision = state.get("vision") if isinstance(state, dict) else None
        if not uses_visual_prefix(vision):
            return None
        encoder = get_vision_encoder()
        patches = encoder.last_patches
        if patches is None:
            return None
        hidden = int(self.model.config.hidden_size)
        in_dim = int(patches.shape[-1])
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        if self._vis_proj is None or self._vis_proj.in_dim != in_dim or self._vis_proj.out_dim != hidden:
            self._vis_proj = VisualPrefixProjector(in_dim, hidden, device, dtype)
        return self._vis_proj(patches)

    def _visual_null_prior(self, n: int, row: Dict[str, Any]) -> Optional[List[float]]:
        cached = self._vis_null_priors.get(n)
        if cached is not None and len(cached) == n:
            return cached
        from semif_phase1.vision import get_vision_encoder

        encoder = get_vision_encoder()
        blank = encoder.null_patches()
        if blank is None or self._vis_proj is None:
            return None
        prefix = self._vis_proj(blank)
        try:
            res = score(
                self.model,
                self.tokenizer,
                null_prompt_row(options_count=n),
                {},
                sliced_head=True,
                prior_logits=None,
                graph_runner=None,
                visual_prefix=prefix,
            )
            logits = list(res["option_logits"])
            if len(logits) != n:
                return None
            self._vis_null_priors[n] = logits
            return logits
        except Exception as exc:
            logger.warning(f"Visual null prior failed: {exc}")
            return None

    def build_driving_row(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert driving state into SemIf standard categorical decision row."""
        def _clean_num(val: Any, default: float) -> Any:
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return "CORRUPTED_ANOMALY_NAN"
                return round(f, 2)
            except (TypeError, ValueError):
                return "CORRUPTED_PAYLOAD"

        speed = _clean_num(telemetry.get("speed_kmh", 60.0), 60.0)
        lat_offset = _clean_num(telemetry.get("lateral_offset_m", 0.0), 0.0)
        curve = _clean_num(telemetry.get("track_curvature", 0.0), 0.0)
        dist_ahead = _clean_num(telemetry.get("obstacle_distance_m", 100.0), 100.0)
        obs_lane = str(telemetry.get("obstacle_lane", "none"))
        weather = str(telemetry.get("weather_condition", "clear"))
        anomaly = telemetry.get("anomaly", None)

        state: Dict[str, Any] = {
            "speed_kmh": speed,
            "lateral_offset_m": lat_offset,
            "track_curvature": curve,
            "obstacle_distance_m": dist_ahead,
            "obstacle_lane": obs_lane,
            "weather_condition": weather,
        }
        if anomaly is not None:
            state["sensor_anomaly"] = str(anomaly)

        return {
            "id": f"drive_{int(time.time() * 1000)}",
            "state": state,
            "question": "Given the current vehicle telemetry, select the optimal immediate driving maneuver to maintain safety and progress.",
            "options": DRIVING_ACTIONS,
        }

    def _heuristic_decision(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Fast rule-based PID / heuristic baseline."""
        t0 = time.perf_counter()
        speed = float(telemetry.get("speed_kmh", 60.0))
        lat_offset = float(telemetry.get("lateral_offset_m", 0.0))
        curve = float(telemetry.get("track_curvature", 0.0))
        dist_ahead = float(telemetry.get("obstacle_distance_m", 100.0))
        obs_lane = str(telemetry.get("obstacle_lane", "none"))
        is_corrupt = telemetry.get("anomaly") is not None or math.isnan(speed) or math.isnan(lat_offset)

        if is_corrupt:
            action = "brake"
            probs = [0.05, 0.05, 0.05, 0.80, 0.05]
            energy = 5.0
            is_ood = True
        elif dist_ahead < 25.0 and obs_lane in ("ego", "center"):
            action = "brake"
            probs = [0.1, 0.05, 0.1, 0.7, 0.05]
            energy = -45.0
            is_ood = False
        elif curve < -0.3 or lat_offset > 0.8:
            action = "steer_left"
            probs = [0.75, 0.15, 0.05, 0.02, 0.03]
            energy = -48.0
            is_ood = False
        elif curve > 0.3 or lat_offset < -0.8:
            action = "steer_right"
            probs = [0.05, 0.15, 0.75, 0.02, 0.03]
            energy = -48.0
            is_ood = False
        elif speed < 50.0 and dist_ahead > 60.0:
            action = "accelerate"
            probs = [0.05, 0.2, 0.05, 0.05, 0.65]
            energy = -47.0
            is_ood = False
        else:
            action = "maintain"
            probs = [0.1, 0.7, 0.1, 0.05, 0.05]
            energy = -49.0
            is_ood = False

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "action": action,
            "target_steering": -0.8 if action == "steer_left" else (0.8 if action == "steer_right" else 0.0),
            "target_throttle": -1.0 if action == "brake" else (0.8 if action == "accelerate" else 0.3),
            "probabilities": {act: prob for act, prob in zip(ACTION_IDS, probs)},
            "calibrated": False,
            "free_energy": energy,
            "is_ood": is_ood,
            "latency_ms": latency_ms,
            "mode": "heuristic",
        }

    def decide(self, telemetry: Dict[str, Any], mode: str = "semif") -> Dict[str, Any]:
        """Compute immediate maneuver decision given vehicle state."""
        if mode == "heuristic" or self.use_mock:
            return self._heuristic_decision(telemetry)

        t0 = time.perf_counter()
        row = self.build_driving_row(telemetry)

        use_prior = self.prior_logits if mode == "semif" else None

        # Execute direct scoring pass
        scored = score(
            self.model,
            self.tokenizer,
            row,
            {},
            sliced_head=True,
            prior_logits=use_prior,
            graph_runner=self.graph_runner,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract probabilities and raw/calibrated logits
        probs = scored["probabilities"]
        option_logits = scored["calibrated_logits"] if mode == "semif" else scored["option_logits"]
        energy = compute_free_energy(option_logits)

        # Gate with Helmholtz free energy
        gate = gate_decision(
            probs,
            option_logits,
            ACTION_IDS,
            min_confidence=0.35,
            energy_threshold=-20.0,
            abstain_option_id="brake",
        )

        final_action = gate["selected_id"]
        is_ood = gate["disposition"] == "abstain" or "high_free_energy_ood" in gate["rejection_reasons"]

        # If OOD anomaly detected in SemIf mode, fail-safe to Emergency Brake
        if is_ood and mode == "semif":
            final_action = "brake"

        prob_dict = {aid: p for aid, p in zip(ACTION_IDS, probs)}

        # Continuous control translation using SemIf calibrated probabilities
        p_left = prob_dict.get("steer_left", 0.0)
        p_right = prob_dict.get("steer_right", 0.0)
        p_accel = prob_dict.get("accelerate", 0.0)
        p_brake = prob_dict.get("brake", 0.0)
        p_maintain = prob_dict.get("maintain", 0.0)

        if final_action == "brake":
            steer = 0.0
            throttle = -1.0
        elif is_ood:
            steer = 0.0
            throttle = -1.0
        else:
            # Continuous smooth steering: net lateral force
            steer = float(p_right - p_left) * 1.2
            steer = max(-1.0, min(1.0, steer))
            # Continuous smooth throttle
            throttle = float(p_accel * 0.8 + p_maintain * 0.4 - p_brake * 1.0)
            throttle = max(-1.0, min(1.0, throttle))

        # Update stats
        self.stats["total_decisions"] += 1
        self.stats["total_latency_ms"] += latency_ms
        self.stats["min_latency_ms"] = min(self.stats["min_latency_ms"], latency_ms)
        self.stats["max_latency_ms"] = max(self.stats["max_latency_ms"], latency_ms)

        return {
            "action": final_action,
            "target_steering": steer,
            "target_throttle": throttle,
            "probabilities": prob_dict,
            "calibrated": mode == "semif",
            "free_energy": energy,
            "is_ood": is_ood,
            "rejection_reasons": gate["rejection_reasons"],
            "latency_ms": latency_ms,
            "mode": mode,
        }

    def _determine_tier1_maneuver(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """JevPilot 2.0 Tier 1 Strategic Maneuver Reasoning.
        Analyzes traffic light signals, road-segment speed ceiling, and hazard proximity.
        """
        if not isinstance(state, dict):
            return {"intent": "CRUISE", "directive": "Standard cruise and lane tracking."}

        anomaly = state.get("anomaly")
        speed_raw = state.get("speed_mps")
        speed_corrupt = False
        try:
            if speed_raw is not None and (math.isnan(float(speed_raw)) or math.isinf(float(speed_raw))):
                speed_corrupt = True
        except (TypeError, ValueError):
            speed_corrupt = True
        if anomaly is not None or speed_corrupt:
            return {
                "intent": "OOD_FAIL_SAFE",
                "directive": "SENSOR OOD / free-energy spike. Reject the observation and fail-closed to a stop.",
                "target_stop": True,
                "ood_fail_safe": True,
            }

        emer = state.get("emergency_vehicle")
        if isinstance(emer, dict) and emer.get("siren"):
            behind = bool(emer.get("behind", True))
            e_dist = emer.get("distance_m", 99.0)
            if behind and e_dist is not None and e_dist <= 40.0:
                return {
                    "intent": "GIVE_WAY_EMERGENCY",
                    "directive": f"EMERGENCY VEHICLE {e_dist}m behind with siren. Pull right and yield.",
                    "must_avoid_collision": True,
                    "pull_right": True,
                }

        cons = state.get("construction")
        if isinstance(cons, dict):
            c_dist = cons.get("distance_m", 100.0)
            if c_dist is not None and c_dist <= 45.0:
                return {
                    "intent": "FOLLOW_DETOUR",
                    "directive": f"TEMPORARY CONSTRUCTION SIGN ({cons.get('sign', 'LANE_CLOSED')}) at {c_dist}m. Take the detour, do not stay in the closed ego lane.",
                    "must_avoid_collision": True,
                    "nudge_left": True,
                }

        other = state.get("other_vehicle")
        intersection = state.get("intersection") if isinstance(state.get("intersection"), dict) else None
        if isinstance(other, dict) and other.get("arriving"):
            unmarked = bool(intersection and str(intersection.get("control", "")).lower() in ("unmarked", "none", ""))
            o_dist = other.get("distance_m", 99.0)
            if unmarked and o_dist is not None and o_dist <= 30.0:
                return {
                    "intent": "YIELD_UNMARKED_PRIORITY",
                    "directive": f"UNMARKED JUNCTION: other vehicle arriving at {o_dist}m. Yield; do not take geometric right-of-way.",
                    "target_stop": True,
                }

        # 1. Traffic Light & Intersection Analysis
        intersection = state.get("intersection")
        if isinstance(intersection, dict):
            sig = str(intersection.get("signal", "")).lower()
            dist = intersection.get("distance_to_line_m")
            already_entered = bool(intersection.get("already_entered", False))
            stop_completed = bool(intersection.get("stop_completed", False))

            if sig in ("red", "yellow") and not already_entered and not stop_completed:
                return {
                    "intent": "YIELD_RED_LIGHT",
                    "directive": f"RED/YELLOW SIGNAL ({sig.upper()}) at {dist}m ahead. Bring vehicle to a smooth halt before stop line. Do not enter intersection.",
                    "target_stop": True,
                }

        # 2. Road Speed Ceiling Compliance
        speed_mps = state.get("speed_mps")
        speed_ceiling = state.get("speed_ceiling_mps")
        if speed_mps is not None and speed_ceiling is not None:
            try:
                s = float(speed_mps)
                sc = float(speed_ceiling)
                if sc > 0 and s > sc * 1.05:
                    return {
                        "intent": "GOVERN_SPEED",
                        "directive": f"OVER SPEED LIMIT (Current: {s:.1f} m/s > Ceiling: {sc:.1f} m/s). Select decelerating trajectory to match limit.",
                        "target_max_speed": sc,
                    }
            except (ValueError, TypeError):
                pass

        # 3. Pedestrian Detection (Jaywalking / Crosswalk)
        ped = state.get("pedestrian")
        if isinstance(ped, dict):
            p_dist = ped.get("distance_m", 100.0)
            if p_dist is not None and p_dist <= 35.0:
                return {
                    "intent": "YIELD_PEDESTRIAN",
                    "directive": f"PEDESTRIAN DETECTED at {p_dist:.1f}m ahead. Yield immediately and bring vehicle to full stop to prevent casualty.",
                    "target_stop": True,
                    "critical_safety": True,
                }

        # 4. Roadside Parked Hazard / Lane Blockage
        roadside = state.get("roadside_obstacle")
        if isinstance(roadside, dict):
            r_dist = roadside.get("distance_m", 100.0)
            if r_dist is not None and r_dist <= 40.0:
                return {
                    "intent": "AVOID_ROADSIDE_OBSTACLE",
                    "directive": f"ROADSIDE OBSTACLE ({roadside.get('type', 'hazard')}) at {r_dist:.1f}m. Choose lateral clearance trajectory to safely pass.",
                    "must_avoid_collision": True,
                }

        # 5. General Collision / Hazard Avoidance
        candidates = state.get("candidates", {})
        if isinstance(candidates, dict):
            has_collision = any(v and len(v) >= 5 and v[4] for v in candidates.values())
            has_safe = any(v and len(v) >= 5 and not v[4] for v in candidates.values())
            if has_collision and has_safe:
                return {
                    "intent": "HAZARD_AVOID",
                    "directive": "COLLISION DETECTED on forward path. Select safe alternative lateral lane or emergency brake.",
                    "must_avoid_collision": True,
                }

        return {
            "intent": "SAFE_CRUISE",
            "directive": "Clear path ahead. Minimize route error and maintain efficient cruising speed.",
        }

    def _score_neural_options(
        self,
        state: Any,
        instructions: str,
        options: List[Dict[str, str]],
        prior: Optional[List[float]],
    ) -> Dict[str, Any]:
        def _sanitize_finite_json(val: Any) -> Any:
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return "ANOMALY_CORRUPTED"
                return val
            if isinstance(val, dict):
                return {k: _sanitize_finite_json(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_sanitize_finite_json(v) for v in val]
            return val

        row = {
            "id": f"jev_{int(time.time() * 1000)}",
            "state": _sanitize_finite_json(state),
            "question": instructions,
            "options": options,
        }
        visual_prefix = self._visual_prefix_from_state(state)
        vis_prior = None
        if visual_prefix is not None:
            vis_prior = self._visual_null_prior(len(options), row)
        scored = score(
            self.model,
            self.tokenizer,
            row,
            {},
            sliced_head=True,
            prior_logits=vis_prior if vis_prior is not None else prior,
            graph_runner=None if visual_prefix is not None else self.graph_runner,
            visual_prefix=visual_prefix,
        )
        from semif_phase1.gating import compute_free_energy

        energy = compute_free_energy(scored["calibrated_logits"])
        probs = {opt["id"]: p for opt, p in zip(options, scored["probabilities"])}
        chosen_idx = max(range(len(scored["probabilities"])), key=scored["probabilities"].__getitem__)
        return {
            "choice": options[chosen_idx]["id"],
            "probabilities": probs,
            "input_tokens": scored.get("input_tokens", 150),
            "visual_prefix_tokens": scored.get("visual_prefix_tokens", 0),
            "vision_free_energy": energy,
        }

    def _mock_branch_choice(self, question: str, options: List[Dict[str, str]], evidence: Dict[str, Any]) -> str:
        ids = [opt["id"] for opt in options]
        if "must_stop" in ids:
            intersection = evidence.get("intersection") if isinstance(evidence.get("intersection"), dict) else {}
            signal = str(intersection.get("signal") or "").lower()
            entered = bool(intersection.get("already_entered"))
            stopped = bool(intersection.get("stop_completed"))
            ped = evidence.get("pedestrian") if isinstance(evidence.get("pedestrian"), dict) else None
            other = evidence.get("other_vehicle") if isinstance(evidence.get("other_vehicle"), dict) else None
            ped_near = bool(ped and ped.get("distance_m") is not None and ped["distance_m"] <= 40)
            unmarked = str(intersection.get("control") or "").lower() in {"unmarked", "none", ""}
            arriving = bool(other and other.get("arriving") and (other.get("distance_m") or 99) <= 30)
            if (signal in {"red", "yellow"} and not entered and not stopped) or ped_near or (unmarked and arriving):
                return "must_stop"
            return "keep_moving"
        if "go_around" in ids:
            for key in ("construction", "roadside_obstacle"):
                item = evidence.get(key)
                if isinstance(item, dict) and item.get("distance_m") is not None and item["distance_m"] <= 45:
                    return "go_around"
            emergency = evidence.get("emergency_vehicle")
            if isinstance(emergency, dict) and emergency.get("siren") and emergency.get("behind", True):
                return "go_around"
            return "stay_in_lane"
        if "slow" in ids:
            try:
                if float(evidence.get("speed_mps")) > float(evidence.get("speed_ceiling_mps")) * 1.05:
                    return "slow"
            except (TypeError, ValueError):
                pass
            return "cruise"
        return ids[0]

    def _rank_vector_ids(self, action_ids: List[str], candidates: Dict[str, Any], mode: str) -> str:
        best = action_ids[0]
        best_score = -1e18
        for opt_id in action_ids:
            vec = candidates.get(opt_id)
            if not vec or len(vec) < 6:
                score_val = 0.0
            else:
                speed, _steer, r_err, offroad, collision = vec[0], vec[1], vec[2], vec[3], vec[4]
                try:
                    speed_f = 0.0 if speed is None or (isinstance(speed, float) and math.isnan(speed)) else float(speed)
                except (TypeError, ValueError):
                    speed_f = 0.0
                if collision:
                    score_val = -1000.0
                elif offroad > 0.1:
                    score_val = -500.0
                elif mode == "heuristic":
                    score_val = 10.0 + speed_f * 2.0 - abs(r_err) * 5.0
                else:
                    score_val = speed_f * 10.0 - r_err * 2.0
            if score_val > best_score:
                best_score = score_val
                best = opt_id
        return best

    def _explain_action_tree(self, state: Dict[str, Any], candidates: Dict[str, Any]) -> Dict[str, Any]:
        """Walk the tree for a rationale only. Does not pick or delete trajectories."""
        all_ids = list(candidates.keys())
        if sensors_corrupt(state):
            return {
                "path": [{"node": "sensor", "choice": "corrupt", "skipped": False, "parser": True}],
                "leaf_set": all_ids,
                "prunes": False,
                "true_ood": True,
            }

        def score_branches(question: str, options: List[Dict[str, str]], evidence: Dict[str, Any]) -> str:
            if self.use_mock or self.model is None:
                return self._mock_branch_choice(question, options, evidence)
            scored = self._score_neural_options(evidence, question, options, self._prior_for(len(options)))
            return scored["choice"]

        def ignore_leaves(action_ids: List[str]) -> str:
            return action_ids[0] if action_ids else (all_ids[0] if all_ids else "t00")

        meta = state.get("candidate_meta") if isinstance(state, dict) else None
        tree = build_drive_tree(candidates, meta if isinstance(meta, dict) else None)
        walked = walk_action_tree(tree, state, score_branches, ignore_leaves)
        return {
            "path": walked.get("path") or [],
            "leaf_set": all_ids,
            "prunes": False,
            "true_ood": False,
            "explain_bucket": (walked.get("path") or [{}])[-1].get("choice") if walked.get("path") else None,
        }

    def _mock_probs(self, option_ids: List[str], choice: str) -> Dict[str, float]:
        probs = {oid: 0.05 for oid in option_ids}
        if option_ids:
            probs[choice] = max(0.80, 1.0 - 0.05 * (len(option_ids) - 1))
            norm = sum(probs.values())
            probs = {k: round(v / norm, 4) for k, v in probs.items()}
        return probs

    def classify_jev(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Jev classifier. Flat and SemIf share the same action ids. SemIf adds a maneuver class."""
        state = payload.get("state", {})
        questions = payload.get("questions", {})
        mode = payload.get("mode", "flat")
        if mode == "semif_hierarchical":
            # Retired as a JevPilot executor. Same full-set scoring as flat.
            mode = "flat"
        raw_mode = bool(payload.get("raw_mode") or (isinstance(state, dict) and state.get("raw_mode")))
        image = payload.get("image")
        if not image and isinstance(state, dict):
            image = state.get("image")
        if isinstance(image, str) and len(image) > 64:
            try:
                from semif_phase1.vision import get_vision_encoder

                vis = get_vision_encoder().infer_b64(image)
                state = dict(state) if isinstance(state, dict) else {}
                state["vision"] = vis
            except Exception as exc:
                logger.warning("vision encode skipped: %s", exc)
        answers: Dict[str, Any] = {}
        total_input_tokens = 0

        rule_tier1 = self._determine_tier1_maneuver(state) if isinstance(state, dict) else {
            "intent": "SAFE_CRUISE",
            "directive": "Standard cruise and lane tracking.",
        }

        tree_path: List[Any] = []
        tree_walk: Optional[Dict[str, Any]] = None
        true_ood = sensors_corrupt(state) if isinstance(state, dict) else False
        if mode == "flat":
            maneuver = None
            strategic_intent = "FLAT_SEMIF"
            strategic_directive = "Sliced-head SemIf over the shared action set. No classification tree."
        elif mode == "heuristic":
            maneuver = None
            strategic_intent = "GEOMETRIC_HEURISTIC"
            strategic_directive = "Pure geometric scoring over the shared action set."
        else:
            if not isinstance(state, dict):
                state = {}
            candidates_now = state.get("candidates") if isinstance(state.get("candidates"), dict) else {}
            tree_walk = self._explain_action_tree(state, candidates_now)
            tree_path = tree_walk.get("path") or []
            true_ood = bool(tree_walk.get("true_ood"))
            maneuver = tree_walk.get("explain_bucket")
            strategic_intent = "EXPLAIN_TREE"
            strategic_directive = "Tree is rationale only. Score every sampled trajectory."
            answers["maneuver"] = {
                "choice": str(maneuver or "explain"),
                "probabilities": self._mock_probs(["explain"], "explain"),
            }

        visual_meta: Dict[str, Any] = {}
        for q_key, q_data in questions.items():
            instructions = q_data.get("instructions", "Choose optimal driving option.")
            criteria = q_data.get("criteria", {})

            # Prepare options
            options = []
            if isinstance(criteria, dict):
                for opt_id, opt_desc in criteria.items():
                    desc = str(opt_desc) if opt_desc is not None else f"Path {opt_id}"
                    options.append({"id": str(opt_id), "description": desc})
            elif isinstance(criteria, list):
                for item in criteria:
                    options.append({"id": str(item), "description": f"Level {item}"})

            if not options:
                continue
            if q_key == "motion":
                ids = [o["id"] for o in options]
                pick = "drive" if "drive" in ids else ids[0]
                answers[q_key] = {
                    "choice": pick,
                    "probabilities": self._mock_probs(ids, pick),
                }
                continue

            if q_key == "maneuver":
                answers[q_key] = answers.get("maneuver") or {
                    "choice": maneuver,
                    "probabilities": self._mock_probs([o["id"] for o in options], str(maneuver or options[0]["id"])),
                }
                continue

            candidates = state.get("candidates", {}) if isinstance(state, dict) else {}

            if q_key == "vector" and true_ood and isinstance(candidates, dict) and candidates:
                def _spd(cid: str) -> float:
                    vec = candidates.get(cid) or [1e9]
                    try:
                        val = float(vec[0])
                        return val if math.isfinite(val) else 1e9
                    except (TypeError, ValueError, IndexError):
                        return 1e9
                slowest = min(candidates, key=_spd)
                answers[q_key] = {
                    "choice": slowest,
                    "probabilities": self._mock_probs(list(candidates), slowest),
                }
                continue

            n_opts = len(options)
            can_neural = (
                not self.use_mock
                and self.model is not None
                and mode != "heuristic"
                and 2 <= n_opts <= 16
            )
            if not can_neural:
                best_choice = options[0]["id"]
                ranked_candidates = []

                for opt in options:
                    cand_vec = candidates.get(opt["id"])
                    # candidates vector: [speed, steer, route_error, offroad_fraction, collision, stop_at_line]
                    if not cand_vec or len(cand_vec) < 6:
                        score_val = 0.0
                    else:
                        speed, steer, r_err, offroad, collision, stop_line = cand_vec[0], cand_vec[1], cand_vec[2], cand_vec[3], cand_vec[4], cand_vec[5]
                        try:
                            speed_f = 0.0 if speed is None or (isinstance(speed, float) and math.isnan(speed)) else float(speed)
                        except (TypeError, ValueError):
                            speed_f = 0.0
                        speed = speed_f
                        if collision:
                            score_val = -1000.0
                        elif offroad > 0.1:
                            score_val = -500.0
                        elif mode == "heuristic":
                            score_val = 10.0 + speed * 2.0 - abs(r_err) * 5.0
                        else:
                            # Flat and mock SemIf share this ranker. SemIf's class is not a second scorer.
                            score_val = speed * 10.0 - r_err * 2.0
                    ranked_candidates.append((opt["id"], score_val))

                ranked_candidates.sort(key=lambda x: x[1], reverse=True)
                best_choice = ranked_candidates[0][0]

                probs = {opt["id"]: 0.05 for opt in options}
                probs[best_choice] = max(0.80, 1.0 - 0.05 * (len(options) - 1))
                norm = sum(probs.values())
                probs = {k: round(v / norm, 4) for k, v in probs.items()}

                answers[q_key] = {"choice": best_choice, "probabilities": probs}
                total_input_tokens += 120
            else:
                use_prior = self._prior_for(len(options))
                final_instructions = instructions
                ego_x = None
                try:
                    if state.get("lateral_offset_m") is not None:
                        ego_x = float(state.get("lateral_offset_m"))
                except (TypeError, ValueError):
                    ego_x = None
                vis = state.get("vision") if isinstance(state.get("vision"), dict) else {}
                inter = state.get("intersection") if isinstance(state.get("intersection"), dict) else {}
                sig = str(vis.get("signal") or inter.get("signal") or "").lower() or None
                if sig != "red" and "red" in str(vis.get("event") or "").lower():
                    sig = "red"
                final_options = []
                for opt in options:
                    cand_vec = candidates.get(opt["id"]) if isinstance(candidates, dict) else None
                    if cand_vec and len(cand_vec) >= 6:
                        desc = vector_option_tag(cand_vec, ego_x=ego_x, signal=sig)
                    else:
                        desc = opt["description"]
                    final_options.append({"id": opt["id"], "description": desc})
                neural_vec = self._score_neural_options(
                    compact_jev_state(state),
                    final_instructions,
                    final_options,
                    use_prior,
                )
                total_input_tokens += neural_vec["input_tokens"]
                answers[q_key] = {
                    "choice": neural_vec["choice"],
                    "probabilities": neural_vec["probabilities"],
                }
                visual_meta = {
                    "visual_prefix_tokens": neural_vec.get("visual_prefix_tokens", 0),
                    "vision_free_energy": neural_vec.get("vision_free_energy"),
                }

        self.stats["total_decisions"] += 1

        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": 0,
            },
            "meta": {
                "maneuver": maneuver,
                "tree_path": tree_path,
                "leaf_set": list((state.get("candidates") or {}).keys()) if isinstance(state, dict) else (tree_walk or {}).get("leaf_set"),
                "tree_prunes": False,
                "tier1_maneuver": strategic_intent,
                "tier1_directive": strategic_directive,
                "reason": rule_tier1.get("intent") if mode == "semif_hierarchical" else None,
                "hierarchical": mode == "semif_hierarchical",
                "raw_mode": raw_mode,
                "true_ood": true_ood,
                "ood_fail_safe": true_ood,
                "bound_vector": False,
                "semif_leaf": mode in {"flat", "semif_hierarchical"},
                "seed": (state.get("seed") if isinstance(state, dict) else None),
                "visual_prefix_tokens": visual_meta.get("visual_prefix_tokens", 0),
                "vision_free_energy": visual_meta.get("vision_free_energy"),
            },
        }


# Initialize FastAPI app
app = FastAPI(
    title="SemIf Decision Server",
    description="Real-time sub-15ms semantic decision server for JevPilot Three.js autonomous driving simulator.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def jevpilot_no_cache(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/jevpilot") and (
        path.endswith((".js", ".css", ".html")) or path.endswith("/")
    ):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

engine: Optional[DecisionEngine] = None


def get_engine() -> DecisionEngine:
    global engine
    if engine is None:
        engine = DecisionEngine(use_mock=True)
    return engine


@app.get("/health")
async def health_check():
    eng = get_engine()
    avg_latency = (
        eng.stats["total_latency_ms"] / eng.stats["total_decisions"]
        if eng.stats["total_decisions"] > 0
        else 0.0
    )
    return {
        "status": "online",
        "model": eng.model_name,
        "device": eng.device,
        "mock_mode": eng.use_mock,
        "cuda_graph_enabled": eng.graph_runner is not None,
        "vision": True,
        "decisions_served": eng.stats["total_decisions"],
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(eng.stats["min_latency_ms"], 2) if eng.stats["min_latency_ms"] != float("inf") else 0.0,
        "max_latency_ms": round(eng.stats["max_latency_ms"], 2),
    }


@app.post("/v1/classifier")
@app.post("/v1/systemone")
async def classifier_endpoint(payload: Dict[str, Any]):
    return get_engine().classify_jev(payload)


@app.post("/v1/vision")
async def vision_endpoint(payload: Dict[str, Any]):
    image = payload.get("image") or payload.get("image_base64")
    if not isinstance(image, str) or len(image) < 64:
        return {"error": "image (data URL or base64) required"}
    from semif_phase1.vision import get_vision_encoder

    evidence = get_vision_encoder().infer_b64(image)
    return {"vision": evidence}


@app.post("/decide")
async def decide_endpoint(payload: Dict[str, Any]):
    mode = payload.get("mode", "semif")
    telemetry = payload.get("telemetry", payload)
    return get_engine().decide(telemetry, mode=mode)


@app.websocket("/stream-decide")
async def websocket_stream_decide(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected to /stream-decide")
    eng = get_engine()
    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                data = json.loads(text_data)
            except Exception:
                continue

            mode = data.get("mode", "semif")
            telemetry = data.get("telemetry", data)

            result = eng.decide(telemetry, mode=mode)
            result["server_timestamp"] = time.time()

            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /stream-decide")
    except Exception as e:
        logger.error(f"WebSocket exception: {e}")


# Mount static assets for Three.js demo
jevpilot_dir = REPO_ROOT / "demo" / "jevpilot"
if jevpilot_dir.exists():
    app.mount("/jevpilot", StaticFiles(directory=str(jevpilot_dir), html=True), name="jevpilot")


@app.get("/")
async def root():
    index_file = jevpilot_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>SemIf Decision Server Online</h1><p>Visit /health or /jevpilot/</p>")


def main():
    global engine
    parser = argparse.ArgumentParser(description="SemIf Live Decision Server for JevPilot")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model id")
    parser.add_argument("--device", default=None, help="Inference device: cuda, cpu, mps")
    parser.add_argument("--mock", action="store_true", help="Run with mock heuristic engine (zero GPU)")
    parser.add_argument("--no-graph", action="store_true", help="Disable CUDA Graphs")
    args = parser.parse_args()

    engine = DecisionEngine(
        model_name=args.model,
        device=args.device,
        use_mock=args.mock,
        enable_graph=not args.no_graph,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
