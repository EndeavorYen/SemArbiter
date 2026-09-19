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
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

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
        self.prior_logits: List[float] = DEFAULT_5OPT_PRIOR
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
            logger.info(f"5-option null prior calibrated: {self.prior_logits}")
        except Exception as e:
            logger.warning(f"Failed to auto-compute null prior: {e}. Using precomputed prior: {self.prior_logits}")

        # CUDA Graph initialization if on CUDA
        if self.device == "cuda" and self.enable_graph:
            try:
                from semif_phase1.cuda_graph import BucketGraphRunner
                logger.info("Initializing BucketGraphRunner for shape-bucketed inference...")
                self.graph_runner = BucketGraphRunner(self.model, buckets=(256, 512), device="cuda", warmup_on_init=True)
                logger.info("CUDA Graph buckets captured successfully.")
            except Exception as e:
                logger.warning(f"CUDA Graph warmup skipped or failed: {e}. Falling back to dynamic sliced LM head.")
                self.graph_runner = None

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

        # 1. Traffic Light & Intersection Analysis
        intersection = state.get("intersection")
        if isinstance(intersection, dict):
            sig = str(intersection.get("signal", "")).lower()
            dist = intersection.get("distance_to_line_m")
            already_entered = bool(intersection.get("already_entered", False))
            stop_completed = bool(intersection.get("stop_completed", False))

            if sig in ("red", "yellow") and not already_entered and not stop_completed:
                if dist is None or dist <= 30.0:
                    return {
                        "intent": "YIELD_RED_LIGHT",
                        "directive": f"RED/YELLOW SIGNAL ({sig.upper()}) at {dist}m ahead. Bring vehicle to a smooth halt before stop line (stop_at_line=True). Do not enter intersection.",
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

    def classify_jev(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Jev-compatible structured classification request with selectable strategy mode (flat vs semif_hierarchical)."""
        state = payload.get("state", {})
        questions = payload.get("questions", {})
        mode = payload.get("mode", "semif_hierarchical")
        answers: Dict[str, Any] = {}
        total_input_tokens = 0

        # Tier 1 Strategic Maneuver analysis
        if mode == "flat":
            strategic_intent = "NONE (Flat 1-of-N)"
            strategic_directive = "Naive unconstrained flat action selection."
            tier1 = {"intent": strategic_intent, "directive": strategic_directive}
        elif mode == "heuristic":
            strategic_intent = "GEOMETRIC_HEURISTIC"
            strategic_directive = "Pure geometric road-boundary following (baseline)."
            tier1 = {"intent": strategic_intent, "directive": strategic_directive}
        else:
            tier1 = self._determine_tier1_maneuver(state)
            strategic_intent = tier1["intent"]
            strategic_directive = tier1["directive"]

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

            # Special case for motion (drive vs stop)
            if q_key == "motion":
                if mode == "semif_hierarchical" and strategic_intent in ("YIELD_RED_LIGHT", "YIELD_PEDESTRIAN"):
                    inter = state.get("intersection", {}) if isinstance(state, dict) else {}
                    dist = inter.get("distance_to_line_m", 100) if isinstance(inter, dict) else 100
                    ped = state.get("pedestrian", {}) if isinstance(state, dict) else {}
                    p_dist = ped.get("distance_m", 100) if isinstance(ped, dict) else 100
                    best_choice = "stop" if (dist is not None and dist < 2.5) or (p_dist is not None and p_dist < 8.0) else "drive"
                else:
                    best_choice = "drive" if any(o["id"] == "drive" for o in options) else options[0]["id"]
                probs = {opt["id"]: 0.05 for opt in options}
                probs[best_choice] = max(0.80, 1.0 - 0.05 * (len(options) - 1))
                norm = sum(probs.values())
                probs = {k: round(v / norm, 4) for k, v in probs.items()}
                answers[q_key] = {"choice": best_choice, "probabilities": probs}
                continue

            candidates = state.get("candidates", {}) if isinstance(state, dict) else {}

            if self.use_mock or self.model is None:
                best_choice = options[0]["id"]
                ranked_candidates = []

                for opt in options:
                    cand_vec = candidates.get(opt["id"])
                    # candidates vector: [speed, steer, route_error, offroad_fraction, collision, stop_at_line]
                    if not cand_vec or len(cand_vec) < 6:
                        score_val = 0.0
                    else:
                        speed, steer, r_err, offroad, collision, stop_line = cand_vec[0], cand_vec[1], cand_vec[2], cand_vec[3], cand_vec[4], cand_vec[5]
                        if collision:
                            score_val = -1000.0
                        elif offroad > 0.1:
                            score_val = -500.0
                        elif mode == "flat":
                            # Naive flat scoring: prioritizes speed, ignores red lights, speed limits & pedestrians
                            score_val = speed * 10.0 - r_err * 2.0
                        elif mode == "heuristic":
                            # Geometric lane tracking: balance route error and progress
                            score_val = 10.0 + speed * 2.0 - r_err * 5.0
                        elif strategic_intent == "YIELD_RED_LIGHT":
                            # Heavily prioritize stopping at line with 0 velocity
                            score_val = 100.0 if stop_line else (-200.0 - speed * 10)
                        elif strategic_intent == "YIELD_PEDESTRIAN":
                            # Emergency yield for pedestrian: choose 0 velocity / maximum braking
                            score_val = 150.0 if speed == 0.0 else (-300.0 - speed * 15.0)
                        elif strategic_intent == "AVOID_ROADSIDE_OBSTACLE":
                            # Favor trajectory that nudges away from roadside obstacle
                            score_val = 20.0 + speed * 1.5 - abs(steer) * 10.0 - r_err * 2.0
                        elif strategic_intent == "GOVERN_SPEED":
                            sc = tier1.get("target_max_speed", 20.0)
                            score_val = (50.0 - abs(speed - sc) * 5.0) - r_err * 2.0
                        else:
                            score_val = 10.0 + speed * 2.0 - r_err * 5.0
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
                # Neural SemIf
                if mode == "flat":
                    final_instructions = instructions
                    final_options = options
                    use_prior = None  # No prior calibration for flat baseline
                else:
                    # Hierarchical Neural SemIf: Condition instructions with Tier 1 Strategic Directive
                    final_instructions = f"STRATEGIC DIRECTIVE: {strategic_directive}\n{instructions}"
                    final_options = []
                    for opt in options:
                        cand_vec = candidates.get(opt["id"])
                        if cand_vec and len(cand_vec) >= 6:
                            sp, st, re, of, col, stp = cand_vec[:6]
                            tag = f"speed: {sp:.1f}m/s, steer: {st:+.2f}, collision: {col}, stop_at_line: {stp}"
                            desc = f"{opt['description']} [{tag}]"
                        else:
                            desc = opt["description"]
                        final_options.append({"id": opt["id"], "description": desc})
                    use_prior = self.prior_logits[:len(options)] if len(options) <= len(self.prior_logits) else None

                row = {
                    "id": f"jev_{int(time.time() * 1000)}_{q_key}",
                    "state": state,
                    "question": final_instructions,
                    "options": final_options,
                }
                scored = score(
                    self.model,
                    self.tokenizer,
                    row,
                    {},
                    sliced_head=True,
                    prior_logits=use_prior,
                    graph_runner=self.graph_runner,
                )
                total_input_tokens += scored.get("input_tokens", 150)
                prob_dict = {opt["id"]: p for opt, p in zip(options, scored["probabilities"])}
                chosen_idx = max(range(len(scored["probabilities"])), key=scored["probabilities"].__getitem__)
                best_choice = options[chosen_idx]["id"]

                answers[q_key] = {
                    "choice": best_choice,
                    "probabilities": prob_dict,
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
                "tier1_maneuver": strategic_intent,
                "tier1_directive": strategic_directive,
                "hierarchical": True,
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
        "decisions_served": eng.stats["total_decisions"],
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(eng.stats["min_latency_ms"], 2) if eng.stats["min_latency_ms"] != float("inf") else 0.0,
        "max_latency_ms": round(eng.stats["max_latency_ms"], 2),
    }


@app.post("/v1/classifier")
@app.post("/v1/systemone")
async def classifier_endpoint(payload: Dict[str, Any]):
    return get_engine().classify_jev(payload)


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
