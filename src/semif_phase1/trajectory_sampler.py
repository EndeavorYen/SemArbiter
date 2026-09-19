"""Shared geometric trajectory sampler.

The sampler does not read traffic-light semantics. A stop line is a meter mark.
Web 3D and the Python loop share this option contract; they do not share a world.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

VECTOR_COLUMNS = (
    "speed",
    "steer",
    "route_error",
    "offroad_fraction",
    "collision",
    "stop_at_line",
)
VECTOR_INSTRUCTIONS = "Choose a safe driving path."


STEERS = (-0.28, -0.14, 0.0, 0.14, 0.28)
SPEED_OFFSETS = (2.5, 0.0, -4.0)
ROLLOUT_STEPS = 40
DT = 0.05


@dataclass
class Sample:
    id: str
    speed: float
    steer: float
    route_error: float
    offroad: float
    collision: bool
    stop_at_line: bool
    end_speed: float
    end_x: float
    end_z: float
    description: str

    def as_vec(self) -> List[Any]:
        return [
            round(self.speed, 1),
            round(self.steer, 2),
            round(self.route_error, 2),
            round(self.offroad, 3),
            self.collision,
            self.stop_at_line,
        ]

    def meta(self) -> Dict[str, Any]:
        return {
            "end_speed": round(self.end_speed, 2),
            "end_x": round(self.end_x, 2),
            "end_z": round(self.end_z, 2),
            "steer": round(self.steer, 2),
            "stop_at_line": self.stop_at_line,
            "description": self.description,
        }


def _hits(px: float, pz: float, ox: float, oz: float, radius: float) -> bool:
    return abs(pz - oz) < radius and abs(px - ox) < radius * 0.85


def rollout(
    ego_x: float,
    ego_z: float,
    speed: float,
    target_speed: float,
    target_steer: float,
    curvature: float,
    stop_line_z: Optional[float],
    obstacles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    x, z, v = ego_x, ego_z, speed
    steer = 0.0
    offroad_steps = 0
    collision = False
    for _ in range(ROLLOUT_STEPS):
        accel = max(-12.0, min(6.0, (target_speed - v) * 4.0))
        v = max(0.0, v + accel * DT)
        steer += (target_steer - steer) * 6.0 * DT
        z += v * DT
        x += steer * v * DT * 2.0
        x -= curvature * v * DT * 1.5
        if abs(x) > 4.0:
            offroad_steps += 1
        for obj in obstacles:
            oz = obj.get("z")
            ox = obj.get("x", 0.0)
            if oz is None:
                continue
            rad = float(obj.get("radius", 2.4))
            if 0.0 <= (oz - z) < rad + 1.5 and _hits(x, z, ox, oz, rad):
                collision = True
    stop_at_line = False
    if stop_line_z is not None and (stop_line_z - ego_z) > 0.5:
        stop_at_line = v < 0.8 and z < stop_line_z - 0.3
    return {
        "end_speed": v,
        "end_x": x,
        "end_z": z,
        "offroad": offroad_steps / ROLLOUT_STEPS,
        "collision": collision,
        "stop_at_line": stop_at_line,
        "route_error": x,
    }


def sample_trajectories(
    *,
    ego_x: float,
    ego_z: float,
    speed: float,
    curvature: float = 0.0,
    stop_line_z: Optional[float] = None,
    obstacles: Optional[Sequence[Dict[str, Any]]] = None,
    seed: int = 0,
) -> Dict[str, Sample]:
    """Deterministic grid with a little seeded jitter. Signal color is not an input."""
    rng = random.Random(int(seed) ^ (int(ego_z * 10) << 3))
    obstacles = list(obstacles or [])
    samples: Dict[str, Sample] = {}
    index = 0
    combos: List[Tuple[float, float]] = []
    for steer in STEERS:
        for offset in SPEED_OFFSETS:
            jitter_s = (rng.random() - 0.5) * 0.02
            jitter_v = (rng.random() - 0.5) * 0.3
            combos.append((steer + jitter_s, max(0.0, min(30.0, speed + offset + jitter_v))))
    combos.append((0.0, 0.0))
    for steer, target_speed in combos:
        geom = rollout(
            ego_x, ego_z, speed, target_speed, steer, curvature, stop_line_z, obstacles
        )
        if geom["offroad"] > 0.55 and target_speed > 0.5:
            continue
        sid = f"t{index:02d}"
        index += 1
        desc = (
            f"target {target_speed:.1f} m/s, steer {steer:+.2f}, "
            f"end_speed {geom['end_speed']:.1f} m/s, end_x {geom['end_x']:.1f} m, "
            f"collision {geom['collision']}, halt_geom {geom['stop_at_line']}"
        )
        samples[sid] = Sample(
            id=sid,
            speed=target_speed,
            steer=steer,
            route_error=geom["route_error"],
            offroad=geom["offroad"],
            collision=geom["collision"],
            stop_at_line=geom["stop_at_line"],
            end_speed=geom["end_speed"],
            end_x=geom["end_x"],
            end_z=geom["end_z"],
            description=desc,
        )
        if index >= 16:
            break
    if len(samples) < 2:
        samples["t00"] = Sample("t00", speed, 0.0, ego_x, 0.0, False, False, speed, ego_x, ego_z, "hold")
        samples["t01"] = Sample("t01", 0.0, 0.0, ego_x, 0.0, False, True, 0.0, ego_x, ego_z, "stop")
    return samples


def partition_ids(candidates: Dict[str, List[Any]], meta: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[str], List[str]]:
    halt: List[str] = []
    lateral: List[str] = []
    lane: List[str] = []
    meta = meta or {}
    for cid, vec in candidates.items():
        info = meta.get(cid) if isinstance(meta.get(cid), dict) else {}
        end_speed = float(info.get("end_speed", vec[0] if vec else 0.0) or 0.0)
        steer = float(info.get("steer", vec[1] if vec and len(vec) > 1 else 0.0) or 0.0)
        stop_line = bool(info.get("stop_at_line", vec[5] if vec and len(vec) > 5 else False))
        if end_speed < 1.2 or stop_line:
            halt.append(cid)
        elif abs(steer) >= 0.12:
            lateral.append(cid)
        else:
            lane.append(cid)
    if not lane:
        lane = [cid for cid in candidates if cid not in halt]
    return halt, lateral, lane


def candidates_as_vecs(samples: Dict[str, Sample]) -> Dict[str, List[Any]]:
    return {sid: sample.as_vec() for sid, sample in samples.items()}


def candidates_meta(samples: Dict[str, Sample]) -> Dict[str, Any]:
    return {sid: sample.meta() for sid, sample in samples.items()}
