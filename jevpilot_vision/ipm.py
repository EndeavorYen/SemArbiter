"""Ego-centric IPM from a ground-contact pixel. No world (x, z).

Toy camera is fitted to the 224×224 schematic: horizon at v=120.
rel_z is metres ahead of the bumper, rel_x is metres to the right.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

IMAGE_W = 224
IMAGE_H = 224
CAM_H_M = 1.4
CAM_F_PX = 280.0
CAM_U0 = IMAGE_W / 2.0
CAM_V0 = IMAGE_H / 2.0
HORIZON_V = 120.0
CAM_THETA = -math.atan((HORIZON_V - CAM_V0) / CAM_F_PX)

_KIND = {
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "construction": "roadside",
}


def ground_uv_to_ego(u: float, v: float) -> Optional[Dict[str, float]]:
    """Pinhole + flat ground. Pixels on or above the horizon are dropped."""
    if v <= HORIZON_V + 1.5:
        return None
    pitch = CAM_THETA + math.atan((v - CAM_V0) / CAM_F_PX)
    if abs(pitch) < 1e-4:
        return None
    rel_z = CAM_H_M / math.tan(pitch)
    if rel_z <= 0.3 or rel_z > 80.0:
        return None
    rel_x = rel_z * (u - CAM_U0) / CAM_F_PX
    return {"rel_x": float(rel_x), "rel_z": float(rel_z)}


def camera_obstacles_from_blobs(
    blobs: Dict[str, Dict[str, float]],
    *,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
) -> List[Dict[str, Any]]:
    """Bottom-center of each blob → ego (rel_x, rel_z). Lights are not obstacles."""
    out: List[Dict[str, Any]] = []
    for kind, box in blobs.items():
        mapped = _KIND.get(kind)
        if mapped is None:
            continue
        if float(box.get("w") or 0) > 0.45:
            continue
        u = float(box["cx"]) * image_w
        v = (float(box["cy"]) + float(box["h"]) / 2.0) * image_h
        ego = ground_uv_to_ego(u, v)
        if ego is None:
            continue
        out.append(
            {
                "kind": mapped,
                "rel_x": round(ego["rel_x"], 2),
                "rel_z": round(ego["rel_z"], 2),
            }
        )
    return out
