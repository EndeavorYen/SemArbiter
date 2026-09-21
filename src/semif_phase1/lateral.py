"""Lane-keep reference and the one-dimensional plant's lateral PD.

The web bicycle does not use this PD. Its steer is pure pursuit from A().
"""

from __future__ import annotations

import math

from semif_phase1.trajectory_sampler import STEER_LIMIT

LATERAL_KP = 0.45  # rad per meter of lane offset
LATERAL_KD = 0.15  # rad per m/s of offset rate
LATERAL_PD_LIMIT = 0.20  # max |correction| so a detour still wins
DETOUR_STEER = 0.28  # skip PD when the selected steer is already a lane change
# Web sampler: t<14 ±0.1 m, t<30 ±0.65 m, else ±1.35 m. A() holds that offset.
LANE_KEEP_OFFSET_M = 1.4
LOOKAHEAD_MIN_M = 4.0
LOOKAHEAD_MAX_M = 8.0
LOOKAHEAD_S = 0.40  # 10 m/s → 4 m; 4.5 m cut the Interstate bend at 0.17 m


def signed_lane_offset(px: float, pz: float, qx: float, qz: float, heading: float) -> float:
    """Right-positive cross-track. Heading 0 is -Z, same as the Web bicycle."""
    return (float(px) - float(qx)) * math.cos(heading) + (float(pz) - float(qz)) * math.sin(heading)


def closest_segment_offset(
    px: float,
    pz: float,
    ax: float,
    az: float,
    bx: float,
    bz: float,
    heading: float | None = None,
) -> float:
    """Project (px,pz) onto segment ab, then signed lateral vs heading."""
    abx = float(bx) - float(ax)
    abz = float(bz) - float(az)
    length2 = abx * abx + abz * abz or 1.0
    t = ((float(px) - float(ax)) * abx + (float(pz) - float(az)) * abz) / length2
    t = max(0.0, min(1.0, t))
    qx = float(ax) + abx * t
    qz = float(az) + abz * t
    h = heading if heading is not None else math.atan2(abx, -abz)
    return signed_lane_offset(px, pz, qx, qz, h)


def lane_keep_pursuit_offset(selected_offset_m):
    """A() tracks this lateral target. Lane-keep snaps to the centerline."""
    if selected_offset_m is None:
        return None
    offset = float(selected_offset_m)
    if abs(offset) <= LANE_KEEP_OFFSET_M:
        return 0.0
    return offset


def lane_keep_maneuver(selected_offset_m, speed_mps: float) -> dict:
    """Centerline pursuit for lane-keep, including Web steer-only (null offset) leaves.

    Lookahead is 4–8 m. 10 m/s stays at 4 m so Interstate bends stay inside 0.15 m.
    """
    keep = selected_offset_m is None or abs(float(selected_offset_m)) <= LANE_KEEP_OFFSET_M
    if not keep:
        return {"lane_offset_m": float(selected_offset_m), "lookahead_m": None}
    look = max(LOOKAHEAD_MIN_M, min(LOOKAHEAD_MAX_M, LOOKAHEAD_S * abs(float(speed_mps))))
    return {"lane_offset_m": 0.0, "lookahead_m": look}


def lateral_pd(
    u_selected: float,
    offset_m: float,
    offset_dot: float,
    *,
    kp: float = LATERAL_KP,
    kd: float = LATERAL_KD,
    limit: float = LATERAL_PD_LIMIT,
) -> float:
    """u = u_selected - Kp * e - Kd * e_dot, clipped. Skip large selected steers."""
    u_selected = float(u_selected)
    if abs(u_selected) > DETOUR_STEER:
        return max(-STEER_LIMIT, min(STEER_LIMIT, u_selected))
    pd = kp * float(offset_m) + kd * float(offset_dot)
    if pd > limit:
        pd = limit
    elif pd < -limit:
        pd = -limit
    return max(-STEER_LIMIT, min(STEER_LIMIT, u_selected - pd))
