"""Fast-loop lateral PD on top of a selected steer. Does not pick a trajectory."""

from __future__ import annotations

from semif_phase1.trajectory_sampler import STEER_LIMIT

LATERAL_KP = 0.45  # rad per meter of lane offset
LATERAL_KD = 0.15  # rad per m/s of offset rate
LATERAL_PD_LIMIT = 0.12  # max |correction| so a detour still wins
DETOUR_STEER = 0.28  # skip PD when the selected steer is already a lane change


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
