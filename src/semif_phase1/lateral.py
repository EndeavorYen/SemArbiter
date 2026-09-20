"""Fast-loop lateral PD on top of a selected steer. Does not pick a trajectory."""

from __future__ import annotations

from semif_phase1.trajectory_sampler import STEER_LIMIT

LATERAL_KP = 0.45  # rad per meter of lane offset
LATERAL_KD = 0.15  # rad per m/s of offset rate
LATERAL_PD_LIMIT = 0.12  # max |correction| so a detour still wins
DETOUR_STEER = 0.28  # skip PD when the selected steer is already a lane change
# Web sampler: t<14 ±0.1 m, t<30 ±0.65 m, else ±1.35 m. A() holds that offset.
LANE_KEEP_OFFSET_M = 1.4
LOOKAHEAD_MIN_M = 4.0
LOOKAHEAD_MAX_M = 8.0
LOOKAHEAD_S = 0.45  # seconds of path A() should look ahead (city scale)
YAW_KD = 0.25  # rad per (rad/s) of yaw, damps Stanley weave
STEER_SLEW = 0.9  # rad/s cap on the command into w()


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

    Lookahead is 4–8 m so Stanley regain is stiff enough in town.
    """
    keep = selected_offset_m is None or abs(float(selected_offset_m)) <= LANE_KEEP_OFFSET_M
    if not keep:
        return {"lane_offset_m": float(selected_offset_m), "lookahead_m": None}
    look = max(LOOKAHEAD_MIN_M, min(LOOKAHEAD_MAX_M, LOOKAHEAD_S * abs(float(speed_mps))))
    return {"lane_offset_m": 0.0, "lookahead_m": look}


def dampen_stanley_steer(
    u_selected: float,
    yaw_rate: float,
    prev_u: float | None,
    dt: float,
) -> float:
    """Oppose yaw and slew-limit A() output. Not an EMA on player.steer."""
    u = float(u_selected) - YAW_KD * float(yaw_rate)
    step = max(1e-3, float(dt))
    if prev_u is not None:
        max_du = STEER_SLEW * step
        du = u - float(prev_u)
        if du > max_du:
            u = float(prev_u) + max_du
        elif du < -max_du:
            u = float(prev_u) - max_du
    return max(-STEER_LIMIT, min(STEER_LIMIT, u))


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
