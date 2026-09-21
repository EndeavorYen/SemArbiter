"""Lane-center PD damping for #113. Straight-road envelope, not CUDA evidence."""

import math
import re
from pathlib import Path

from semif_phase1 import lateral as lat
from semif_phase1.lateral import (
    LANE_KEEP_OFFSET_M,
    LATERAL_KD,
    LATERAL_KP,
    LATERAL_PD_LIMIT,
    LOOKAHEAD_MAX_M,
    LOOKAHEAD_MIN_M,
    LOOKAHEAD_S,
    STEER_LIMIT,
    closest_segment_offset,
    lane_keep_maneuver,
    lane_keep_pursuit_offset,
    lateral_pd,
    signed_lane_offset,
)

OVERLAY_JS = Path("demo/jevpilot/semif-layer.js")
CONTROL_CONSTS = (
    "LANE_KEEP_OFFSET_M",
    "LOOKAHEAD_MIN_M",
    "LOOKAHEAD_MAX_M",
    "LOOKAHEAD_S",
)
WHEELBASE_M = 2.7
WHEEL_SLEW = 1.8


def test_signed_lane_offset_is_right_positive_at_heading_zero():
    assert signed_lane_offset(0.25, -10.0, 0.0, -10.0, 0.0) == 0.25
    assert signed_lane_offset(-0.1, -10.0, 0.0, -10.0, 0.0) == -0.1


def test_closest_segment_offset_projects_onto_the_line():
    # Segment along -Z (heading 0). Point 0.3 m to the right, past the start.
    assert abs(closest_segment_offset(0.3, -5.0, 0.0, 0.0, 0.0, -10.0, 0.0) - 0.3) < 1e-9


def test_pd_opposes_right_offset():
    u = lateral_pd(0.0, offset_m=0.2, offset_dot=0.4)
    assert u < 0.0
    assert abs(u) <= LATERAL_PD_LIMIT + 1e-9


def test_pd_keeps_selected_when_centered():
    u = lateral_pd(0.12, offset_m=0.0, offset_dot=0.0)
    assert abs(u - 0.12) < 1e-9


def test_pd_does_not_replace_a_detour_steer():
    u = lateral_pd(0.45, offset_m=0.3, offset_dot=0.2)
    assert abs(u - 0.45) < 1e-9


def test_lane_keep_pursuit_zeros_all_web_sample_bands():
    """Web bands are ±0.1 / ±0.65 / ±1.35 m. Any of those held by A() weaves."""
    assert LANE_KEEP_OFFSET_M >= 1.35
    assert lane_keep_pursuit_offset(0.1) == 0.0
    assert lane_keep_pursuit_offset(-0.65) == 0.0
    assert lane_keep_pursuit_offset(1.35) == 0.0
    assert lane_keep_pursuit_offset(-1.35) == 0.0


def test_lane_keep_pursuit_keeps_an_oversized_pullout():
    assert lane_keep_pursuit_offset(2.0) == 2.0
    assert lane_keep_pursuit_offset(-2.0) == -2.0
    assert lane_keep_pursuit_offset(None) is None


def test_lane_keep_maneuver_converts_steer_only_to_centerline():
    """t>=44 samples have null offset and fall back to discrete steer. That hunts."""
    m = lane_keep_maneuver(None, 16.0)
    assert m["lane_offset_m"] == 0.0
    assert LOOKAHEAD_MIN_M <= m["lookahead_m"] <= LOOKAHEAD_MAX_M
    assert m["lookahead_m"] == max(
        LOOKAHEAD_MIN_M, min(LOOKAHEAD_MAX_M, LOOKAHEAD_S * 16.0)
    )
    held = lane_keep_maneuver(1.35, 12.0)
    assert held["lane_offset_m"] == 0.0
    pull = lane_keep_maneuver(2.0, 12.0)
    assert pull["lane_offset_m"] == 2.0
    assert pull["lookahead_m"] is None


def test_open_loop_microsteer_exceeds_15cm():
    peak = _straight_peak(use_pd=False, seed=42)
    assert peak > 0.15


def test_pd_holds_straight_200m_within_15cm():
    for seed in (42, 100):
        peak = _straight_peak(use_pd=True, seed=seed)
        assert peak <= 0.15, f"seed {seed} peak {peak:.3f} m"


def test_live_closed_loop_cruise_stays_within_15cm():
    """#113 envelope on run_jevpilot2_episode, the step that calls lateral_pd."""
    from demo.server import DecisionEngine
    from benchmarks.diagnose_control_quality import collect_episode

    engine = DecisionEngine(use_mock=True)
    for seed in (42, 100):
        _episode, trace = collect_episode(
            engine,
            mode="heuristic",
            scenario="speed_zone_city",
            seed=seed,
            use_camera_obstacles=False,
        )
        assert trace, seed
        peak = max(abs(float(sample["x"])) for sample in trace)
        assert peak <= 0.15, f"seed {seed} peak {peak:.3f} m"


def test_web_offset_pursuit_without_center_ref_exceeds_15cm():
    peak = _web_pursuit_peak(use_center_ref=False, seed=42)
    assert peak > 0.15


def test_web_offset_pursuit_200m_stays_within_15cm():
    for seed in (42, 100):
        peak = _web_pursuit_peak(use_center_ref=True, seed=seed)
        assert peak <= 0.15, f"seed {seed} peak {peak:.3f} m"


def test_web_stanley_curve_stays_within_15cm():
    """Interstate seed 42: A() at 10 m/s with look 4.5 m peaked 0.17 m; 4.0 m held 0.13 m."""
    m = lane_keep_maneuver(0.0, 10.0)
    assert m["lookahead_m"] <= 4.0, f"10 m/s lookahead {m['lookahead_m']} m"


def _straight_peak(*, use_pd: bool, seed: int) -> float:
    """10 Hz discrete micro-steer, 60 Hz bicycle matching JevPilot2Simulator."""
    rng = __import__("random").Random(seed)
    x = 0.04
    z = 0.0
    v = 16.0
    steer = 0.02
    dt = 1.0 / 60.0
    t = 0.0
    next_dec = 0.0
    u_sel = 0.03
    peak = abs(x)
    while z < 200.0:
        if t >= next_dec:
            u_sel = 0.03 if rng.random() < 0.55 else -0.02
            next_dec = t + 0.1
        e_dot = steer * v * 2.0
        u = lateral_pd(u_sel, x, e_dot) if use_pd else u_sel
        steer += (u - steer) * 6.0 * dt
        x += steer * v * dt * 2.0
        z += v * dt
        t += dt
        peak = max(peak, abs(x))
    return peak


def _web_pursuit_peak(*, use_center_ref: bool, seed: int) -> float:
    """A() holds the selected lane_offset_m. That is the Web weave, not discrete steer."""
    rng = __import__("random").Random(seed)
    e = 0.04
    z = 0.0
    v = 16.0
    dt = 1.0 / 60.0
    t = 0.0
    next_dec = 0.0
    target = 0.55
    peak = abs(e)
    while z < 200.0:
        if t >= next_dec:
            target = 1.35 if rng.random() < 0.55 else -1.2
            if use_center_ref:
                target = lane_keep_pursuit_offset(target)
            next_dec = t + 0.1
        e += (target - e) * min(1.0, 8.0 * dt)
        z += v * dt
        t += dt
        peak = max(peak, abs(e))
    return peak



def test_overlay_control_constants_match_python():
    js = OVERLAY_JS.read_text(encoding="utf-8")
    for name in CONTROL_CONSTS:
        match = re.search(rf"const {name} = ([0-9.]+);", js)
        assert match, name
        assert float(match.group(1)) == float(getattr(lat, name)), name


def test_overlay_live_steer_hook_returns_pursuit():
    js = OVERLAY_JS.read_text(encoding="utf-8")
    assert "function applyLaneKeepReference" in js
    steer_fn = js.split("window.SEMIF_APPLY_STEER")[1].split("function laneKeepManeuver")[0]
    assert "return u" in steer_fn
    assert "applySteerCommand(" not in steer_fn
    assert "lateralPd(" not in steer_fn
    assert "realtimeLaneOffsetM(" not in js
    keep_fn = js.split("function applyLaneKeepReference")[1].split("function applyRawMode")[0]
    assert "laneKeepManeuver(" in keep_fn
    assert "applySteerCommand(" not in keep_fn
    assert "_steerEma" not in js
    assert "p.target = -4" not in js


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _speed_gain(speed):
    return 0.58 + 0.37 * (1.0 - _clip((abs(speed) - 3.0) / 5.0, 0.0, 1.0))


def _steer_from_curvature(curvature, speed):
    return _clip(math.atan(WHEELBASE_M * curvature) / _speed_gain(speed), -STEER_LIMIT, STEER_LIMIT)


def _straight_route(length=160.0, step=2.0):
    n = int(length / step) + 1
    return [(i * step, 0.0, -i * step, 0.0) for i in range(n)]


def _closest_s(x, z, pts):
    best_s, best_d = 0.0, 1e18
    for i in range(len(pts) - 1):
        s0, ax, az, _h = pts[i]
        s1, bx, bz, _h2 = pts[i + 1]
        abx, abz = bx - ax, bz - az
        length2 = abx * abx + abz * abz or 1.0
        t = _clip(((x - ax) * abx + (z - az) * abz) / length2, 0.0, 1.0)
        qx, qz = ax + abx * t, az + abz * t
        d2 = (x - qx) ** 2 + (z - qz) ** 2
        if d2 < best_d:
            best_d = d2
            best_s = s0 + (s1 - s0) * t
    return best_s


def _at_s(pts, s):
    if s <= pts[0][0]:
        return pts[0]
    if s >= pts[-1][0]:
        return pts[-1]
    for i in range(len(pts) - 1):
        s0, ax, az, h0 = pts[i]
        s1, bx, bz, _h1 = pts[i + 1]
        if s0 <= s <= s1:
            t = (s - s0) / ((s1 - s0) or 1.0)
            return (s, ax + (bx - ax) * t, az + (bz - az) * t, h0)
    return pts[-1]


def _pursuit(x, z, heading, speed, pts, lane_offset, lookahead):
    s = _closest_s(x, z, pts)
    _s, ax, az, ahead = _at_s(pts, s + lookahead)
    h = ahead + math.pi / 2.0
    ox = ax + math.sin(h) * lane_offset
    oz = az - math.cos(h) * lane_offset
    aim = math.atan2(ox - x, z - oz)
    err = math.atan2(math.sin(aim - heading), math.cos(aim - heading))
    dist = math.hypot(ox - x, oz - z)
    return _steer_from_curvature(2.0 * math.sin(err) / max(2.0, dist), speed)


def _retired_stack(u_a, offset, offset_dot, yaw_rate, prev_u, dt):
    """The layer that used to sit between A() and w(). Kp 0.45, yaw 0.08, slew 0.9."""
    u = lateral_pd(u_a, offset, offset_dot)
    u = u - 0.08 * yaw_rate
    step = max(0.001, dt)
    if prev_u is not None:
        max_du = 0.9 * step
        du = u - prev_u
        if du > max_du:
            u = prev_u + max_du
        elif du < -max_du:
            u = prev_u - max_du
    return max(-STEER_LIMIT, min(STEER_LIMIT, u))


def _band_crossings(samples):
    prev = 0
    count = 0
    for off in samples:
        side = 1 if off > 0.08 else -1 if off < -0.08 else 0
        if side and prev and side != prev:
            count += 1
        if side:
            prev = side
    return count


def _bicycle_offsets(x0, heading0, *, stack: bool):
    pts = _straight_route()
    keep = lane_keep_maneuver(0.0, 12.0)
    x, z, heading, speed, wheel = x0, 0.0, heading0, 12.0, 0.0
    dt = 1.0 / 60.0
    prev_u = None
    prev_off = None
    prev_h = None
    offs = []
    for _ in range(8 * 60):
        u = _pursuit(x, z, heading, speed, pts, keep["lane_offset_m"], keep["lookahead_m"])
        if stack:
            off_dot = 0.0 if prev_off is None else (x - prev_off) / dt
            yaw = 0.0 if prev_h is None else math.atan2(math.sin(heading - prev_h), math.cos(heading - prev_h)) / dt
            u = _retired_stack(u, x, off_dot, yaw, prev_u, dt)
            prev_off = x
            prev_h = heading
            prev_u = u
        speed += _clip(12.0 - speed, -8.0 * dt, 5.0 * dt)
        wheel += _clip(u - wheel, -WHEEL_SLEW * dt, WHEEL_SLEW * dt)
        yaw_gain = math.tan(wheel * _speed_gain(speed)) / WHEELBASE_M
        heading = math.atan2(math.sin(heading + speed * yaw_gain * dt), math.cos(heading + speed * yaw_gain * dt))
        x += math.sin(heading) * speed * dt
        z -= math.cos(heading) * speed * dt
        offs.append(x)
    return offs


def test_retired_steer_stack_weaves_on_the_web_bicycle():
    offs = _bicycle_offsets(0.60, 0.0, stack=True)
    assert _band_crossings(offs) > 0


def test_pure_pursuit_does_not_change_sides():
    for x0, heading0 in ((0.35, 0.05), (0.60, 0.0)):
        offs = _bicycle_offsets(x0, heading0, stack=False)
        assert _band_crossings(offs) == 0, (x0, heading0, _band_crossings(offs))
