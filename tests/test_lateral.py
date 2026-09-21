"""Lane-center PD damping for #113. Straight-road envelope, not CUDA evidence."""

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
    STEER_SLEW,
    YAW_KD,
    apply_steer_command,
    closest_segment_offset,
    dampen_stanley_steer,
    lane_keep_maneuver,
    lane_keep_pursuit_offset,
    lateral_pd,
    signed_lane_offset,
)

OVERLAY_JS = Path("demo/jevpilot/semif-layer.js")
CONTROL_CONSTS = (
    "STEER_LIMIT",
    "LATERAL_KP",
    "LATERAL_KD",
    "LATERAL_PD_LIMIT",
    "DETOUR_STEER",
    "LANE_KEEP_OFFSET_M",
    "LOOKAHEAD_MIN_M",
    "LOOKAHEAD_MAX_M",
    "LOOKAHEAD_S",
    "YAW_KD",
    "STEER_SLEW",
)


def test_signed_lane_offset_is_right_positive_at_heading_zero():
    assert signed_lane_offset(0.25, -10.0, 0.0, -10.0, 0.0) == 0.25
    assert signed_lane_offset(-0.1, -10.0, 0.0, -10.0, 0.0) == -0.1


def test_closest_segment_offset_projects_onto_the_line():
    # Segment along -Z (heading 0). Point 0.3 m to the right, past the start.
    assert abs(closest_segment_offset(0.3, -5.0, 0.0, 0.0, 0.0, -10.0, 0.0) - 0.3) < 1e-9


def test_yaw_kd_does_not_cancel_a_centering_steer():
    """YAW_KD 0.25 fought the return-to-center yaw. Keep it light."""
    assert 0.05 <= YAW_KD <= 0.10
    u = dampen_stanley_steer(-0.12, yaw_rate=-0.4, prev_u=-0.12, dt=1.0 / 60.0)
    assert u < 0.0


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


def test_yaw_damp_opposes_oscillation():
    u = dampen_stanley_steer(0.2, yaw_rate=0.5, prev_u=0.2, dt=1.0 / 60.0)
    assert u < 0.2


def test_steer_slew_limits_a_step():
    u = dampen_stanley_steer(0.8, yaw_rate=0.0, prev_u=0.0, dt=1.0 / 60.0)
    assert u <= STEER_SLEW / 60.0 + 1e-9
    assert u > 0.0


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


def test_apply_steer_command_uses_offset():
    a = apply_steer_command(0.0, 0.0, 0.0, 0.0, None, 0.016)
    b = apply_steer_command(0.0, 0.3, 0.0, 0.0, None, 0.016)
    assert a == 0.0
    assert b < 0.0


def test_overlay_control_constants_match_python():
    js = OVERLAY_JS.read_text(encoding="utf-8")
    for name in CONTROL_CONSTS:
        match = re.search(rf"const {name} = ([0-9.]+);", js)
        assert match, name
        assert float(match.group(1)) == float(getattr(lat, name)), name


def test_overlay_live_steer_hook_calls_stacked_command():
    js = OVERLAY_JS.read_text(encoding="utf-8")
    assert "applyLateralPd" not in js
    assert "function applyLaneKeepReference" in js
    steer_fn = js.split("window.SEMIF_APPLY_STEER")[1].split("function applyLaneKeepReference")[0]
    assert "applySteerCommand(" in steer_fn
    assert "realtimeLaneOffsetM(" in steer_fn
    assert "laneOffsetM(sim) || 0" not in steer_fn
    keep_fn = js.split("function applyLaneKeepReference")[1].split("function applyRawMode")[0]
    assert "applySteerCommand(" not in keep_fn
    assert "lateralPd(" not in keep_fn
    assert "_steerEma" not in js
    assert "p.target = -4" not in js
    assert "lateral_offset_m = player.x" not in js
