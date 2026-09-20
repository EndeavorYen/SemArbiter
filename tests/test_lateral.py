"""Lane-center PD damping for #113. Straight-road envelope, not CUDA evidence."""

from semif_phase1.lateral import (
    LANE_KEEP_OFFSET_M,
    LATERAL_KD,
    LATERAL_KP,
    LATERAL_PD_LIMIT,
    lane_keep_pursuit_offset,
    lateral_pd,
)


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


def test_lane_keep_pursuit_zeros_the_065_sample_band():
    """Web sampler second band is ±0.65 m. Holding that offset is the weave."""
    assert LANE_KEEP_OFFSET_M >= 0.65
    assert lane_keep_pursuit_offset(0.1) == 0.0
    assert lane_keep_pursuit_offset(-0.65) == 0.0
    assert lane_keep_pursuit_offset(0.65) == 0.0


def test_lane_keep_pursuit_keeps_a_real_nudge():
    assert lane_keep_pursuit_offset(1.35) == 1.35
    assert lane_keep_pursuit_offset(-1.2) == -1.2
    assert lane_keep_pursuit_offset(None) is None


def test_open_loop_microsteer_exceeds_15cm():
    peak = _straight_peak(use_pd=False, seed=42)
    assert peak > 0.15


def test_pd_holds_straight_200m_within_15cm():
    for seed in (42, 100):
        peak = _straight_peak(use_pd=True, seed=seed)
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
            target = 0.55 if rng.random() < 0.55 else -0.45
            if use_center_ref:
                target = lane_keep_pursuit_offset(target)
            next_dec = t + 0.1
        e += (target - e) * min(1.0, 8.0 * dt)
        z += v * dt
        t += dt
        peak = max(peak, abs(e))
    return peak


def test_overlay_applies_pd_not_steer_ema():
    js = open("demo/jevpilot/semif-layer.js", encoding="utf-8").read()
    assert "applyLateralPd" in js
    assert "LANE_KEEP_OFFSET_M" in js
    assert str(LANE_KEEP_OFFSET_M) in js
    assert "lane_keep_pursuit_offset" in js or "src.lane_offset_m = 0" in js
    assert "LATERAL_KP" in js
    assert str(LATERAL_KP) in js
    assert str(LATERAL_KD) in js
    assert str(LATERAL_PD_LIMIT) in js
    assert "_steerEma" not in js
    assert "p.target = -4" not in js
    assert "lateral_offset_m = player.x" not in js
