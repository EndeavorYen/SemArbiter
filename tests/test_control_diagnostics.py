"""#117 headless SIL control-quality observer.

Known-bug traces must trip the matching fingerprint. Clean traces must not.
The observer is the regression net; live CUDA scores are not required here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from semif_phase1.control_diagnostics import (
    DEADBAND_ABS_M,
    DEADBAND_DRIFT_MPS,
    DEADBAND_MIN_DWELL_S,
    DEADLOCK_HOLD_S,
    OFFSET_SIGMA_M,
    STALL_PROGRESS_M,
    STALL_SPEED_MPS,
    STEER_ZERO_CROSS_HZ,
    cluster_diagnoses,
    diagnose_trace,
    format_issue_body,
    format_issue_title,
    format_report,
    make_sample,
)

DT = 0.05


def _samples(
    *,
    n: int,
    offset=0.0,
    speed=12.0,
    steer=0.0,
    z0: float = 0.0,
    signal=None,
    chosen_speed=None,
    chosen_id: str = "t00",
    chosen_tags: str = "12.0m/s +0.00 hold clear go",
    menu=None,
):
    out = []
    for i in range(n):
        t = i * DT
        off = offset(t, i) if callable(offset) else float(offset)
        v = speed(t, i) if callable(speed) else float(speed)
        st = steer(t, i) if callable(steer) else float(steer)
        sig = signal(t, i) if callable(signal) else signal
        cs = chosen_speed(t, i) if callable(chosen_speed) else chosen_speed
        z = z0
        if i:
            prev = out[-1]
            z = prev["z"] + prev["speed"] * DT
        out.append(
            make_sample(
                t=t,
                x=off,
                z=z,
                speed=v,
                steer=st,
                lane_offset=off,
                chosen_id=chosen_id,
                chosen_tags=chosen_tags,
                signal=sig,
                chosen_speed=cs,
                menu=menu,
            )
        )
    return out


def test_smooth_cruise_is_clean():
    trace = _samples(
        n=200,
        offset=lambda t, _i: 0.03 * math.sin(0.2 * t),
        steer=lambda t, _i: 0.02 * math.sin(0.2 * t),
        speed=14.0,
    )
    d = diagnose_trace(trace)
    assert d.violations == []
    assert d.ok is True


def test_lane_weave_trips_lateral_wander():
    # σ(0.25 sin) ≈ 0.177 m > 0.15; steer 0.3 Hz → 0.6 Hz zero-cross > 0.4.
    trace = _samples(
        n=120,
        offset=lambda t, _i: 0.25 * math.sin(2 * math.pi * 0.3 * t),
        steer=lambda t, _i: 0.35 * math.sin(2 * math.pi * 0.3 * t),
        speed=14.0,
    )
    d = diagnose_trace(trace)
    kinds = [v.kind for v in d.violations]
    assert "wander" in kinds
    w = next(v for v in d.violations if v.kind == "wander")
    assert w.metrics["offset_sigma_m"] > OFFSET_SIGMA_M
    assert w.metrics["steer_zero_cross_hz"] > STEER_ZERO_CROSS_HZ


def test_one_way_detour_is_not_wander():
    """A single lateral move has high σ but almost no steer zero-cross."""

    def offset(t, _i):
        return min(1.2, t * 0.6)

    trace = _samples(n=120, offset=offset, steer=0.08, speed=12.0)
    d = diagnose_trace(trace)
    assert "wander" not in [v.kind for v in d.violations]


def test_startup_reverse_trips_gear_chatter():
    def chosen_speed(t, _i):
        return -1.4 if t < 1.0 else 8.0

    trace = _samples(n=80, speed=0.0, chosen_speed=chosen_speed)
    d = diagnose_trace(trace)
    assert "chatter" in [v.kind for v in d.violations]
    c = next(v for v in d.violations if v.kind == "chatter")
    assert c.metrics["startup_reverse"] is True


def test_forward_reverse_flip_flop_in_two_second_window():
    def chosen_speed(t, _i):
        # After the 3 s startup window: + − + inside 2 s → two sign flips.
        if t < 3.2:
            return 2.0
        if t < 3.7:
            return -1.5
        if t < 4.2:
            return 2.0
        return 8.0

    trace = _samples(n=120, speed=0.2, chosen_speed=chosen_speed)
    d = diagnose_trace(trace)
    assert "chatter" in [v.kind for v in d.violations]
    c = next(v for v in d.violations if v.kind == "chatter")
    assert c.metrics["sign_flips"] >= 2


def test_halted_zero_speed_is_not_gear_chatter():
    trace = _samples(n=80, speed=0.0, chosen_speed=0.0)
    d = diagnose_trace(trace)
    assert "chatter" not in [v.kind for v in d.violations]


def test_deadband_escape_when_center_dwell_is_too_short():
    def offset(t, _i):
        if t < 0.2:
            return 0.4
        if t < 0.2 + DEADBAND_MIN_DWELL_S * 0.4:
            return 0.02
        return 0.02 + (t - 0.35) * (DEADBAND_DRIFT_MPS + 0.15)

    trace = _samples(n=80, offset=offset, speed=12.0)
    d = diagnose_trace(trace)
    assert "deadband" in [v.kind for v in d.violations]
    v = next(v for v in d.violations if v.kind == "deadband")
    assert v.metrics["dwell_s"] < DEADBAND_MIN_DWELL_S
    assert v.metrics["drift_mps"] > DEADBAND_DRIFT_MPS
    assert v.metrics["center_abs_m"] == DEADBAND_ABS_M


def test_centered_hold_is_not_deadband_escape():
    trace = _samples(n=80, offset=0.03, speed=12.0, steer=0.0)
    d = diagnose_trace(trace)
    assert "deadband" not in [v.kind for v in d.violations]


def test_spawn_graze_of_center_band_is_not_deadband():
    def offset(t, _i):
        return 0.0 if t < 0.12 else 0.25

    trace = _samples(n=40, offset=offset, speed=12.0)
    d = diagnose_trace(trace)
    assert "deadband" not in [v.kind for v in d.violations]


def test_deadlock_stall_without_red_light():
    n = int(DEADLOCK_HOLD_S / DT) + 10
    trace = _samples(n=n, speed=0.05, z0=40.0, signal=None)
    # z must not advance more than STALL_PROGRESS_M
    for s in trace:
        s["z"] = 40.0
        s["speed"] = 0.05
    d = diagnose_trace(trace)
    assert "deadlock" in [v.kind for v in d.violations]
    v = next(v for v in d.violations if v.kind == "deadlock")
    assert v.metrics["held_s"] > DEADLOCK_HOLD_S
    assert v.metrics["progress_m"] < STALL_PROGRESS_M
    assert v.metrics["speed_mps"] < STALL_SPEED_MPS


def test_red_light_wait_is_not_deadlock():
    n = int(DEADLOCK_HOLD_S / DT) + 10
    trace = _samples(n=n, speed=0.0, z0=50.0, signal="red")
    for s in trace:
        s["z"] = 50.0
        s["speed"] = 0.0
        s["signal"] = "red"
    d = diagnose_trace(trace)
    assert "deadlock" not in [v.kind for v in d.violations]


def test_report_includes_three_second_slice_and_ascii_plot():
    trace = _samples(
        n=120,
        offset=lambda t, _i: 0.25 * math.sin(2 * math.pi * 0.3 * t),
        steer=lambda t, _i: 0.35 * math.sin(2 * math.pi * 0.3 * t),
        speed=14.0,
    )
    d = diagnose_trace(trace, seed=42, scenario="speed_zone_city", mode="heuristic")
    text = format_report(d)
    assert "wander" in text
    assert "seed 42" in text.lower() or "Seed 42" in text
    assert "ascii" in text.lower() or any(ch in text for ch in "▁▂▃▄▅▆▇█-+|/\\")
    slice_t0 = d.violations[0].t0
    assert d.slice and d.slice[0]["t"] <= max(0.0, slice_t0 - 3.0) + 1e-6
    assert d.slice[-1]["t"] >= min(trace[-1]["t"], slice_t0 + 3.0) - 1e-6


def test_auto_issue_markdown_has_repro_and_thresholds():
    trace = _samples(n=40, speed=0.0, chosen_speed=lambda t, _i: -1.2 if t < 1.0 else 4.0)
    d = diagnose_trace(trace, seed=42, scenario="speed_zone_city", mode="heuristic")
    title = format_issue_title(d)
    body = format_issue_body(d)
    assert title.startswith("[Bug/Control]")
    assert "Seed 42" in title
    assert "python benchmarks/diagnose_control_quality.py --seed 42" in body
    assert "startup" in body.lower() or "reverse" in body.lower()
    assert str(OFFSET_SIGMA_M) in body or "0.15" in body


def test_cli_flags_and_default_seed_job():
    from benchmarks.diagnose_control_quality import _jobs, build_parser

    parser = build_parser()
    fast = parser.parse_args(["--fast"])
    cuda = parser.parse_args(["--cuda"])
    seed = parser.parse_args(["--seed", "42"])
    issues = parser.parse_args(["--file-issues"])
    assert fast.fast and cuda.cuda and issues.file_issues
    assert seed.seed == 42
    assert _jobs(seed) == [("speed_zone_city", 42)]
    fast_jobs = _jobs(parser.parse_args(["--fast"]))
    assert len(fast_jobs) == 100


def test_file_issue_uses_gh_and_skips_clean_traces():
    from benchmarks.diagnose_control_quality import file_issue

    clean = diagnose_trace(_samples(n=40, speed=12.0, offset=0.02), seed=42)
    assert file_issue(clean)["filed"] is False

    bad = diagnose_trace(
        _samples(n=40, speed=0.0, chosen_speed=lambda t, _i: -1.2 if t < 1.0 else 4.0),
        seed=42,
        scenario="speed_zone_city",
        mode="heuristic",
    )
    calls = []

    class Result:
        returncode = 0
        stdout = "https://github.com/EndeavorYen/SemIf/issues/0\n"
        stderr = ""

    def runner(cmd, **_kwargs):
        calls.append(cmd)
        return Result()

    out = file_issue(bad, runner=runner)
    assert out["filed"] is True
    assert calls and calls[0][0] == "gh"
    assert "issue" in calls[0] and "create" in calls[0]
    assert "--title" in calls[0]


def test_mock_episode_trace_has_required_fields():
    from demo.server import DecisionEngine
    from benchmarks.diagnose_control_quality import collect_episode, diagnose_trace

    engine = DecisionEngine(use_mock=True)
    episode, trace = collect_episode(
        engine, mode="heuristic", scenario="speed_zone_city", seed=42
    )
    assert trace and "t" in trace[0] and "lane_offset" in trace[0]
    assert "chosen_id" in trace[0] and "chosen_tags" in trace[0]
    d = diagnose_trace(trace, seed=42, scenario="speed_zone_city", mode="heuristic")
    assert episode["seed"] == 42
    assert d.n_samples == len(trace)
    assert "cand_min_offset" in trace[0]
    assert "cand_safe_count" in trace[0]
    assert "action_switch" in trace[0]


def _deadband_offset(t, _i):
    if t < 0.2:
        return 0.4
    if t < 0.4:
        return 0.02
    return 0.02 + (t - 0.35) * (DEADBAND_DRIFT_MPS + 0.15)


def test_sampler_defect_when_pool_has_no_center_leaf():
    menu = [
        {"id": "t00", "speed": 12.0, "offset": 0.40, "collision": False, "tag": "12.0m/s +0.20 diverge clear go"},
        {"id": "t01", "speed": 12.0, "offset": 0.55, "collision": False, "tag": "12.0m/s +0.30 diverge clear go"},
    ]
    trace = _samples(n=80, offset=_deadband_offset, speed=12.0, chosen_id="t00", menu=menu)
    d = diagnose_trace(trace, seed=42, scenario="speed_zone_city")
    v = next(v for v in d.violations if v.kind == "deadband")
    assert v.cause == "SAMPLER_DEFECT"
    assert v.actionable is True
    assert d.ok is False


def test_policy_defect_when_center_leaf_exists_but_was_not_picked():
    menu = [
        {"id": "t00", "speed": 16.0, "offset": 0.40, "collision": False, "tag": "16.0m/s +0.20 diverge clear go"},
        {"id": "t01", "speed": 12.0, "offset": 0.02, "collision": False, "tag": "12.0m/s -0.02 center clear go"},
    ]
    trace = _samples(
        n=80,
        offset=_deadband_offset,
        speed=12.0,
        chosen_id="t00",
        chosen_tags="16.0m/s +0.20 diverge clear go",
        menu=menu,
    )
    d = diagnose_trace(trace, seed=42, scenario="speed_zone_city")
    v = next(v for v in d.violations if v.kind == "deadband")
    assert v.cause == "POLICY_DEFECT"
    assert v.actionable is True
    assert d.frame and len(d.frame["top3"]) <= 3


def test_truncated_collision_is_not_ok():
    trace = _samples(n=20, offset=0.02, speed=14.0, steer=0.0)
    d = diagnose_trace(trace, seed=1, scenario="pedestrian_jaywalking", collision=True, t_end=1.0)
    causes = [v.cause for v in d.violations]
    assert "INVALID_COLLISION_TRUNCATED" in causes
    assert d.ok is False


def test_sensor_anomaly_reverse_is_expected_maneuver():
    menu = [
        {"id": "t00", "speed": 0.0, "offset": 0.0, "collision": False, "tag": "0.0m/s +0.00 hold clear halt"},
        {"id": "t01", "speed": -3.2, "offset": 0.0, "collision": False, "tag": "-3.2m/s +0.00 hold clear go"},
    ]
    trace = _samples(
        n=80,
        speed=0.0,
        chosen_speed=lambda t, _i: -3.2 if t >= 2.4 else 8.0,
        chosen_id="t01",
        menu=menu,
    )
    d = diagnose_trace(trace, seed=42, scenario="sensor_anomaly")
    c = next(v for v in d.violations if v.kind == "chatter")
    assert c.cause == "EXPECTED_SCENARIO_MANEUVER"
    assert c.actionable is False
    assert d.ok is True


def test_issue_body_prints_attribution_and_top3():
    menu = [
        {"id": "t00", "speed": 16.0, "offset": 0.40, "collision": False, "tag": "16.0m/s +0.20 diverge clear go"},
        {"id": "t01", "speed": 12.0, "offset": 0.02, "collision": False, "tag": "12.0m/s -0.02 center clear go"},
        {"id": "t02", "speed": 10.0, "offset": 0.08, "collision": False, "tag": "10.0m/s +0.05 hold clear go"},
    ]
    trace = _samples(n=80, offset=_deadband_offset, speed=12.0, chosen_id="t00", menu=menu)
    d = diagnose_trace(trace, seed=42, scenario="speed_zone_city", mode="heuristic")
    text = format_report(d)
    body = format_issue_body(d)
    assert "[POLICY_DEFECT]" in text
    assert "## Attribution" in body
    assert "POLICY_DEFECT" in body
    assert "t01" in body
    assert "Top-3" in body or "top3" in body.lower()


def test_file_issue_skips_expected_maneuver():
    from benchmarks.diagnose_control_quality import file_issue

    menu = [
        {"id": "t01", "speed": -3.2, "offset": 0.0, "collision": False, "tag": "-3.2m/s +0.00 hold clear go"},
    ]
    trace = _samples(
        n=80,
        speed=0.0,
        chosen_speed=lambda t, _i: -3.2 if t >= 2.4 else 8.0,
        menu=menu,
    )
    d = diagnose_trace(trace, seed=42, scenario="sensor_anomaly")
    out = file_issue(d)
    assert out["filed"] is False
    assert out["reason"] == "expected"


def test_cluster_merges_same_scenario_and_cause():
    menu = [
        {"id": "t00", "speed": 12.0, "offset": 0.40, "collision": False, "tag": "diverge"},
        {"id": "t01", "speed": 12.0, "offset": 0.02, "collision": False, "tag": "center"},
    ]

    def one(seed):
        return diagnose_trace(
            _samples(n=80, offset=_deadband_offset, speed=12.0, chosen_id="t00", menu=menu),
            seed=seed,
            scenario="speed_zone_city",
        )

    groups = cluster_diagnoses([one(42), one(43), one(44)])
    assert len(groups) == 1
    assert groups[0]["cause"] == "POLICY_DEFECT"
    assert groups[0]["seeds"] == [42, 43, 44]


def test_fast_path_uses_ground_truth_obstacles():
    from benchmarks.benchmark_jevpilot_hierarchical import JevPilot2Simulator

    sim = JevPilot2Simulator("pedestrian_jaywalking", seed=42)
    sim.use_camera_obstacles = False
    boxes = sim.ground_truth_obstacles()
    assert boxes and boxes[0]["kind"] == "pedestrian"
    assert "rel_z" in boxes[0] and boxes[0]["rel_z"] > 0
    sim.z = 42.0
    obs = sim.get_observation()
    assert any(bool(vec[4]) for vec in obs["candidates"].values())
