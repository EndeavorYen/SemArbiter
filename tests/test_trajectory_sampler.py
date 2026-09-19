import inspect

from semif_phase1.trajectory_sampler import (
    PLAN_POINTS,
    STEER_LIMIT,
    VECTOR_COLUMNS,
    VECTOR_INSTRUCTIONS,
    compact_jev_state,
    planning_max,
    sample_trajectories,
    vector_option_tag,
)


def test_vector_option_tag_is_short():
    csv = vector_option_tag([16.2, -0.14, 0.3, 0.0, False, True], style="csv")
    assert csv == "16.2,-0.14,0.00,0,1"
    words = vector_option_tag([16.2, -0.14, 0.3, 0.0, False, True], style="words")
    assert "collision=no" in words and "halt=yes" in words
    verbose = vector_option_tag([16.2, -0.14, 0.3, 0.0, False, True], style="verbose")
    assert "stop_at_line True" in verbose


def test_compact_state_drops_candidates():
    packed = compact_jev_state(
        {
            "speed_mps": 16.0,
            "intersection": {"signal": "red"},
            "candidates": {"t00": [16, 0, 0, 0, False, False]},
            "candidate_meta": {"t00": {"description": "long"}},
            "seed": 1,
        }
    )
    assert "candidates" not in packed
    assert "candidate_meta" not in packed
    assert packed["speed_mps"] == 16.0
    assert packed["intersection"]["signal"] == "red"


def test_option_contract_has_no_signal_and_no_stop_coaching():
    assert "signal" not in inspect.signature(sample_trajectories).parameters
    assert len(VECTOR_COLUMNS) == 6
    assert VECTOR_COLUMNS[-1] == "stop_at_line"
    assert "stop_at_line" not in VECTOR_INSTRUCTIONS
    assert "required" not in VECTOR_INSTRUCTIONS.lower()
    main = (__import__("pathlib").Path("demo/jevpilot/assets/main-CvLEeHjW.js").read_text(encoding="utf-8"))
    worker = (__import__("pathlib").Path("demo/jevpilot/assets/planner.worker-DFdG3q6n.js").read_text(encoding="utf-8"))
    assert "When a stop is required" not in main
    assert VECTOR_INSTRUCTIONS in main
    assert "l=!1&&O&&r<8" in worker


def test_sampler_is_deterministic():
    kwargs = dict(ego_x=0.0, ego_z=10.0, speed=16.0, curvature=0.0, stop_line_z=55.0, seed=42)
    a = sample_trajectories(**kwargs)
    b = sample_trajectories(**kwargs)
    assert list(a) == list(b)
    assert [s.as_vec() for s in a.values()] == [s.as_vec() for s in b.values()]
    assert len(a) >= 2


def test_sampler_ignores_signal_semantics():
    """Stop-line geometry is an input. Signal color is not."""
    red = sample_trajectories(ego_x=0.0, ego_z=10.0, speed=16.0, stop_line_z=55.0, seed=7)
    green = sample_trajectories(ego_x=0.0, ego_z=10.0, speed=16.0, stop_line_z=55.0, seed=7)
    assert [s.stop_at_line for s in red.values()] == [s.stop_at_line for s in green.values()]
    assert any(s.stop_at_line or s.end_speed < 1.2 for s in red.values())


def test_planner_policy_steer_horizon_and_ceiling():
    assert STEER_LIMIT == 0.85
    assert PLAN_POINTS == 31
    samples = sample_trajectories(ego_x=0.0, ego_z=10.0, speed=16.0, stop_line_z=55.0, seed=42)
    steers = [abs(s.steer) for s in samples.values()]
    assert max(steers) > 0.4
    assert all(abs(s.steer) <= STEER_LIMIT + 1e-6 for s in samples.values())
    capped = sample_trajectories(
        ego_x=0.0, ego_z=0.0, speed=16.0, speed_ceiling=8.0, seed=1
    )
    assert planning_max(16.0, 8.0) == 8.0
    assert max(s.speed for s in capped.values()) <= 8.0 + 1e-6


def test_collision_radius_matches_closed_loop_pedestrian():
    samples = sample_trajectories(
        ego_x=0.0,
        ego_z=0.0,
        speed=16.0,
        obstacles=[{"kind": "pedestrian", "x": 0.0, "z": 8.0}],
        seed=3,
    )
    assert any(s.collision for s in samples.values())


def test_closed_loop_obs_uses_sampled_ids(engine=None):
    from benchmarks.benchmark_jevpilot_hierarchical import JevPilot2Simulator
    from demo.server import DecisionEngine

    sim = JevPilot2Simulator("traffic_light_red", seed=42, raw_mode=True)
    obs = sim.get_observation()
    assert all(key.startswith("t") for key in obs["candidates"])
    assert "candidate_meta" in obs
    eng = DecisionEngine(use_mock=True)
    modes = []
    for mode in ("heuristic", "flat"):
        res = eng.classify_jev({
            "model": eng.model_name,
            "mode": mode,
            "state": obs,
            "questions": {"vector": {"type": "choice", "criteria": {k: None for k in obs["candidates"]}}},
        })
        modes.append(set(res["answers"]["vector"]["probabilities"]) | {res["answers"]["vector"]["choice"]})
        assert res["answers"]["vector"]["choice"] in obs["candidates"]
    assert modes[0] <= set(obs["candidates"])
    assert set(obs["candidates"]) == set(eng.classify_jev({
        "model": eng.model_name,
        "mode": "flat",
        "state": obs,
        "questions": {"vector": {"type": "choice", "criteria": {k: None for k in obs["candidates"]}}},
    })["answers"]["vector"]["probabilities"])
