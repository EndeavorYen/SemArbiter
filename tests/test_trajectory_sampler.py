from semif_phase1.trajectory_sampler import sample_trajectories


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
