import pytest

from benchmarks.benchmark_jevpilot_hierarchical import JevPilot2Simulator, run_jevpilot2_episode
from demo.server import DecisionEngine
from semif_phase1.trajectory_sampler import compact_jev_state
from semif_phase1.vision import compact_vision, synthetic_vision, vision_from_scenario


def test_synthetic_vision_picks_red_when_dominant():
    vis = synthetic_vision(red=0.7, green=0.1, pedestrian=0.2)
    assert vis["backend"] == "synthetic"
    assert vis["signal"] == "red"
    assert vis["pedestrian"] == 0.2


def test_compact_state_keeps_vision_not_pixels():
    packed = compact_jev_state(
        {
            "speed_mps": 12.0,
            "vision": synthetic_vision(red=0.6, pedestrian=0.4),
            "image": "data:image/jpeg;base64,AAAA",
            "candidates": {"t00": [12, 0, 0, 0, False, False]},
        }
    )
    assert "candidates" not in packed
    assert "image" not in packed
    assert packed["vision"]["signal"] == "red"
    assert "backend" in packed["vision"]


def test_compact_vision_drops_unknown_keys():
    slim = compact_vision({"signal": "green", "noise": 1, "red": 0.1})
    assert slim == {"signal": "green", "red": 0.1}


def test_vision_from_scenario_red_and_person():
    assert vision_from_scenario("traffic_light_red")["signal"] == "red"
    assert vision_from_scenario("pedestrian_jaywalking")["pedestrian"] >= 0.8


def test_vision_does_not_change_sampled_ids():
    sim = JevPilot2Simulator("traffic_light_red", seed=42, raw_mode=True)
    obs = sim.get_observation()
    keys = set(obs["candidates"])
    obs2 = dict(obs)
    obs2["vision"] = vision_from_scenario("traffic_light_red")
    assert set(obs2["candidates"]) == keys
    packed = compact_jev_state(obs2)
    assert "candidates" not in packed
    assert packed["vision"]["signal"] == "red"


def test_render_red_light_has_red_pixels():
    pytest.importorskip("PIL")
    from semif_phase1.vision import render_scenario_frame

    img = render_scenario_frame("traffic_light_red")
    pixels = list(img.getdata())
    assert any(r > 180 and g < 80 and b < 80 for r, g, b in pixels)


def test_synthetic_vision_closed_loop_mock():
    engine = DecisionEngine(use_mock=True)
    off = run_jevpilot2_episode(engine, "flat", "traffic_light_red", 42, raw_mode=True, vision_mode="off")
    vis = run_jevpilot2_episode(engine, "flat", "traffic_light_red", 42, raw_mode=True, vision_mode="synthetic")
    assert vis["vision_mode"] == "synthetic"
    assert off["seed"] == vis["seed"] == 42
