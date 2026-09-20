import pytest

from benchmarks.benchmark_jevpilot_hierarchical import JevPilot2Simulator, run_jevpilot2_episode
from demo.server import DecisionEngine
from semif_phase1.trajectory_sampler import compact_jev_state
from semif_phase1.vision import (
    blobs_from_frame,
    camera_event,
    compact_vision,
    event_from_motion,
    frame_motion,
    render_scenario_frame,
    synthetic_vision,
    vision_from_scenario,
)


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
    assert "prepare to stop" in packed["vision"]["event"]
    assert "red" not in packed["vision"]
    assert "backend" not in packed["vision"]


def test_compact_vision_drops_unknown_keys():
    slim = compact_vision({"signal": "green", "noise": 1, "red": 0.1})
    assert slim == {"signal": "green", "red": 0.1}


def test_pool_patches_is_32_tokens():
    torch = pytest.importorskip("torch")
    from semif_phase1.visual_prefix import pool_patches

    hidden = torch.randn(1, 196, 768)
    pooled = pool_patches(hidden, 32)
    assert tuple(pooled.shape) == (1, 32, 768)
    same = pool_patches(pooled, 32)
    assert tuple(same.shape) == (1, 32, 768)


def test_pixel_blobs_grow_and_cut_in_without_world_coords():
    pytest.importorskip("PIL")

    class _Z:
        def __init__(self, z):
            self.z = z

    far = render_scenario_frame("cut_in_vehicle", _Z(8.0))
    near = render_scenario_frame("cut_in_vehicle", _Z(40.0))
    far_b = blobs_from_frame(far)
    near_b = blobs_from_frame(near)
    assert "vehicle" in far_b and "vehicle" in near_b
    assert near_b["vehicle"]["w"] > far_b["vehicle"]["w"]
    motion = frame_motion(far_b, near_b)
    text = event_from_motion(motion)
    assert "TTC" not in text
    assert "growing" in text or "closing" in text or "cutting" in text


def test_camera_event_uses_score_delta_not_world_ttc():
    first = camera_event({"signal": "unknown", "vehicle": 0.8})
    assert "appeared" in first or "rising" in first or "cut-in" in first
    assert "TTC" not in first
    stable = camera_event(
        {"signal": "unknown", "vehicle": 0.8},
        {"signal": "unknown", "vehicle": 0.75},
    )
    assert "vehicle visible" in stable
    red = camera_event({"signal": "red", "red": 0.9, "vehicle": 0.0})
    assert "red" in red
    assert "maintain lane" in camera_event({})


def test_compact_state_drops_backend_and_patches():
    packed = compact_jev_state(
        {
            "speed_mps": 12.0,
            "vision": {
                "backend": "google/siglip-base-patch16-224",
                "signal": "red",
                "prefix_tokens": 32,
                "patches": [[0.1] * 8] * 32,
            },
        }
    )
    assert packed["vision"]["signal"] == "red"
    assert "backend" not in packed["vision"]
    assert "prefix_tokens" not in packed["vision"]
    assert "patches" not in packed["vision"]


def test_uses_visual_prefix_opt_in(monkeypatch):
    from semif_phase1.visual_prefix import uses_visual_prefix

    monkeypatch.delenv("SEMIF_VISION_PREFIX", raising=False)
    assert uses_visual_prefix({"backend": "google/siglip-base-patch16-224"}) is False
    monkeypatch.setenv("SEMIF_VISION_PREFIX", "1")
    assert uses_visual_prefix({"backend": "google/siglip-base-patch16-224"}) is True
    assert uses_visual_prefix({"backend": "synthetic"}) is False
    assert uses_visual_prefix({"backend": "stub"}) is False
    assert uses_visual_prefix(None) is False


def test_sliced_logits_with_prefix_last_position():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from semif_phase1.visual_prefix import sliced_logits_with_prefix

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 8)
            self.lm_head = nn.Linear(8, 32, bias=False)

        def forward(self, inputs_embeds, attention_mask, use_cache=False, return_dict=True):
            hidden = inputs_embeds
            if return_dict:
                return type("O", (), {"last_hidden_state": hidden})()
            return (hidden,)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Tiny()
            self.lm_head = self.model.lm_head

    model = Wrap()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.long)
    prefix = torch.randn(1, 32, 8)
    logits = sliced_logits_with_prefix(model, ids, mask, prefix, [0, 1])
    assert logits.shape == (2,)


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
