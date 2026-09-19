from semif_phase1.trajectory_sampler import compact_jev_state
from semif_phase1.vision import compact_vision, synthetic_vision


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
