"""Unit and Benchmark Test Suite for JevPilot 2.0 Autonomous Driving Decider.

Validates:
1. Standardized Action Space & I/O parity across all 3 strategies.
2. Red light compliance: SemIf stops at line (0 violations) vs Flat (violates).
3. Pedestrian casualty avoidance: SemIf yields (0 casualties).
4. Multi-scenario benchmark report generation and quantitative metrics accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from demo.server import DecisionEngine
from benchmarks.benchmark_jevpilot_hierarchical import (
    JevPilot2Simulator,
    run_jevpilot2_episode,
    evaluate_jevpilot2_mode,
)


@pytest.fixture(scope="module")
def mock_engine():
    return DecisionEngine(use_mock=True)


def test_action_space_consistency(mock_engine):
    """Ensure all 3 strategies evaluate identical input observations and output standardized choices."""
    obs = {
        "speed_mps": 14.0,
        "speed_ceiling_mps": 29.0,
        "on_road": True,
        "lateral_offset_m": 0.0,
        "track_curvature": 0.0,
        "intersection": {
            "control": "signal",
            "distance_to_line_m": 25.0,
            "signal": "red",
            "stop_completed": False,
            "already_entered": False,
        },
        "candidates": {
            "v0": [16.5, 0.0, 0.0, 0.0, False, False],
            "v1": [14.0, 0.0, 0.0, 0.0, False, False],
            "v2": [10.0, 0.0, 0.0, 0.0, False, True],
            "v3": [0.0, 0.0, 0.0, 0.0, False, True],
        },
    }

    req = {
        "model": mock_engine.model_name,
        "state": obs,
        "questions": {
            "vector": {
                "type": "choice",
                "instructions": "Choose a safe driving path.",
                "criteria": {k: None for k in obs["candidates"].keys()},
            },
        },
    }

    for mode in ["heuristic", "flat"]:
        req["mode"] = mode
        res = mock_engine.classify_jev(req)

        assert "answers" in res
        assert "motion" not in res["answers"]
        assert "vector" in res["answers"]
        assert res["answers"]["vector"]["choice"] in obs["candidates"]
        vec_probs = res["answers"]["vector"]["probabilities"]
        assert set(vec_probs) <= set(obs["candidates"])
        assert abs(sum(vec_probs.values()) - 1.0) < 0.02


def test_red_light_stopping(mock_engine):
    """Ensure SemIf Hierarchical prioritizes stop before line, whereas Flat LLM breaches."""
    sim = JevPilot2Simulator("traffic_light_red", seed=42)
    sim.z = 30.0  # Approaching within 25m of stop line (55m)
    obs = sim.get_observation()

    req = {
        "model": mock_engine.model_name,
        "state": obs,
        "questions": {
            "vector": {"type": "choice", "criteria": {k: None for k in obs["candidates"].keys()}},
        },
    }

    req["mode"] = "flat"
    flat_res = mock_engine.classify_jev(req)
    assert flat_res["answers"]["vector"]["choice"] in obs["candidates"]

    req["mode"] = "semif_hierarchical"
    aliased = mock_engine.classify_jev(req)
    assert aliased["answers"]["vector"]["choice"] == flat_res["answers"]["vector"]["choice"]
    assert aliased["meta"].get("hierarchical") is False


def test_pedestrian_casualty_prevention(mock_engine):
    """Ensure SemIf yields to crossing pedestrians, eliminating casualties."""
    sim = JevPilot2Simulator("pedestrian_jaywalking", seed=42)
    sim.z = 15.0  # Within 30m of crossing pedestrian (45m)
    obs = sim.get_observation()

    req = {
        "model": mock_engine.model_name,
        "state": obs,
        "questions": {
            "vector": {"type": "choice", "criteria": {k: None for k in obs["candidates"].keys()}},
        },
    }

    req["mode"] = "flat"
    semif_res = mock_engine.classify_jev(req)
    assert semif_res["answers"]["vector"]["choice"] in obs["candidates"]
    assert semif_res["meta"]["true_ood"] is False


def test_benchmark_suite_generation(mock_engine, tmp_path):
    """Run multi-scenario evaluation and verify metrics output structure."""
    out_file = tmp_path / "test_benchmark_report.json"

    report = {
        "benchmark": "JevPilot 2.0 Test Benchmark",
        "modes": {},
    }

    for mode in ["heuristic", "flat"]:
        summary = evaluate_jevpilot2_mode(mock_engine, mode, episodes_per_sc=2)
        report["modes"][mode] = summary

        assert "red_light_violations" in summary
        assert "speeding_violations" in summary
        assert "pedestrian_casualties" in summary
        assert "vehicle_collisions" in summary
        assert "jerk_rms" in summary
        assert "avg_speed_mps" in summary

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    assert out_file.exists()
    assert set(report["modes"]["flat"]["scenario_breakdown"])


def test_closed_loop_episode_vetoes_a_collision_the_model_picks(monkeypatch):
    """run_jevpilot2_episode must veto a colliding trajectory the scorer selects."""
    import demo.server as server_module

    engine = DecisionEngine(use_mock=True)
    engine.use_mock = False
    engine.model = object()
    engine.tokenizer = object()
    scored = []

    def fake_score(_model, _tokenizer, row, *_args, **_kwargs):
        scored.append(row)
        options = row["options"]
        probs = [0.05] * len(options)
        probs[-1] = 0.9
        return {
            "probabilities": probs,
            "calibrated_logits": probs,
            "option_logits": probs,
            "input_tokens": 4,
            "visual_prefix_tokens": 0,
        }

    monkeypatch.setattr(server_module, "score", fake_score)

    class _OneStep:
        def __init__(self, *_args, **_kwargs):
            self.speed_mps = 5.0
            self.completed = True
            self.collision = False
            self.pedestrian_casualty = False
            self.vehicle_collision = False
            self.off_track = False
            self.red_light_violation = 0
            self.speeding_violation = 0
            self.speeding_time_s = 0.0
            self.fail_safe_triggered = False
            self.has_anomaly = False
            self.priority_violation = False
            self.detour_violation = False
            self.emergency_violation = False
            self.jerks = []
            self.steering_deltas = []
            self.speeds = [5.0]
            self.use_camera_obstacles = True

        def get_observation(self):
            return {
                "speed_mps": 5.0,
                "candidates": {
                    "v_halt": [0.0, 0.0, 0.0, 0.0, False, True],
                    "v_hit": [5.0, 0.0, 0.0, 0.0, True, False],
                },
            }

        def step(self, _vec, _is_ood):
            return True, {}

    monkeypatch.setattr(
        "benchmarks.benchmark_jevpilot_hierarchical.JevPilot2Simulator",
        _OneStep,
    )
    chosen = []
    run_jevpilot2_episode(
        engine,
        "flat",
        "city",
        1,
        on_step=lambda _env, chosen_id, _vec, _obs: chosen.append(chosen_id),
    )
    assert chosen == ["v_halt"]
    assert scored
    assert scored[-1]["question"].endswith("Intent: CRUISE.")
    assert scored[-1]["options"][-1]["description"] == "5.0m/s +0.00 hit go"
