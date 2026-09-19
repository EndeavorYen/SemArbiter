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
            "motion": {
                "type": "choice",
                "instructions": "Decide drive or stop.",
                "criteria": {"drive": None, "stop": None},
            },
            "vector": {
                "type": "choice",
                "instructions": "Choose a safe driving path.",
                "criteria": {k: None for k in obs["candidates"].keys()},
            },
        },
    }

    for mode in ["heuristic", "flat", "semif_hierarchical"]:
        req["mode"] = mode
        res = mock_engine.classify_jev(req)

        assert "answers" in res
        assert "motion" in res["answers"]
        assert "vector" in res["answers"]

        # Choice valid in candidate domain
        assert res["answers"]["motion"]["choice"] in ["drive", "stop"]
        assert res["answers"]["vector"]["choice"] in ["v0", "v1", "v2", "v3"]

        # Probabilities properly normalized
        vec_probs = res["answers"]["vector"]["probabilities"]
        assert len(vec_probs) == 4
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
            "motion": {"type": "choice", "criteria": {"drive": None, "stop": None}},
            "vector": {"type": "choice", "criteria": {k: None for k in obs["candidates"].keys()}},
        },
    }

    # Flat LLM
    req["mode"] = "flat"
    flat_res = mock_engine.classify_jev(req)
    # Flat chooses fastest vector (v0), ignoring red light
    assert flat_res["answers"]["vector"]["choice"] == "v0"

    # SemIf Hierarchical
    req["mode"] = "semif_hierarchical"
    semif_res = mock_engine.classify_jev(req)
    assert semif_res["meta"]["tier1_maneuver"] == "YIELD_RED_LIGHT"
    # SemIf chooses decelerate/stop vector (v2 or v3 with stop_at_line)
    chosen_id = semif_res["answers"]["vector"]["choice"]
    assert chosen_id in ["v2", "v3", "v5"]


def test_pedestrian_casualty_prevention(mock_engine):
    """Ensure SemIf yields to crossing pedestrians, eliminating casualties."""
    sim = JevPilot2Simulator("pedestrian_jaywalking", seed=42)
    sim.z = 15.0  # Within 30m of crossing pedestrian (45m)
    obs = sim.get_observation()

    req = {
        "model": mock_engine.model_name,
        "state": obs,
        "questions": {
            "motion": {"type": "choice", "criteria": {"drive": None, "stop": None}},
            "vector": {"type": "choice", "criteria": {k: None for k in obs["candidates"].keys()}},
        },
    }

    # SemIf Hierarchical detection
    req["mode"] = "semif_hierarchical"
    semif_res = mock_engine.classify_jev(req)
    assert semif_res["meta"]["tier1_maneuver"] == "YIELD_PEDESTRIAN"


def test_benchmark_suite_generation(mock_engine, tmp_path):
    """Run multi-scenario evaluation and verify metrics output structure."""
    out_file = tmp_path / "test_benchmark_report.json"

    report = {
        "benchmark": "JevPilot 2.0 Test Benchmark",
        "modes": {},
    }

    for mode in ["heuristic", "flat", "semif_hierarchical"]:
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
    assert report["modes"]["semif_hierarchical"]["red_light_violations"] == 0
