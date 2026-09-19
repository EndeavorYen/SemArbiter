"""Tests for SemIf Live Decision Server and JevPilot telemetry integration."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.server import DecisionEngine, app, DRIVING_ACTIONS, ACTION_IDS


@pytest.fixture(scope="module")
def mock_engine():
    import demo.server as server_module
    engine = DecisionEngine(use_mock=True)
    server_module.engine = engine
    return engine


def test_decision_engine_mock(mock_engine):
    telemetry = {
        "speed_kmh": 65.0,
        "lateral_offset_m": 0.0,
        "track_curvature": 0.0,
        "obstacle_distance_m": 100.0,
        "obstacle_lane": "none",
    }
    res = mock_engine.decide(telemetry, mode="semif")
    assert res["action"] in ACTION_IDS
    assert "probabilities" in res
    assert len(res["probabilities"]) == 5
    assert res["is_ood"] is False
    assert res["latency_ms"] >= 0


def test_decision_engine_sharp_curve(mock_engine):
    telemetry = {
        "speed_kmh": 70.0,
        "lateral_offset_m": -1.0,
        "track_curvature": 0.6,  # Sharp right turn
        "obstacle_distance_m": 100.0,
    }
    res = mock_engine.decide(telemetry, mode="semif")
    assert res["action"] == "steer_right"
    assert res["target_steering"] > 0


def test_decision_engine_obstacle_ahead(mock_engine):
    telemetry = {
        "speed_kmh": 80.0,
        "lateral_offset_m": 0.0,
        "track_curvature": 0.0,
        "obstacle_distance_m": 15.0,  # Close obstacle!
        "obstacle_lane": "ego",
    }
    res = mock_engine.decide(telemetry, mode="semif")
    assert res["action"] == "brake"
    assert res["target_throttle"] < 0


def test_decision_engine_ood_anomaly(mock_engine):
    telemetry = {
        "anomaly": "CORRUPTED_STREAM_BAD_PAYLOAD",
        "speed_kmh": float("nan"),
    }
    res = mock_engine.decide(telemetry, mode="semif")
    assert res["is_ood"] is True
    assert res["action"] == "brake"


def test_fastapi_endpoints(mock_engine):
    client = TestClient(app)
    
    # GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert data["status"] == "online"
    assert data["mock_mode"] is True

    # POST /decide
    post_resp = client.post("/decide", json={
        "mode": "semif",
        "telemetry": {
            "speed_kmh": 55.0,
            "lateral_offset_m": 0.0,
            "track_curvature": 0.0,
            "obstacle_distance_m": 120.0,
            "obstacle_lane": "none"
        }
    })
    assert post_resp.status_code == 200
    pdata = post_resp.json()
    assert pdata["action"] in ACTION_IDS


def test_websocket_stream_decide(mock_engine):
    client = TestClient(app)
    with client.websocket_connect("/stream-decide") as ws:
        ws.send_json({
            "mode": "semif",
            "telemetry": {
                "speed_kmh": 60.0,
                "lateral_offset_m": 0.1,
                "track_curvature": 0.0,
                "obstacle_distance_m": 90.0,
                "obstacle_lane": "none"
            }
        })
        resp = ws.receive_json()
        assert resp["action"] in ACTION_IDS
        assert "server_timestamp" in resp
        assert "probabilities" in resp


def test_v1_classifier_endpoint(mock_engine):
    client = TestClient(app)
    req_body = {
        "model": "SemIf/Qwen2.5-3B-Instruct",
        "state": {
            "speed_mps": 12.0,
            "on_road": True,
            "candidates": {
                "v0": [12.0, 0.0, 0.1, 0.0, False, False],
                "v1": [12.0, -0.2, 0.3, 0.0, True, False],  # has collision
            }
        },
        "questions": {
            "vector": {
                "type": "choice",
                "instructions": "Choose a safe driving path.",
                "criteria": {
                    "v0": "maintain lane safely",
                    "v1": "swerve left with collision risk"
                }
            }
        }
    }
    resp = client.post("/v1/classifier", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert "answers" in data
    assert "vector" in data["answers"]
    assert data["answers"]["vector"]["choice"] == "v0"
    assert "probabilities" in data["answers"]["vector"]
    assert "usage" in data
    assert "meta" in data
    assert data["meta"]["hierarchical"] is True


def test_v1_classifier_red_light_compliance(mock_engine):
    """Verify JevPilot 2.0 Tier 1 Strategic Maneuver correctly halts before red light."""
    client = TestClient(app)
    req_body = {
        "model": "SemIf/Qwen2.5-3B-Instruct",
        "state": {
            "speed_mps": 10.0,
            "on_road": True,
            "intersection": {
                "control": "traffic_light",
                "distance_to_line_m": 12.0,
                "signal": "red",
                "stop_completed": False,
                "already_entered": False,
            },
            "candidates": {
                "v_cruise": [10.0, 0.0, 0.0, 0.0, False, False],  # cruises straight through red light (violation)
                "v_stop": [0.0, 0.0, 0.0, 0.0, False, True],     # stops safely at line (compliant)
            }
        },
        "questions": {
            "vector": {
                "type": "choice",
                "instructions": "Select path.",
                "criteria": {
                    "v_cruise": "continue driving at speed",
                    "v_stop": "stop at stop line"
                }
            }
        }
    }
    resp = client.post("/v1/classifier", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["tier1_maneuver"] == "YIELD_RED_LIGHT"
    assert data["answers"]["vector"]["choice"] == "v_stop"


def test_v1_classifier_speed_ceiling_governor(mock_engine):
    """Verify JevPilot 2.0 Tier 1 Strategic Maneuver complies with road speed ceiling."""
    client = TestClient(app)
    req_body = {
        "model": "SemIf/Qwen2.5-3B-Instruct",
        "state": {
            "speed_mps": 25.0,  # speeding!
            "speed_ceiling_mps": 13.4,  # 30 mph city zone limit
            "on_road": True,
            "candidates": {
                "v_fast": [26.0, 0.0, 0.0, 0.0, False, False],  # over speed limit
                "v_govern": [13.0, 0.0, 0.0, 0.0, False, False], # compliant with speed ceiling
            }
        },
        "questions": {
            "vector": {
                "type": "choice",
                "instructions": "Select path.",
                "criteria": {
                    "v_fast": "fast trajectory",
                    "v_govern": "speed compliant trajectory"
                }
            }
        }
    }
    resp = client.post("/v1/classifier", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["tier1_maneuver"] == "GOVERN_SPEED"
    assert data["answers"]["vector"]["choice"] == "v_govern"

