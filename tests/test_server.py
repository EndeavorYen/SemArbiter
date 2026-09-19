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
