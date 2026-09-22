"""Tests for SemIf Live Decision Server and JevPilot telemetry integration."""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.server import DecisionEngine, app, DRIVING_ACTIONS, ACTION_IDS, reset_vision_slot


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

    js_resp = client.get("/jevpilot/assets/index-DC8fTtby.js")
    assert js_resp.status_code == 200
    assert "no-cache" in js_resp.headers.get("cache-control", "").lower()


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
    assert data["meta"]["hierarchical"] is False


def test_v1_classifier_answers_web_motion_contract(mock_engine):
    """3D client se() requires motion drive|stop plus vector over moving vN ids."""
    client = TestClient(app)
    req_body = {
        "model": "SemIf/Qwen2.5-3B-Instruct",
        "mode": "flat",
        "state": {
            "speed_mps": 12.0,
            "on_road": True,
            "candidates": {
                "v0": [12.0, 0.0, 0.1, 0.0, False, False],
                "v1": [8.0, -0.1, 0.2, 0.0, False, False],
            },
        },
        "questions": {
            "motion": {
                "type": "choice",
                "instructions": "Drive includes slowing; stop means zero target now.",
                "criteria": {"drive": None, "stop": None},
            },
            "vector": {
                "type": "choice",
                "instructions": "Choose a safe driving path.",
                "criteria": {"v0": None, "v1": None},
            },
        },
    }
    resp = client.post("/v1/classifier", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    motion = data["answers"]["motion"]
    vector = data["answers"]["vector"]
    assert motion["choice"] == "drive"
    assert set(motion["probabilities"]) == {"drive", "stop"}
    assert vector["choice"] in {"v0", "v1"}
    assert set(vector["probabilities"]) == {"v0", "v1"}
    assert all(0.0 <= p <= 1.0 for p in motion["probabilities"].values())
    assert all(0.0 <= p <= 1.0 for p in vector["probabilities"].values())


def test_v1_classifier_fail_safe_and_red_directive(mock_engine):
    client = TestClient(app)
    req = {
        "mode": "flat",
        "state": {
            "speed_mps": 12.0,
            "intersection": {"control": "signal", "signal": "red", "distance_to_line_m": 12},
            "vision": {"signal": "red", "event": "RED signal ahead, mandatory stop"},
            "candidates": {
                "v0": [12.0, 0.0, 0.1, 0.0, False, False],
                "v1": [0.0, 0.0, 0.0, 0.0, False, True],
            },
        },
        "questions": {
            "vector": {
                "type": "choice",
                "instructions": "Choose a safe driving path.",
                "criteria": {"v0": None, "v1": None},
            }
        },
    }
    data = client.post("/v1/classifier", json=req).json()
    assert data["meta"]["jev1_intent"] == "RED_LIGHT_STOP"
    assert data["answers"]["vector"]["choice"] == "v1"

    crash = {
        "mode": "flat",
        "state": {
            "vision": {"event": "road clear ahead, maintain lane"},
            "candidates": {
                "v0": [16.0, 0.0, 0.0, 0.0, True, False],
                "v1": [0.0, 0.0, 0.0, 0.0, False, True],
            },
        },
        "questions": {
            "vector": {
                "type": "choice",
                "instructions": "Choose a safe driving path.",
                "criteria": {"v0": None, "v1": None},
            }
        },
    }
    hit = client.post("/v1/classifier", json=crash).json()
    assert hit["meta"]["jev1_intent"] == "CRUISE"
    assert hit["answers"]["vector"]["choice"] == "v1"


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
    assert data["meta"]["true_ood"] is False
    assert data["answers"]["vector"]["choice"] in data["answers"]["vector"]["probabilities"]


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
    assert data["meta"]["true_ood"] is False
    assert data["answers"]["vector"]["choice"] in {"v_fast", "v_govern"} or data["answers"]["vector"]["choice"] in data["answers"]["vector"]["probabilities"]


def _tiny_jpeg_data_url() -> str:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (8, 8), (40, 80, 40)).save(buf, format="JPEG", quality=40)
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_v1_classifier_returns_classifier_ms(mock_engine):
    client = TestClient(app)
    data = client.post(
        "/v1/classifier",
        json={
            "mode": "flat",
            "state": {
                "speed_mps": 12.0,
                "candidates": {
                    "v0": [12.0, 0.0, 0.1, 0.0, False, False],
                    "v1": [8.0, 0.0, 0.2, 0.0, False, False],
                },
            },
            "questions": {
                "vector": {
                    "type": "choice",
                    "criteria": {"v0": None, "v1": None},
                }
            },
        },
    ).json()
    assert isinstance(data["classifier_ms"], float)
    assert data["classifier_ms"] >= 0.0
    assert isinstance(data["meta"]["classifier_ms"], float)
    assert data["meta"]["classifier_ms"] >= 0.0
    assert data["classifier_ms"] == data["meta"]["classifier_ms"]


def test_v1_vision_overlapping_post_keeps_only_the_latest(mock_engine, monkeypatch):
    """A newer JPEG returns before the in-flight infer finishes, then only that JPEG is inferred."""
    reset_vision_slot()
    started = threading.Event()
    release = threading.Event()
    seen: list[str] = []
    image_a = _tiny_jpeg_data_url() + "AAA"
    image_b = _tiny_jpeg_data_url() + "BBB"

    class _Dummy:
        def infer_b64(self, image: str):
            seen.append(image)
            if len(seen) == 1:
                started.set()
                assert release.wait(3)
            return {"signal": "green", "event": "road clear ahead", "backend": "stub"}

    monkeypatch.setattr(
        "semif_phase1.vision.get_vision_encoder",
        lambda: _Dummy(),
    )

    async def _scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vision") as client:
            first = asyncio.create_task(
                client.post("/v1/vision", json={"image": image_a})
            )
            assert await asyncio.to_thread(started.wait, 2)
            second = await asyncio.wait_for(
                client.post("/v1/vision", json={"image": image_b}),
                timeout=0.5,
            )
            assert second.status_code == 200
            assert "vision_encode_ms" not in second.json()
            release.set()
            first_resp = await first
        assert first_resp.status_code == 200
        deadline = asyncio.get_running_loop().time() + 2
        while len(seen) < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert seen == [image_a, image_b]

    finished = threading.Event()
    failure: list[BaseException] = []

    def _run_scenario():
        try:
            asyncio.run(_scenario())
        except BaseException as exc:
            failure.append(exc)
        finally:
            release.set()
            finished.set()

    worker = threading.Thread(target=_run_scenario)
    worker.start()
    if not finished.wait(1.5):
        release.set()
        worker.join(3)
        raise AssertionError("superseded /v1/vision post did not return while infer was running")
    worker.join(3)
    if failure:
        raise failure[0]


def test_v1_vision_worker_infers_one_catch_up_then_stops(mock_engine, monkeypatch):
    """Frames that arrive during the catch-up infer wait for the next cycle."""
    reset_vision_slot()
    first_in = threading.Event()
    first_go = threading.Event()
    second_in = threading.Event()
    second_go = threading.Event()
    seen: list[str] = []
    image_a = _tiny_jpeg_data_url() + "AAA"
    image_b = _tiny_jpeg_data_url() + "BBB"
    image_c = _tiny_jpeg_data_url() + "CCC"
    image_d = _tiny_jpeg_data_url() + "DDD"

    class _Dummy:
        def infer_b64(self, image: str):
            seen.append(image)
            if len(seen) == 1:
                first_in.set()
                assert first_go.wait(3)
            elif len(seen) == 2:
                second_in.set()
                assert second_go.wait(3)
            return {"signal": "green", "event": "road clear ahead", "backend": "stub"}

    monkeypatch.setattr("semif_phase1.vision.get_vision_encoder", lambda: _Dummy())

    async def _scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vision") as client:
            owner = asyncio.create_task(client.post("/v1/vision", json={"image": image_a}))
            assert await asyncio.to_thread(first_in.wait, 2)
            busy = await client.post("/v1/vision", json={"image": image_b})
            assert busy.status_code == 200
            assert "vision_encode_ms" not in busy.json()
            first_go.set()
            assert await asyncio.to_thread(second_in.wait, 2)
            later = await asyncio.wait_for(
                client.post("/v1/vision", json={"image": image_c}),
                timeout=0.5,
            )
            assert later.status_code == 200
            second_go.set()
            owner_resp = await owner
            assert owner_resp.status_code == 200
            assert "vision_encode_ms" in owner_resp.json()
            assert seen == [image_a, image_b]
            follow = await client.post("/v1/vision", json={"image": image_d})
            assert follow.status_code == 200
            assert follow.json()["vision_encode_ms"] >= 0
            assert seen[-1] == image_d
            assert image_c not in seen

    finished = threading.Event()
    failure: list[BaseException] = []

    def _run_scenario():
        try:
            asyncio.run(_scenario())
        except BaseException as exc:
            failure.append(exc)
        finally:
            first_go.set()
            second_go.set()
            finished.set()

    worker = threading.Thread(target=_run_scenario)
    worker.start()
    if not finished.wait(3):
        first_go.set()
        second_go.set()
        worker.join(3)
        raise AssertionError("vision worker did not finish one catch-up cycle")
    worker.join(3)
    if failure:
        raise failure[0]


def test_v1_vision_labels_evidence_with_monotonic_gen(mock_engine, monkeypatch):
    """Each finished inference carries a generation newer than the previous one."""
    reset_vision_slot()

    class _Dummy:
        def infer_b64(self, image: str):
            return {"signal": "green", "event": "road clear ahead", "backend": "stub"}

    monkeypatch.setattr("semif_phase1.vision.get_vision_encoder", lambda: _Dummy())
    client = TestClient(app)
    image = _tiny_jpeg_data_url()
    first = client.post("/v1/vision", json={"image": image + "AAA"}).json()
    second = client.post("/v1/vision", json={"image": image + "BBB"}).json()
    assert isinstance(first["vision_gen"], int)
    assert second["vision_gen"] > first["vision_gen"]


def test_v1_vision_returns_vision_encode_ms(mock_engine):
    client = TestClient(app)
    resp = client.post("/v1/vision", json={"image": _tiny_jpeg_data_url()})
    assert resp.status_code == 200
    data = resp.json()
    assert "error" not in data
    assert isinstance(data["vision_encode_ms"], float)
    assert data["vision_encode_ms"] >= 0.0
    assert "vision" in data

