"""Tests for the SemArbiter decision door: evidence plus option ids in, an aligned choice out."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.server import DecisionEngine, app


@pytest.fixture(scope="module")
def mock_engine():
    import demo.server as server_module

    engine = DecisionEngine(use_mock=True)
    previous = server_module.engine
    server_module.engine = engine
    yield engine
    server_module.engine = previous


def _neural_engine(server_module):
    engine = DecisionEngine(use_mock=True)
    engine.use_mock = False
    engine.model = object()
    engine.tokenizer = object()
    previous = server_module.engine
    server_module.engine = engine
    return engine, previous


def _fake_score_last_option(seen):
    def fake_score(model, tokenizer, row, *args, **kwargs):
        seen.append(row)
        options = row["options"]
        probs = [0.05] * len(options)
        probs[-1] = 0.9
        return {
            "probabilities": probs,
            "calibrated_logits": probs,
            "option_logits": probs,
            "input_tokens": 4,
        }

    return fake_score


SIX_COLUMN_PAYLOAD = {
    "state": {
        "speed_mps": 8.0,
        "candidates": {
            "v_halt": [0.0, 0.0, 0.0, 0.0, False, True],
            "v_hit": [5.0, 0.0, 0.0, 0.0, True, False],
        },
    },
    "questions": {
        "vector": {
            "instructions": "Select path.",
            "criteria": {"v_halt": "halt", "v_hit": "hit"},
        }
    },
}


def test_health(mock_engine):
    data = TestClient(app).get("/health").json()
    assert data["status"] == "online"
    assert data["mock_mode"] is True


def test_root_is_the_decision_door():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "SemArbiter" in response.text
    assert "jevpilot" not in response.text.lower()


def test_door_serves_no_driving_routes(mock_engine):
    paths = {getattr(route, "path", None) for route in app.routes}
    assert {"/v1/classifier", "/v1/systemone", "/health"} <= paths
    for gone in ("/decide", "/stream-decide", "/v1/vision", "/jevpilot"):
        assert gone not in paths, gone
    client = TestClient(app)
    assert client.post("/decide", json={"telemetry": {}}).status_code == 404
    assert client.post("/v1/vision", json={"image": "A" * 80}).status_code == 404
    assert client.get("/jevpilot/").status_code == 404


def test_live_routes_score_supplied_options(monkeypatch):
    """Evidence plus options, no JPEG. The answer is the scorer's argmax by option id."""
    import demo.server as server_module

    _engine, previous = _neural_engine(server_module)
    seen = []
    monkeypatch.setattr(server_module, "score", _fake_score_last_option(seen))
    payload = {
        "state": {"note": "refund window is open"},
        "questions": {
            "decision": {
                "instructions": "Is the refund approved?",
                "criteria": {"no": "Deny the refund", "yes": "Approve the refund"},
            }
        },
    }
    try:
        client = TestClient(app)
        for path in ("/v1/classifier", "/v1/systemone"):
            response = client.post(path, json=payload)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["answers"]["decision"]["choice"] == "yes"
            assert set(body["answers"]["decision"]["probabilities"]) == {"no", "yes"}
        scored = [row for row in seen if row.get("question") == "Is the refund approved?"]
        assert scored
        assert [opt["id"] for opt in scored[0]["options"]] == ["no", "yes"]
    finally:
        server_module.engine = previous


def test_six_column_state_is_scored_like_any_state(monkeypatch):
    """No driving veto or option rewriting on the door. The app owns both."""
    import demo.server as server_module

    _engine, previous = _neural_engine(server_module)
    seen = []
    monkeypatch.setattr(server_module, "score", _fake_score_last_option(seen))
    try:
        response = TestClient(app).post("/v1/classifier", json=SIX_COLUMN_PAYLOAD)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answers"]["vector"]["choice"] == "v_hit"
        assert "fail_safe" not in body["meta"]
        rows = [row for row in seen if row["id"] != "null_prior_anchor"]
        assert len(rows) == 1
        assert rows[0]["question"] == "Select path."
        assert rows[0]["options"] == [
            {"id": "v_halt", "description": "halt"},
            {"id": "v_hit", "description": "hit"},
        ]
        assert rows[0]["state"] == SIX_COLUMN_PAYLOAD["state"]
    finally:
        server_module.engine = previous


def test_mock_engine_does_not_rank_trajectory_columns(mock_engine):
    """Mock is a placeholder, not a geometric executor. It ignores candidate vectors."""
    payload = {
        "state": {
            "candidates": {
                "v_hit": [9.0, 0.0, 0.0, 0.0, True, False],
                "v_fast": [20.0, 0.0, 0.0, 0.0, False, False],
            }
        },
        "questions": {"vector": {"criteria": {"v_hit": "hit", "v_fast": "fast"}}},
    }
    body = TestClient(app).post("/v1/classifier", json=payload).json()
    assert body["answers"]["vector"]["choice"] == "v_hit"
    assert body["meta"]["mock"] is True


def test_v1_classifier_returns_classifier_ms(mock_engine):
    data = TestClient(app).post(
        "/v1/classifier",
        json={
            "state": {"note": "text"},
            "questions": {"decision": {"criteria": {"a": "Alpha", "b": "Beta"}}},
        },
    ).json()
    assert isinstance(data["classifier_ms"], float)
    assert data["classifier_ms"] >= 0.0
    assert data["classifier_ms"] == data["meta"]["classifier_ms"]


def test_classifier_routes_reject_image_bytes(mock_engine, monkeypatch):
    import demo.server as server_module

    def scored(*_args, **_kwargs):
        raise AssertionError("image request must not be scored")

    monkeypatch.setattr(server_module, "score", scored)
    image = "data:image/jpeg;base64," + ("A" * 80)
    questions = {"decision": {"instructions": "Pick one", "criteria": {"a": "Alpha", "b": "Beta"}}}
    client = TestClient(app)
    for path in ("/v1/classifier", "/v1/systemone"):
        for payload in (
            {"image": image, "state": {"note": "text"}, "questions": questions},
            {"state": {"note": "text", "image": image}, "questions": questions},
        ):
            response = client.post(path, json=payload)
            assert response.status_code == 422, response.text
            assert "answers" not in response.json()


def test_driving_app_lives_in_its_own_repo():
    assert not list((REPO_ROOT / "jevpilot_vision").rglob("*.py"))
    for folder in ("src", "demo", "benchmarks", "tests"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if path.name == "test_server.py":
                continue
            text = path.read_text(encoding="utf-8")
            assert "jevpilot_vision" not in text, path
            assert "JevPilot2Simulator" not in text, path
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/EndeavorYen/JevPilot-Vision" in readme
    assert "localhost:8000/jevpilot" not in readme
