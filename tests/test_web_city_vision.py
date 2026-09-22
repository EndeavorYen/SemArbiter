import json

from benchmarks.web_city_vision import drive_lap, lap_url, score_lap, write_official_report


def _obs(*, complete, red=0, vehicles=0, pedestrians=0, crash=None, speeding=0, seed=42):
    return {
        "complete": complete,
        "red_light": red,
        "vehicle_collisions": vehicles,
        "pedestrian": pedestrians,
        "crash": crash,
        "speeding": speeding,
        "seed": seed,
        "autopilot": True,
        "time_s": 12.0,
    }


def test_lap_urls_use_existing_page_and_vision_switch():
    on = lap_url("http://127.0.0.1:8000", 42, True)
    off = lap_url("http://127.0.0.1:8000/", 42, False)
    assert on == "http://127.0.0.1:8000/jevpilot/?seed=42"
    assert off == "http://127.0.0.1:8000/jevpilot/?seed=42&vision=0"


def test_clean_is_complete_with_four_failures_at_zero():
    clean = score_lap(_obs(complete=True, speeding=4), seed=42, vision_on=True)
    assert clean["clean"] is True
    assert clean["speeding"] == 4
    assert clean["vision"] == "on"
    assert score_lap(_obs(complete=True, red=1), seed=42, vision_on=False)["clean"] is False
    assert score_lap(_obs(complete=False), seed=42, vision_on=True)["clean"] is False
    crashed = score_lap(_obs(complete=True, crash={"type": "car"}), seed=42, vision_on=True)
    assert crashed["clean"] is False


def test_official_report_is_written_only_when_both_laps_complete(tmp_path):
    path = tmp_path / "phase5-jevpilot-vision-cuda.json"
    incomplete = {
        "seed": 42,
        "device": "cuda",
        "mock": False,
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "world": "web-city-onboard-camera",
        "laps": [
            score_lap(_obs(complete=True), seed=42, vision_on=True),
            score_lap(_obs(complete=False), seed=42, vision_on=False),
        ],
    }
    assert write_official_report(path, incomplete) is False
    assert not path.exists()

    ready = dict(incomplete)
    ready["laps"] = [
        score_lap(_obs(complete=True, speeding=1), seed=42, vision_on=True),
        score_lap(_obs(complete=True), seed=42, vision_on=False),
    ]
    assert write_official_report(path, ready) is True
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["device"] == "cuda"
    assert body["mock"] is False
    assert body["model"] != "MockDecisionEngine"
    assert body["seed"] == 42
    assert body["world"] == "web-city-onboard-camera"
    assert [lap["vision"] for lap in body["laps"]] == ["on", "off"]
    assert all(lap["complete"] for lap in body["laps"])


def test_drive_lap_records_complete_or_timeout():
    steps = {"n": 0}

    def evaluate(_expression):
        steps["n"] += 1
        done = steps["n"] >= 2
        return {
            "ready": True,
            "complete": done,
            "red_light": 0,
            "vehicle_collisions": 0,
            "pedestrian": 0,
            "crash": None,
            "speeding": 2,
            "seed": 42,
            "autopilot": True,
            "time_s": steps["n"],
        }

    now = {"t": 0.0}

    def clock():
        return now["t"]

    def sleep(_seconds):
        now["t"] += 0.5

    finished = drive_lap(evaluate, seed=42, vision_on=True, timeout_s=10, sleep=sleep, clock=clock)
    assert finished["complete"] is True
    assert finished["clean"] is True
    assert finished["speeding"] == 2

    def stuck(_expression):
        return {
            "ready": True,
            "complete": False,
            "red_light": 1,
            "vehicle_collisions": 0,
            "pedestrian": 0,
            "crash": None,
            "speeding": 0,
            "seed": 42,
            "autopilot": True,
            "time_s": 1,
        }

    now["t"] = 0.0

    def jump(_seconds):
        now["t"] = 10.0

    unfinished = drive_lap(stuck, seed=42, vision_on=False, timeout_s=5, sleep=jump, clock=clock)
    assert unfinished["vision"] == "off"
    assert unfinished["complete"] is False
    assert unfinished["clean"] is False
    assert unfinished["red_light"] == 1


def test_mock_or_non_cuda_report_is_not_the_official_file(tmp_path):
    path = tmp_path / "phase5-jevpilot-vision-cuda.json"
    report = {
        "seed": 42,
        "device": "cpu",
        "mock": True,
        "model": "MockDecisionEngine",
        "world": "web-city-onboard-camera",
        "laps": [
            score_lap(_obs(complete=True), seed=42, vision_on=True),
            score_lap(_obs(complete=True), seed=42, vision_on=False),
        ],
    }
    assert write_official_report(path, report) is False
    assert not path.exists()
