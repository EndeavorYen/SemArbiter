from semif_phase1.action_tree import sensors_corrupt, slice_is_empty, walk_action_tree
from jevpilot_vision.drive import DRIVE_TREE


def test_sensors_corrupt_ignores_null_anomaly():
    assert sensors_corrupt({"speed_mps": 16.0, "anomaly": None}) is False
    assert sensors_corrupt({"speed_mps": 16.0}) is False
    assert sensors_corrupt({"speed_mps": float("nan"), "anomaly": None}) is True
    assert sensors_corrupt({"speed_mps": 16.0, "anomaly": "SENSOR_PACKET_CORRUPTED_NAN"}) is True


def test_empty_slice_skips_halt_without_intersection():
    asked = []

    def branches(question, options, evidence):
        asked.append(options[0]["id"])
        ids = [opt["id"] for opt in options]
        if "slow" in ids:
            return "cruise"
        raise AssertionError(f"unexpected node {ids}")

    def leaves(ids):
        return ids[0]

    walked = walk_action_tree(
        DRIVE_TREE,
        {"speed_mps": 16.0, "speed_ceiling_mps": 29.0},
        branches,
        leaves,
    )
    assert walked["path"][0]["node"] == "halt"
    assert walked["path"][0]["skipped"] is True
    assert "must_stop" not in asked
    assert walked["choice"] in {"t00", "t01"}


def test_clean_completion_requires_no_incident():
    from benchmarks.driving_quality import is_clean

    assert is_clean({"completed": True, "collision": False, "off_track": False, "red_light_violation": False, "pedestrian_casualty": False})
    assert not is_clean({"completed": True, "red_light_violation": True})
    assert not is_clean({"completed": False})
