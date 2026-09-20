from semif_phase1.directive import fail_safe_choice, plan_directive


def test_plan_directive_never_returns_a_trajectory_id():
    red = plan_directive({"signal": "red", "event": "RED signal ahead, mandatory stop"})
    assert red["intent"] == "RED_LIGHT_STOP"
    assert "t0" not in red["intent"] and "v0" not in red["directive"]
    cut = plan_directive({"event": "caution: vehicle cutting toward frame center, growing in the camera"})
    assert cut["intent"] == "YIELD_CUT_IN"
    cruise = plan_directive({})
    assert cruise["intent"] == "CRUISE"


def test_fail_safe_overrides_collision_even_on_cruise():
    cands = {
        "v0": [16.0, 0.0, 0.1, 0.0, True, False],
        "v1": [0.0, 0.0, 0.0, 0.0, False, True],
    }
    assert fail_safe_choice(cands, "v0", {"intent": "CRUISE"}) == "v1"


def test_red_light_stop_picks_halt_leaf():
    cands = {
        "v0": [12.0, 0.0, 0.1, 0.0, False, False],
        "v1": [0.2, 0.0, 0.0, 0.0, False, True],
        "v2": [8.0, -0.1, 0.2, 0.0, False, False],
    }
    assert fail_safe_choice(cands, "v0", {"intent": "RED_LIGHT_STOP"}) == "v1"


def test_red_light_without_halt_leaf_does_not_invent_one():
    cands = {
        "v0": [12.0, 0.0, 0.1, 0.0, False, False],
        "v1": [8.0, 0.0, 0.2, 0.0, False, False],
    }
    assert fail_safe_choice(cands, "v0", {"intent": "RED_LIGHT_STOP"}) == "v0"
