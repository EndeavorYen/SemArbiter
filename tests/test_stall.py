from jevpilot_vision.stall import STALL_HOLD_S, stall_update


def test_moving_car_does_not_stall():
    st = stall_update(
        autopilot=True, speed_mps=8.0, route_s=10.0, dt=0.05, held_s=0.0, s0=10.0
    )
    assert st["trigger"] is False
    assert st["held_s"] == 0.0


def test_manual_mode_does_not_stall():
    st = stall_update(
        autopilot=False, speed_mps=0.0, route_s=4.0, dt=1.0, held_s=2.0, s0=4.0
    )
    assert st["trigger"] is False


def test_slow_but_progressing_does_not_stall():
    s0 = 0.0
    held = 0.0
    s = 0.0
    trig = False
    for _ in range(int(STALL_HOLD_S / 0.1) + 2):
        s += 0.3
        st = stall_update(
            autopilot=True, speed_mps=0.1, route_s=s, dt=0.1, held_s=held, s0=s0
        )
        held, s0, trig = st["held_s"], st["s0"], st["trigger"]
        if trig:
            break
    assert trig is False


def test_stopped_three_seconds_triggers_once_then_cooldown():
    st = {"held_s": 0.0, "s0": 12.0, "cooldown_s": 0.0, "trigger": False}
    hits = 0
    for _ in range(80):
        st = stall_update(
            autopilot=True,
            speed_mps=0.05,
            route_s=12.0,
            dt=0.1,
            held_s=st["held_s"],
            s0=st["s0"],
            cooldown_s=st["cooldown_s"],
        )
        if st["trigger"]:
            hits += 1
    assert hits == 1


def test_overlay_replan_is_ego_only_no_forced_reverse():
    js = open("jevpilot_vision/web/semif-layer.js", encoding="utf-8").read()
    assert "requestEgoReplan" in js
    assert "STALL_HOLD_S = 3.0" in js
    assert "p.target = -4" not in js
    assert "player.x" not in js or "lateral_offset_m = player.x" not in js
    assert "traffic" not in js.split("function requestEgoReplan")[1].split("function applyRawMode")[0]
