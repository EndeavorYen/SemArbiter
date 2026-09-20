"""#110 reverse-sample parity. Startup must not reverse; only index 1 may reverse when slow."""

import random
from pathlib import Path

from semif_phase1.trajectory_sampler import _speed_fraction, sample_trajectories

JEV = Path(__file__).resolve().parent.parent / "demo" / "jevpilot"


def test_python_index_zero_is_stop_index_five_is_forward_cruise():
    rng = random.Random(0)
    assert _speed_fraction(0, rng, speed=0.0) == 0.0
    assert _speed_fraction(1, random.Random(0), speed=0.0) < 0
    assert _speed_fraction(1, random.Random(0), speed=16.0) >= 0
    cruise = _speed_fraction(5, random.Random(0), speed=0.0)
    assert cruise > 0.5


def test_startup_pool_is_forward_with_at_most_one_reverse():
    """TC-01 / TC-03: speed 0 must not flood the pool with reverse leaves."""
    samples = sample_trajectories(ego_x=0.0, ego_z=10.0, speed=0.0, seed=42)
    reverse_ids = [sid for sid, s in samples.items() if s.speed < 0]
    assert samples["t00"].speed >= 0
    assert len(reverse_ids) <= 1
    forward = [s for s in samples.values() if s.speed > 0]
    assert len(forward) >= 8


def test_slow_blocked_pool_still_has_one_reverse_leaf():
    """TC-04: when slow, exactly the index-1 reverse remains available."""
    samples = sample_trajectories(ego_x=0.0, ego_z=10.0, speed=1.0, seed=3)
    reverse_ids = [sid for sid, s in samples.items() if s.speed < 0]
    assert len(reverse_ids) == 1


def test_web_bundle_reverse_only_on_index_one():
    """TC-01/TC-03/TC-05 contract in the 3D planner mix (not t%5==0)."""
    main = (JEV / "assets" / "main-CvLEeHjW.js").read_text(encoding="utf-8")
    worker = (JEV / "assets" / "planner.worker-DFdG3q6n.js").read_text(encoding="utf-8")
    assert "(e.speed||0)<4&&t===1?-(1.6+r()*2)" in main
    assert "(t.speed||0)<4&&r===1?-(1.6+c()*2)" in worker
    assert "t%5==0?-(1.6" not in main
    assert "r%5==0?-(1.6" not in worker


def test_hud_seed_control_present_for_cross_seed_repro():
    """TC-05: seed Apply/Random exist so 42/100/999 can be reloaded without god-view."""
    js = (JEV / "semif-layer.js").read_text(encoding="utf-8")
    html = (JEV / "index.html").read_text(encoding="utf-8")
    assert "fsd-seed-apply" in js
    assert "reloadWithSeed" in js
    assert "?v=" in html
