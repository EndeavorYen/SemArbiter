"""IPM from schematic pixels. No world (x, z) in the camera path."""

import pytest

from semif_phase1.ipm import camera_obstacles_from_blobs, ground_uv_to_ego
from semif_phase1.trajectory_sampler import rollout, sample_trajectories
from semif_phase1.vision import blobs_from_frame, render_scenario_frame


def test_horizon_pixels_are_dropped():
    assert ground_uv_to_ego(112, 120) is None
    assert ground_uv_to_ego(112, 80) is None


def test_lower_pixel_is_closer_than_near_horizon():
    far = ground_uv_to_ego(112, 140)
    near = ground_uv_to_ego(112, 190)
    assert far is not None and near is not None
    assert near["rel_z"] < far["rel_z"]
    assert far["rel_z"] > 8.0


def test_right_of_center_is_positive_rel_x():
    ego = ground_uv_to_ego(150, 180)
    assert ego is not None
    assert ego["rel_x"] > 0


def test_cut_in_frame_yields_vehicle_without_world_xyz():
    pytest.importorskip("PIL")

    class _Z:
        z = 12.0

    blobs = blobs_from_frame(render_scenario_frame("cut_in_vehicle", _Z()))
    obs = camera_obstacles_from_blobs(blobs)
    assert any(o["kind"] == "vehicle" and o["rel_z"] > 0 for o in obs)
    assert all("x" not in o and "z" not in o for o in obs)


def test_empty_blobs_mean_no_camera_collision():
    samples = sample_trajectories(ego_x=0.0, ego_z=10.0, speed=16.0, obstacles=[], seed=1)
    assert all(s.collision is False for s in samples.values())


def test_rel_obstacle_can_mark_collision():
    geom = rollout(
        0.0,
        10.0,
        16.0,
        16.0,
        0.0,
        0.0,
        None,
        [{"kind": "vehicle", "rel_x": 0.0, "rel_z": 2.0}],
    )
    assert geom["collision"] is True
