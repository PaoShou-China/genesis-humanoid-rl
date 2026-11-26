"""Synthetic unit fixtures are schema tests, never benchmark demonstrations."""

import numpy as np
import pytest

from scripts.validate_motion import validate_motion


def test_metadata_frequency_wins_over_filename(tmp_path):
    path = tmp_path / "fixture_80fps.npz"
    payload = {
        "fps": np.array(50),
        "joint_names": np.array(["joint"]),
        "body_names": np.array(["body"]),
        "joint_pos": np.zeros((2, 1)),
        "joint_vel": np.zeros((2, 1)),
        "body_pos_w": np.zeros((2, 1, 3)),
        "body_quat_w": np.array([[[1.0, 0, 0, 0]]] * 2),
        "body_lin_vel_w": np.zeros((2, 1, 3)),
        "body_ang_vel_w": np.zeros((2, 1, 3)),
    }
    np.savez(path, **payload)
    report = validate_motion(path)
    assert report["fps"] == 50
    assert report["duration_s"] == pytest.approx(0.02)
    payload["joint_pos"][0, 0] = np.nan
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="non-finite"):
        validate_motion(path)
