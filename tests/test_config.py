"""Reject unsupported tasks and invalid command ranges before simulation."""

import pytest

from humanoid.config import task_config


def test_motion_tracking_requires_external_data():
    from humanoid.task_objectives import require_motion

    with pytest.raises(FileNotFoundError, match="actual motion NPZ"):
        require_motion(task_config("H5"))


def test_standing_commands_are_zero():
    assert all(pair == [0.0, 0.0] for pair in task_config("H1")["commands"].values())
