"""Reject unsupported tasks and invalid command ranges before simulation."""
import pytest

from humanoid.config import task_config


def test_unsupported_motion_tracking_is_explicit():
    with pytest.raises(ValueError, match="not implemented"):
        task_config("H5")


def test_standing_commands_are_zero():
    assert all(pair == [0.0, 0.0] for pair in task_config("H1")["commands"].values())
