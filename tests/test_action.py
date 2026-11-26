"""Verify zero residuals target the actual robot's configured home pose."""
import pytest


@pytest.mark.simulation
def test_zero_action_home_target(env):
    import torch

    env.reset(seed=3)
    env.step(torch.zeros(2, 29))
    data = env.scene["robot"].data
    torch.testing.assert_close(data.joint_pos_target, data.default_joint_pos)
