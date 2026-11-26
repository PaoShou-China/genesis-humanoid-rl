"""Verify nominal H1 terms have physical meaning at the reset pose."""
import pytest


@pytest.mark.simulation
def test_standing_displacement_is_zero_at_reset(env):
    import torch

    from humanoid.rewards import joint_velocity, standing_displacement

    env.reset(seed=4)
    torch.testing.assert_close(standing_displacement(env), torch.zeros(2))
    assert (joint_velocity(env) >= 0).all()
