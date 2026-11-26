"""Observation tensor contract verified in actual physics."""
import pytest


@pytest.mark.simulation
def test_shapes_and_finiteness(env):
    import torch

    obs, _ = env.reset(seed=2)
    assert obs["actor"].shape == (2, 99)
    assert obs["critic"].shape == (2, 109)
    for _ in range(8):
        obs, reward, *_ = env.step(torch.zeros(2, 29))
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(v).all() for v in obs.values())
    assert env.scene["feet_contact"].data.in_contact.any()
