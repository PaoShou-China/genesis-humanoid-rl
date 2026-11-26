"""Check true terminal state and partial reset behavior."""

import pytest


@pytest.mark.simulation
def test_timeout_keeps_terminal_state_until_reset(env):
    import torch

    env.reset(seed=1)
    env.episode_length_buf[0] = env.max_episode_length - 1
    _, _, _, truncated, _ = env.step(torch.zeros(2, 29))
    assert truncated[0]
    assert env.episode_length_buf[0] == env.max_episode_length
    with pytest.raises(RuntimeError, match="must be reset"):
        env.step(torch.zeros(2, 29))
    env.reset(env_ids=torch.tensor([0]))
    assert env.episode_length_buf[0] == 0
    env.step(torch.zeros(2, 29))


def test_h2_nominal_configuration():
    from humanoid.environments import make_env_cfg

    cfg = make_env_cfg("H2", num_envs=2)
    assert "push_robot" not in cfg.events
    assert "randomize_friction" not in cfg.events
    assert cfg.commands["base_velocity"].ang_vel_z == (-0.5, 0.5)
