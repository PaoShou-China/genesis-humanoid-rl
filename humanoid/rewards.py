"""Small benchmark-specific terms composed with upstream locomotion rewards."""

import torch


def standing_displacement(env) -> torch.Tensor:
    """Squared horizontal root displacement from the reset pose, in m²."""
    robot = env.scene["robot"]
    origin = robot.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    return (robot.data.root_pos_w[:, :2] - origin).square().sum(-1)


def joint_velocity(env) -> torch.Tensor:
    """Sum of squared joint velocities, in rad²/s²."""
    return env.scene["robot"].data.joint_vel.square().sum(-1)


def alive(env) -> torch.Tensor:
    """Per-second survival credit; configured independently of locomotion rewards."""
    return torch.ones(env.num_envs, device=env.device)


def fall(env) -> torch.Tensor:
    """Physical failure event; timeouts do not incur a fall penalty."""
    return env.termination_manager.terminated.float()


def joint_limits(env) -> torch.Tensor:
    """Squared distance outside the soft joint-position range."""
    data = env.scene["robot"].data
    limits = data.soft_joint_pos_limits
    return (
        (limits[..., 0] - data.joint_pos).clamp(min=0).square()
        + (data.joint_pos - limits[..., 1]).clamp(min=0).square()
    ).sum(-1)
