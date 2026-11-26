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
