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


def residual_position(env):
    """Requested residual squared norm in rad², before shield/target clipping."""
    projection = getattr(env, "last_projection", None)
    if projection is None or projection.residual_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (projection.residual_action * env.benchmark_spec["action_scale"]).square().sum(-1)


def residual_rate(env):
    """Squared requested residual slew in rad²/s²; reset clears previous state."""
    delta = env.last_residual_change * env.benchmark_spec["action_scale"] / env.step_dt
    return delta.square().sum(-1)
