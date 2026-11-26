"""Configure GenesisLab without modifying upstream implementation files."""
from typing import Any

from humanoid.config import task_config


def make_env_cfg(
    task: str, *, num_envs: int = 16, seed: int = 0, evaluation: bool = False,
    config: dict[str, Any] | None = None,
):
    """Build H1/H2 using the audited 29-DoF G1 factory and nominal physics."""
    from genesislab.managers import RewardTermCfg
    from genesislab.tasks.velocity.config.g1.env_cfgs import g1_flat_env_cfg

    from humanoid import rewards

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    spec = config or task_config(task)
    if spec["task"] != task or spec["robot"] != "g1_29dof":
        raise ValueError("Unsupported task/robot combination")
    cfg = g1_flat_env_cfg(play=evaluation, num_envs=num_envs)
    cfg.seed = seed
    cfg.episode_length_s = spec["episode_length_s"]
    cfg.actions["joint_pos"].scale = spec["action_scale"]
    # Upstream flat training includes velocity kicks. Nominal PPO must not.
    cfg.events.pop("push_robot", None)
    command = cfg.commands["base_velocity"]
    for key, value in spec["commands"].items():
        if key not in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            raise ValueError(f"Unknown command field: {key}")
        setattr(command, key, tuple(value))
    if task == "H1":
        command.rel_standing_envs = 1.0
        cfg.rewards["standing_displacement"] = RewardTermCfg(
            func=rewards.standing_displacement, weight=-2.0,
        )
        cfg.rewards["joint_velocity"] = RewardTermCfg(func=rewards.joint_velocity, weight=-0.001)
    for key, weight in spec["rewards"].items():
        if key not in cfg.rewards:
            raise ValueError(f"Unknown reward term: {key}")
        cfg.rewards[key].weight = float(weight)
    if evaluation:
        cfg.auto_reset = False  # Keep terminal state available to evaluation.
        for group in cfg.observations.values():
            group.enable_corruption = False
    return cfg


def make_env(task: str, *, device: str = "cpu", video: bool = False,
             show_viewer: bool = False, **kwargs):
    """Instantiate actual Genesis physics; there is no mock simulator fallback."""
    from genesislab.envs import ManagerBasedRLEnv

    return ManagerBasedRLEnv(
        make_env_cfg(task, **kwargs), device=device, show_viewer=show_viewer,
        render_mode="rgb_array" if video else None,
    )
