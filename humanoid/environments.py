"""Configure GenesisLab without modifying upstream implementation files."""

from typing import Any

from humanoid.config import task_config, validate_task


def make_env_cfg(
    task: str,
    *,
    num_envs: int = 16,
    seed: int = 0,
    evaluation: bool = False,
    config: dict[str, Any] | None = None,
):
    """Build H1/H2 using the audited 29-DoF G1 factory and nominal physics."""
    from genesislab.managers import RewardTermCfg, EventTermCfg, SceneEntityCfg
    from genesislab.tasks.velocity.config.g1.env_cfgs import g1_flat_env_cfg, g1_rough_env_cfg
    import genesislab.envs.mdp as mdp

    from humanoid import rewards

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    spec = validate_task(config, task) if config else task_config(task)
    factory = g1_rough_env_cfg if spec.get("terrain") == "rough" else g1_flat_env_cfg
    cfg = factory(play=evaluation, num_envs=num_envs)
    cfg.seed = seed
    cfg.episode_length_s = spec["episode_length_s"]
    cfg.actions["joint_pos"].scale = spec["action_scale"]
    fixed_gains = spec.get("pd_gains", {})
    for actuator in cfg.scene.entities["robot"].actuators:
        for field, key in (("stiffness", "stiffness_scale"), ("damping", "damping_scale")):
            value = getattr(actuator, field)
            scale = fixed_gains.get(key, 1.0)
            if value is not None:
                setattr(
                    actuator,
                    field,
                    {k: v * scale for k, v in value.items()}
                    if isinstance(value, dict)
                    else value * scale,
                )
    # Upstream flat training includes velocity kicks. Nominal PPO must not.
    cfg.events.pop("push_robot", None)
    cfg.events.pop("randomize_friction", None)
    cfg.events.pop("randomize_base_mass", None)
    if "reset_joint_range" in spec:
        cfg.events["reset_joints"].params["position_range"] = tuple(spec["reset_joint_range"])
    for group in cfg.observations.values():
        group.enable_corruption = spec.get("observation_noise", group.enable_corruption)
    command = cfg.commands["base_velocity"]
    for key, value in spec["commands"].items():
        if key not in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            raise ValueError(f"Unknown command field: {key}")
        setattr(command, key, tuple(value))
    if task in ("H1", "H3"):
        command.rel_standing_envs = 1.0
        cfg.rewards["standing_displacement"] = RewardTermCfg(
            func=rewards.standing_displacement,
            weight=-2.0,
        )
        cfg.rewards["joint_velocity"] = RewardTermCfg(func=rewards.joint_velocity, weight=-0.001)
    if spec.get("reward_profile") == "survival":
        cfg.rewards["alive"] = RewardTermCfg(func=rewards.alive, weight=2.0)
        cfg.rewards["fall"] = RewardTermCfg(func=rewards.fall, weight=-50.0)
    if spec.get("reward_profile") == "soft_safety":
        cfg.rewards["joint_limits"] = RewardTermCfg(func=rewards.joint_limits, weight=-2.0)
    dr = spec.get("randomization", {})
    if dr:
        cfg.sim.rigid.batch_links_info = True
        cfg.sim.rigid.batch_dofs_info = True
    if "friction_ratio" in dr:
        cfg.events["dr_friction"] = EventTermCfg(
            func=mdp.randomize_friction,
            mode="startup",
            params={"friction_ratio_range": tuple(dr["friction_ratio"])},
        )
    if "torso_mass_shift_kg" in dr:
        cfg.events["dr_mass"] = EventTermCfg(
            func=mdp.randomize_body_mass,
            mode="startup",
            params={
                "mass_shift_range": tuple(dr["torso_mass_shift_kg"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            },
        )
    gain_params = {}
    for source, target in (
        ("stiffness_scale", "stiffness_scale_range"),
        ("damping_scale", "damping_scale_range"),
    ):
        if source in dr:
            gain_params[target] = tuple(dr[source])
    if gain_params:
        cfg.events["dr_gains"] = EventTermCfg(
            func=mdp.randomize_joint_gains, mode="startup", params=gain_params
        )
    for key, weight in spec["rewards"].items():
        if key not in cfg.rewards:
            raise ValueError(f"Unknown reward term: {key}")
        cfg.rewards[key].weight = float(weight)
    if evaluation:
        cfg.auto_reset = False  # Keep terminal state available to evaluation.
        for group in cfg.observations.values():
            group.enable_corruption = False
    return cfg


def make_env(
    task: str, *, device: str = "cpu", video: bool = False, show_viewer: bool = False, **kwargs
):
    """Instantiate actual Genesis physics; there is no mock simulator fallback."""
    from humanoid.runtime import BenchmarkEnv
    import torch

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly")
        if device not in ("cuda", "cuda:0"):
            raise ValueError("Select a physical GPU with CUDA_VISIBLE_DEVICES, then use cuda:0")

    spec = kwargs.pop("config", None) or task_config(task)
    return BenchmarkEnv(
        make_env_cfg(task, config=spec, **kwargs),
        spec=spec,
        device=device,
        show_viewer=show_viewer,
        render_mode="rgb_array" if video else None,
    )
