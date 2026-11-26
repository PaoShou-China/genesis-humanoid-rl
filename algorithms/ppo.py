"""Use upstream RSL-RL through GenesisLab's checkpoint-aware adapter."""

from dataclasses import asdict
from pathlib import Path

from humanoid.config import ROOT, read_yaml


def runner_config(path: str | Path | None = None, *, seed: int = 0) -> dict:
    """Override only explicitly supported PPO configuration fields."""
    from genesislab.tasks.velocity.config.g1.rl_cfg import g1_ppo_runner_cfg

    cfg = g1_ppo_runner_cfg()
    spec = read_yaml(path or ROOT / "configs/algorithms/ppo.yaml")
    cfg.seed = seed
    cfg.upload_model = False
    for key, value in spec.items():
        if key == "actor_hidden_dims":
            cfg.actor.hidden_dims = tuple(value)
        elif key == "critic_hidden_dims":
            cfg.critic.hidden_dims = tuple(value)
        elif key == "init_std":
            if value <= 0:
                raise ValueError("init_std must be positive")
            cfg.actor.distribution_cfg["init_std"] = value
        elif key in ("learning_rate", "entropy_coef", "num_learning_epochs", "num_mini_batches"):
            setattr(cfg.algorithm, key, value)
        elif key in ("num_steps_per_env", "max_iterations", "save_interval", "clip_actions"):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown PPO field: {key}")
    if min(cfg.num_steps_per_env, cfg.max_iterations, cfg.save_interval) < 1:
        raise ValueError("Rollout length, iterations and save interval must be positive")
    return asdict(cfg)


def make_runner(env, cfg: dict, log_dir: str | None = None):
    """Return the real PPO runner; wrap once and preserve the action bound."""
    from copy import deepcopy

    from genesislab.rl.runner import GenesisLabOnPolicyRunner
    from genesislab.rl.vecenv_wrapper import RslRlVecEnvWrapper

    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    return GenesisLabOnPolicyRunner(wrapped, deepcopy(cfg), log_dir, device=str(env.device))
