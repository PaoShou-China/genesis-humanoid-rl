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
        elif key == "zero_actor_output":
            if type(value) is not bool:
                raise ValueError("zero_actor_output must be boolean")
            continue
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
    result = asdict(cfg)
    if "zero_actor_output" in spec:
        result["zero_actor_output"] = spec["zero_actor_output"]
    return result


def make_runner(env, cfg: dict, log_dir: str | None = None):
    """Return the real PPO runner; wrap once and preserve the action bound."""
    from copy import deepcopy

    from genesislab.rl.runner import GenesisLabOnPolicyRunner
    from genesislab.rl.vecenv_wrapper import RslRlVecEnvWrapper

    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    native_cfg = deepcopy(cfg)
    zero_output = native_cfg.pop("zero_actor_output", False)
    runner = GenesisLabOnPolicyRunner(wrapped, native_cfg, log_dir, device=str(env.device))
    if zero_output:
        import torch

        policy = runner.alg.get_policy()
        head = [module for module in policy.mlp.modules() if isinstance(module, torch.nn.Linear)][
            -1
        ]
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        runner.initialization_report = {
            "zero_actor_output": True,
            "head_weight_max_abs": float(head.weight.detach().abs().max()),
            "head_bias_max_abs": float(head.bias.detach().abs().max()),
            "note": "Zero deterministic mean; stochastic exploration is still enabled during PPO",
        }
    return runner
