"""Paired physical conditions and a command RNG independent of policy outcomes."""

import hashlib
import json

import numpy as np


def physical_protocol(spec, *, seed, num_envs, episodes):
    """Separate the test conditions from the algorithm being compared."""
    return {
        "version": 2,
        "task": spec["task"],
        "robot": spec["robot"],
        "episode_length_s": spec["episode_length_s"],
        "commands": spec["commands"],
        "success": spec["success"],
        "terrain": spec.get("terrain", "flat"),
        "randomization": spec.get("randomization", {}),
        "pd_gains": spec.get("pd_gains", {}),
        "push": spec.get("push"),
        "reset_joint_range": spec.get("reset_joint_range", [-0.1, 0.1]),
        "physics": {"dt_s": 0.02, "substeps": 2, "decimation": 1},
        "command_interval_s": 4.0,
        "standing_fraction": 0.1 if spec["task"] in ("H2", "H4") else 1.0,
        "seed": seed,
        "num_envs": num_envs,
        "episodes": episodes,
    }


def fingerprint(protocol):
    """Hash canonical physical conditions, not reward/controller implementation choices."""
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def command_schedule(spec, *, seed, horizon, num_envs, command_steps):
    """Precompute the complete plan, even when every robot falls early."""
    rng = np.random.default_rng(seed + 1_000_000)
    bounds = np.array([spec["commands"][k] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])
    plan = rng.uniform(bounds[:, 0], bounds[:, 1], (1 + horizon // command_steps, num_envs, 3))
    if spec["task"] in ("H2", "H4"):
        plan[rng.random(plan.shape[:2]) < 0.1] = 0.0
    return plan
