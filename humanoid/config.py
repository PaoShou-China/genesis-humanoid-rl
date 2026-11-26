"""Strict configuration loading, shared across training and evaluation."""
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping without evaluating custom YAML objects."""
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def task_config(task: str, path: str | Path | None = None) -> dict[str, Any]:
    """Resolve an implemented task, rejecting unverified robot variants."""
    names = {"H1": "h1_standing", "H2": "h2_velocity"}
    if task not in names:
        raise ValueError(f"Task {task} is not implemented; available: {list(names)}")
    cfg = read_yaml(path or ROOT / "configs" / "tasks" / f"{names[task]}.yaml")
    if cfg["task"] != task or cfg["robot"] != "g1_29dof":
        raise ValueError("Task identity mismatch or unverified robot asset")
    if cfg["episode_length_s"] <= 0 or cfg["action_scale"] <= 0:
        raise ValueError("Episode length and action scale must be positive")
    if cfg["action_clip"] <= 0:
        raise ValueError("Action clip must be positive")
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        lo, hi = cfg["commands"][key]
        if lo > hi:
            raise ValueError(f"Reversed command range: {key}")
    return cfg
