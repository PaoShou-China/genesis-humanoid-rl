"""Strict configuration loading, shared across training and evaluation."""

from pathlib import Path
from typing import Any
import math

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"H1": "h1_stiff_pd", "H2": "h2_velocity", "H3": "h3_stiff_pd", "H4": "h4_terrain"}


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping without evaluating custom YAML objects."""
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def task_config(task: str, path: str | Path | None = None) -> dict[str, Any]:
    """Resolve an implemented task, rejecting unverified robot variants."""
    if task not in TASKS:
        raise ValueError(f"Task {task} is not implemented; available: {list(TASKS)}")
    cfg = read_yaml(path or ROOT / "configs" / "tasks" / f"{TASKS[task]}.yaml")
    return validate_task(cfg, task)


def validate_task(cfg: dict[str, Any], task: str | None = None) -> dict[str, Any]:
    """Validate saved and supplied task configs through the same strict boundary."""
    task = task or cfg.get("task")
    if task not in TASKS:
        raise ValueError(f"Task {task} is not implemented")
    allowed = {
        "task",
        "robot",
        "episode_length_s",
        "action_scale",
        "action_clip",
        "commands",
        "rewards",
        "success",
        "reset_joint_range",
        "observation_noise",
        "randomization",
        "controller",
        "shield",
        "push",
        "terrain",
        "reward_profile",
        "pd_gains",
        "actuator",
        "timing",
        "actor_observation",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown task fields: {sorted(unknown)}")
    if cfg["task"] != task or cfg["robot"] != "g1_29dof":
        raise ValueError("Task identity mismatch or unverified robot asset")
    for key in ("episode_length_s", "action_scale", "action_clip"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if set(cfg["commands"]) != {"lin_vel_x", "lin_vel_y", "ang_vel_z"}:
        raise ValueError("Exactly three velocity command ranges are required")
    if task in ("H1", "H3") and any(list(pair) != [0.0, 0.0] for pair in cfg["commands"].values()):
        raise ValueError("Standing and standing push recovery require zero velocity commands")
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        lo, hi = cfg["commands"][key]
        if not all(math.isfinite(x) for x in (lo, hi)) or lo > hi:
            raise ValueError(f"Reversed command range: {key}")
    if any(not math.isfinite(value) for value in cfg["rewards"].values()):
        raise ValueError("Reward weights must be finite")
    expected_success = (
        {"max_tilt_rad", "max_root_displacement_m"}
        if task in ("H1", "H3")
        else {"max_mean_linear_error_m_s", "max_mean_yaw_error_rad_s"}
    )
    if set(cfg["success"]) != expected_success or any(
        not math.isfinite(value) or value <= 0 for value in cfg["success"].values()
    ):
        raise ValueError("Success thresholds must be positive and match the task")
    reset = cfg.get("reset_joint_range", [-0.1, 0.1])
    if len(reset) != 2 or not all(math.isfinite(v) for v in reset) or reset[0] > reset[1]:
        raise ValueError("Invalid joint reset range")
    if cfg.get("reward_profile") not in (None, "survival", "soft_safety"):
        raise ValueError("Unknown reward profile")
    if cfg.get("actor_observation", "truth_velocity") not in ("truth_velocity", "proprioceptive"):
        raise ValueError("Unknown actor observation profile")
    actuator = cfg.get("actuator", {})
    if set(actuator) - {"mode", "delay_steps", "no_load_speed_rad_s"}:
        raise ValueError("Unknown actuator field")
    mode = actuator.get("mode", "implicit")
    if mode not in ("implicit", "explicit_pd"):
        raise ValueError("Unknown actuator mode")
    delay = actuator.get("delay_steps", [0, 0])
    if (
        not isinstance(delay, (list, tuple))
        or len(delay) != 2
        or any(type(v) is not int for v in delay)
        or not 0 <= delay[0] <= delay[1] <= 1000
    ):
        raise ValueError("delay_steps must be two ordered integers in [0, 1000]")
    speed = actuator.get("no_load_speed_rad_s")
    if speed is not None and (not math.isfinite(speed) or speed <= 0):
        raise ValueError("no_load_speed_rad_s must be finite and positive")
    if mode == "implicit" and (delay != [0, 0] and delay != (0, 0) or speed is not None):
        raise ValueError("Delay and speed envelope require explicit_pd")
    timing = cfg.get("timing", {})
    if set(timing) - {"physics_dt_s", "control_dt_s", "substeps"}:
        raise ValueError("Unknown timing field")
    physics_dt = timing.get("physics_dt_s", 0.02)
    control_dt = timing.get("control_dt_s", 0.02)
    if any(not math.isfinite(v) or v <= 0 for v in (physics_dt, control_dt)):
        raise ValueError("Timing periods must be finite and positive")
    ratio = control_dt / physics_dt
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
        raise ValueError("control_dt_s must be an integer multiple of physics_dt_s")
    if type(timing.get("substeps", 2)) is not int or not 1 <= timing.get("substeps", 2) <= 100:
        raise ValueError("substeps must be an integer in [1, 100]")
    dr = cfg.get("randomization", {})
    gains = cfg.get("pd_gains", {})
    if set(gains) - {"stiffness_scale", "damping_scale"} or any(
        not math.isfinite(value) or value <= 0 for value in gains.values()
    ):
        raise ValueError("PD gain scales must be finite and positive")
    if set(dr) - {"friction_ratio", "torso_mass_shift_kg", "stiffness_scale", "damping_scale"}:
        raise ValueError("Unsupported randomization parameter")
    for key, pair in dr.items():
        lo, hi = pair
        if not all(math.isfinite(x) for x in pair) or lo > hi:
            raise ValueError(f"Invalid randomization range: {key}")
        if key != "torso_mass_shift_kg" and lo <= 0:
            raise ValueError(f"Randomization ratio must be positive: {key}")
        if key == "torso_mass_shift_kg" and lo < -2:
            raise ValueError("Torso mass shifts below -2 kg are outside the verified range")
    controller = cfg.get("controller", {})
    if set(controller) - {"type", "kp", "kd", "ankle_bias_rad", "knee_bias_rad", "residual_scale"}:
        raise ValueError("Unsupported controller field")
    if controller.get("type", "none") not in ("none", "ankle_feedback"):
        raise ValueError("Unknown base controller")
    if not 0 <= controller.get("residual_scale", 1.0) <= 1:
        raise ValueError("residual_scale must be between zero and one")
    if any(not math.isfinite(v) for k, v in controller.items() if k != "type"):
        raise ValueError("Controller parameters must be finite")
    if any(controller.get(k, 0) < 0 for k in ("kp", "kd")):
        raise ValueError("Feedback gains must be nonnegative")
    if set(cfg.get("shield", {})) - {"type"}:
        raise ValueError("Unknown shield field")
    if cfg.get("shield", {}).get("type", "none") not in ("none", "pd_target"):
        raise ValueError("Unknown shield")
    if cfg.get("terrain", "flat") not in ("flat", "rough"):
        raise ValueError("Unknown terrain")
    if "push" in cfg:
        push = cfg["push"]
        if set(push) != {"start_s", "duration_s", "force_n"}:
            raise ValueError("Push requires start_s, duration_s and world-frame force_n")
        if push["start_s"] < 0 or push["duration_s"] <= 0 or len(push["force_n"]) != 3:
            raise ValueError("Invalid push timing or force")
        if not all(
            math.isfinite(x) for x in [push["start_s"], push["duration_s"], *push["force_n"]]
        ):
            raise ValueError("Non-finite push specification")
    return cfg
