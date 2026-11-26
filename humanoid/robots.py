"""Verify the loaded model and export its actual joint ordering."""
import torch


def validate_robot(env) -> dict:
    """Fail on an incorrect DoF count, invalid limits, or invalid default pose."""
    robot = env.scene["robot"]
    names = robot.joint_names
    limits = robot.data.joint_pos_limits[0]
    home = robot.data.default_joint_pos[0]
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError(f"Expected 29 unique G1 joints, got {len(names)}")
    if robot.is_fixed_base or env.action_manager.total_action_dim != 29:
        raise ValueError("G1 must have a floating base and 29 action dimensions")
    if not torch.isfinite(limits).all() or not (limits[:, 0] < limits[:, 1]).all():
        raise ValueError("Invalid joint limits")
    if not ((home >= limits[:, 0]) & (home <= limits[:, 1])).all():
        raise ValueError("Default pose is outside asset joint limits")
    feet = list(env.scene["feet_contact"].link_names)
    if set(feet) != {"left_ankle_roll_link", "right_ankle_roll_link"}:
        raise ValueError(f"Unexpected foot contact links: {feet}")
    return {
        "num_joints": len(names), "joint_names": names,
        "position_limits_rad": limits.cpu().tolist(), "home_rad": home.cpu().tolist(),
        "feet": feet, "floating_base": True,
        "physics_dt_s": env.physics_dt, "control_dt_s": env.step_dt,
        "quaternion_order": "wxyz", "gravity_m_s2": list(env.cfg.sim.gravity),
    }
