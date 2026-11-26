"""Verify the loaded model and export its actual joint ordering."""
import torch

from humanoid.config import ROOT, read_yaml


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
    expected = read_yaml(ROOT / "configs/robots/g1_29dof.yaml")
    if names != [joint["name"] for joint in expected["joints"]]:
        raise ValueError("Runtime joint ordering differs from the audited asset")
    if env.action_manager.get_term("joint_pos").joint_names != names:
        raise ValueError("Action order differs from robot joint order")
    expected_limits = torch.tensor([j["position_limits_rad"] for j in expected["joints"]],
                                   device=limits.device, dtype=limits.dtype)
    torch.testing.assert_close(limits, expected_limits)
    entity = robot.gs_entity
    masses = entity.get_links_inertial_mass()
    if not torch.isfinite(masses).all() or (masses < 0).any() or masses.sum() <= 0:
        raise ValueError("Invalid link masses")
    dofs = robot.data.indexing.joint_dofs_idx
    def first_env(value):
        return value[0] if value.ndim > 1 else value
    kp, kd = first_env(entity.get_dofs_kp(dofs)), first_env(entity.get_dofs_kv(dofs))
    for actual, field in ((kp, "kp"), (kd, "kd")):
        target = torch.tensor([j[field] for j in expected["joints"]],
                              device=actual.device, dtype=actual.dtype)
        torch.testing.assert_close(actual, target)
    feet = list(env.scene["feet_contact"].link_names)
    if set(feet) != {"left_ankle_roll_link", "right_ankle_roll_link"}:
        raise ValueError(f"Unexpected foot contact links: {feet}")
    return {
        "num_joints": len(names), "joint_names": names,
        "position_limits_rad": limits.cpu().tolist(), "home_rad": home.cpu().tolist(),
        "feet": feet, "floating_base": True,
        "physics_dt_s": env.physics_dt, "control_dt_s": env.step_dt,
        "quaternion_order": "wxyz", "gravity_m_s2": list(env.cfg.sim.gravity),
        "kp": kp.cpu().tolist(), "kd": kd.cpu().tolist(),
        "link_masses_kg": first_env(masses).cpu().tolist(),
        "total_mass_kg": float(first_env(masses).sum()),
    }
