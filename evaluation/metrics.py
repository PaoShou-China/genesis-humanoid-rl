"""Physical state metrics sampled before terminal resets."""

import torch

from humanoid.native import effective_link_masses


def measure(env, commands: torch.Tensor) -> dict[str, torch.Tensor]:
    """Measure body-frame tracking, tilt, contact slip and absolute mechanical power."""
    robot = env.scene["robot"]
    data = robot.data
    contact = env.scene["feet_contact"].data.in_contact
    feet, _ = robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"))
    foot_vel = data.body_lin_vel_w[:, feet, :2].norm(dim=-1)
    origin = data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    # Native link COM positions, weighted by actual inertial mass; root != CoM.
    entity = robot.gs_entity
    masses = effective_link_masses(entity)
    if masses.ndim == 1:
        masses = masses.unsqueeze(0)
    com = (entity.get_links_pos(ref="link_com") * masses.unsqueeze(-1)).sum(1)
    com = com / masses.sum(-1, keepdim=True)
    limits = data.joint_pos_limits
    violation = (
        ((data.joint_pos < limits[..., 0] - 1e-4) | (data.joint_pos > limits[..., 1] + 1e-4))
        .float()
        .mean(-1)
    )
    ids = robot.indexing.joint_dofs_idx
    force_low, force_high = entity.get_dofs_force_range(ids)
    torque = data.applied_torque
    torque_violation = ((torque < force_low - 1e-3) | (torque > force_high + 1e-3)).float().mean(-1)
    projection = getattr(env, "last_projection", None)
    zeros = torch.zeros(env.num_envs, device=env.device)
    push = getattr(env, "last_push_force", None)
    explicit = [a for a in robot.actuators if hasattr(a, "requested_effort")]
    saturation = (
        torch.cat(
            [(a.requested_effort.abs() > a.applied_effort.abs() + 1e-5).float() for a in explicit],
            -1,
        ).mean(-1)
        if explicit
        else zeros
    )
    residual = projection.residual_action if projection else None
    effective = projection.effective_residual_action if projection else None
    scale = getattr(env, "benchmark_spec", {}).get("action_scale", 1.0)
    return {
        "residual_requested_rad": residual.abs().amax(-1) * scale
        if residual is not None
        else zeros,
        "residual_effective_rad": effective.abs().amax(-1) * scale
        if effective is not None
        else zeros,
        "linear_error_m_s": (data.root_lin_vel_b[:, :2] - commands[:, :2]).norm(dim=-1),
        "yaw_error_rad_s": (data.root_ang_vel_b[:, 2] - commands[:, 2]).abs(),
        "tilt_rad": torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1)),
        "root_displacement_m": (data.root_pos_w[:, :2] - origin).norm(dim=-1),
        "joint_speed_rad_s": data.joint_vel.abs().mean(-1),
        "power_w": (data.applied_torque * data.joint_vel).abs().sum(-1),
        "slip_speed_m_s": (foot_vel * contact).sum(-1) / contact.sum(-1).clamp(min=1),
        "contact_count": contact.float().sum(-1),
        "com_x_m": com[:, 0] - env.scene.env_origins[:, 0],
        "com_y_m": com[:, 1] - env.scene.env_origins[:, 1],
        "com_z_m": com[:, 2] - env.scene.env_origins[:, 2],
        "joint_limit_violation": violation,
        "torque_limit_violation": torque_violation,
        "actuator_last_step_saturation": saturation,
        "action_intervention": projection.intervention.float() if projection else zeros,
        "shield_intervention": projection.intervention.float()
        if projection
        and getattr(env, "benchmark_spec", {}).get("shield", {}).get("type") == "pd_target"
        else zeros,
        "shield_infeasible": projection.infeasible.float() if projection else zeros,
        "external_force_n": push.norm(dim=-1) if push is not None else zeros,
    }
