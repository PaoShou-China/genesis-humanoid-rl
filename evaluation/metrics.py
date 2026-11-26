"""Physical state metrics sampled before terminal resets."""
import torch


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
    masses = entity.get_links_inertial_mass()
    if masses.ndim == 1:
        masses = masses.unsqueeze(0)
    com = (entity.get_links_pos(ref="link_com") * masses.unsqueeze(-1)).sum(1)
    com = com / masses.sum(-1, keepdim=True)
    return {
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
    }
