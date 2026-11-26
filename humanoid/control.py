"""Interpretable standing feedback and explicitly limited PD-target projection."""

from dataclasses import dataclass

import torch


def ankle_feedback(gravity, angular_velocity, joint_names, spec, action_scale):
    """Correct root tilt with both ankles; an empirical controller, not a gait."""
    pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    kp, kd = spec.get("kp", 0.5), spec.get("kd", 0.2)
    correction = torch.zeros(gravity.shape[0], len(joint_names), device=gravity.device)
    for side in ("left", "right"):
        correction[:, joint_names.index(f"{side}_ankle_pitch_joint")] = (
            kp * pitch + kd * angular_velocity[:, 1] + spec.get("ankle_bias_rad", 0.12)
        ).clamp(-0.4, 0.4)
        correction[:, joint_names.index(f"{side}_ankle_roll_joint")] = (
            kp * roll + kd * angular_velocity[:, 0]
        ).clamp(-0.2, 0.2)
        correction[:, joint_names.index(f"{side}_knee_joint")] = spec.get("knee_bias_rad", -0.1)
    return correction / action_scale


@dataclass
class Projection:
    action: torch.Tensor
    intervention: torch.Tensor
    infeasible: torch.Tensor


def project_pd_target(action, *, q, dq, home, limits, kp, kd, effort, scale, clip) -> Projection:
    """Intersect target-position and affine-PD torque intervals where feasible.

    On an empty intersection, prioritize action/position targets and report it.
    This bounds a nominal PD proxy, not Genesis's implicit force or future states.
    """
    target = home + scale * action.clamp(-clip, clip)
    lower = torch.maximum(limits[..., 0], home - scale * clip)
    upper = torch.minimum(limits[..., 1], home + scale * clip)
    torque_low = q + (-effort + kd * dq) / kp
    torque_high = q + (effort + kd * dq) / kp
    feasible_low, feasible_high = (
        torch.maximum(lower, torque_low),
        torch.minimum(upper, torque_high),
    )
    empty = feasible_low > feasible_high
    projected = torch.minimum(torch.maximum(target, feasible_low), feasible_high)
    fallback = torch.minimum(torch.maximum(target, lower), upper)
    projected = torch.where(empty, fallback, projected)
    result = (projected - home) / scale
    return Projection(result, (result - action).abs().amax(-1) > 1e-6, empty.any(-1))


class ActionPipeline:
    """One action path for training, evaluation, residual feedback and shielding."""

    def __init__(self, env, spec):
        self.env, self.spec = env, spec
        self.robot = env.scene["robot"]
        entity, ids = self.robot.gs_entity, self.robot.indexing.joint_dofs_idx
        self.kp, self.kd = entity.get_dofs_kp(ids), entity.get_dofs_kv(ids)
        lower, upper = entity.get_dofs_force_range(ids)
        self.effort = torch.minimum(lower.abs(), upper.abs())

    def __call__(self, action):
        data = self.robot.data
        ctrl = self.spec.get("controller", {})
        if ctrl.get("type") == "ankle_feedback":
            action = ankle_feedback(
                data.projected_gravity_b,
                data.root_ang_vel_b,
                self.robot.joint_names,
                ctrl,
                self.spec["action_scale"],
            ) + (ctrl.get("residual_scale", 1.0) * action)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Policy produced a non-finite action")
        if self.spec.get("shield", {}).get("type") == "pd_target":
            return project_pd_target(
                action,
                q=data.joint_pos,
                dq=data.joint_vel,
                home=data.default_joint_pos,
                limits=data.joint_pos_limits,
                kp=self.kp,
                kd=self.kd,
                effort=self.effort,
                scale=self.spec["action_scale"],
                clip=self.spec["action_clip"],
            )
        applied = action.clamp(-self.spec["action_clip"], self.spec["action_clip"])
        flags = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return Projection(applied, (applied - action).abs().amax(-1) > 1e-6, flags)
