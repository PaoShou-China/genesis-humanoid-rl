"""Goal-conditioned task states, rewards and measurable success events.

All states are batched. Goals and episode memory reset by environment ID.
No demonstrated motion is synthesized when reference data is unavailable.
"""

from pathlib import Path

import torch

from humanoid.task_catalog import CUSTOM_TASKS


def yaw_from_quat(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def require_motion(spec):
    """Check data before building a simulator, not after an expensive compile."""
    from scripts.validate_motion import validate_motion

    path = Path(spec["objective"]["motion_file"]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"H5 requires an actual motion NPZ: {path}; no trained tracker or synthetic fallback"
        )
    return validate_motion(path)


class TaskObjective:
    def __init__(self, env, spec):
        self.env, self.spec = env, spec
        self.kind = CUSTOM_TASKS.get(spec["task"])
        self.options = spec.get("objective", {})
        n, device = env.num_envs, env.device
        self.goal = torch.zeros(n, 6, device=device)
        self.hold = torch.zeros(n, device=device)
        self.achieved = torch.zeros(n, dtype=torch.bool, device=device)
        self.airborne = torch.zeros_like(self.achieved)
        self.flight_time = torch.zeros(n, device=device)
        self.last_tick = torch.full((n,), -1, device=device, dtype=torch.long)
        self.phase = torch.zeros(n, device=device)
        self.motion = None
        if self.kind == "motion":
            import numpy as np

            info = require_motion(spec)
            with np.load(
                Path(self.options["motion_file"]).expanduser(), allow_pickle=False
            ) as data:
                names = data["joint_names"].tolist()
                if set(names) != set(env.scene["robot"].joint_names):
                    raise ValueError("Motion joints must match the audited 29-DoF G1")
                order = [names.index(name) for name in env.scene["robot"].joint_names]
                self.motion = torch.tensor(
                    data["joint_pos"][:, order], device=device, dtype=torch.float32
                )
                self.motion_velocity = torch.tensor(
                    data["joint_vel"][:, order], device=device, dtype=torch.float32
                )
            self.fps, self.motion_duration = info["fps"], info["duration_s"]
            self.motion_sha256 = info["file_sha256"]
        if self.kind == "reach":
            self.hands, _ = env.scene["robot"].find_bodies(
                ("left_wrist_yaw_link", "right_wrist_yaw_link")
            )
            if len(self.hands) != 2:
                raise ValueError("Bimanual reaching requires two named wrists")

    def reset(self, ids):
        self.hold[ids], self.flight_time[ids], self.phase[ids] = 0, 0, 0
        self.achieved[ids], self.airborne[ids], self.last_tick[ids] = False, False, -1
        robot = self.env.scene["robot"]
        data = robot.data
        count = len(ids)
        if self.kind == "heading":
            span = self.options["heading_range_rad"]
            self.goal[ids, 0] = torch.empty(count, device=self.env.device).uniform_(*span)
        elif self.kind == "waypoint":
            self.goal[ids, :2] = data.root_pos_w[ids, :2]
            self.goal[ids, 0] += self.options["distance_m"]
        elif self.kind == "height":
            self.goal[ids, 0] = torch.empty(count, device=self.env.device).uniform_(
                *self.options["height_range_m"]
            )
        elif self.kind == "reach":
            origin = data.body_pos_w[ids][:, self.hands]
            spread = self.options["target_offset_m"]
            self.goal[ids] = (origin + torch.empty_like(origin).uniform_(-spread, spread)).flatten(
                1
            )
        elif self.kind == "getup":
            # Alternate prone/supine by slot; root state is physically reset.
            pose = data.root_state_w[ids, :7].clone()
            pose[:, 2] = self.env.scene.env_origins[ids, 2] + 0.25
            pitch = torch.where(ids % 2 == 0, torch.pi / 2, -torch.pi / 2)
            pose[:, 3:7] = 0
            pose[:, 3], pose[:, 5] = torch.cos(pitch / 2), torch.sin(pitch / 2)
            robot.write_root_pose_to_sim(pose, ids)
            robot.write_root_velocity_to_sim(torch.zeros(count, 6, device=self.env.device), ids)
        elif self.kind == "object_push":
            self.goal[ids, :3] = self.env.scene["object"].data.root_pos_w[ids]
            self.goal[ids, 0] += self.options["distance_m"]

    def reference(self):
        time = self.env.episode_length_buf * self.env.step_dt
        if self.kind == "motion":
            time = (time + self.phase) % self.motion_duration
            index = time * self.fps
            lo = index.long().clamp(max=len(self.motion) - 2)
            fraction = (index - lo).unsqueeze(-1)
            q = torch.lerp(self.motion[lo], self.motion[lo + 1], fraction)
            dq = torch.lerp(self.motion_velocity[lo], self.motion_velocity[lo + 1], fraction)
            return q, dq
        if self.kind == "jump":
            return torch.where((time >= 0.8) & (time < 1.6), self.options["apex_height_m"], 0.74)
        return self.goal[:, 0]

    def observe(self):
        data = self.env.scene["robot"].data
        if self.kind == "heading":
            error = wrap_angle(self.goal[:, 0] - yaw_from_quat(data.root_quat_w))
            return torch.stack([error.sin(), error.cos()], -1)
        if self.kind in ("waypoint", "object_push"):
            origin = (
                data.root_pos_w
                if self.kind == "waypoint"
                else self.env.scene["object"].data.root_pos_w
            )
            delta = self.goal[:, :2] - origin[:, :2]
            yaw = yaw_from_quat(data.root_quat_w)
            local = torch.stack(
                [
                    yaw.cos() * delta[:, 0] + yaw.sin() * delta[:, 1],
                    -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1],
                ],
                -1,
            )
            if self.kind == "object_push":
                relative = origin - data.root_pos_w
                body_relative = torch.stack(
                    [
                        yaw.cos() * relative[:, 0] + yaw.sin() * relative[:, 1],
                        -yaw.sin() * relative[:, 0] + yaw.cos() * relative[:, 1],
                        relative[:, 2],
                    ],
                    -1,
                )
                return torch.cat([local, body_relative], -1)
            return local
        if self.kind in ("height", "jump"):
            height = data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
            target = self.reference()
            return torch.stack([target, target - height, data.root_lin_vel_w[:, 2]], -1)
        if self.kind == "reach":
            from genesislab.utils.lab_api.math import quat_apply_inverse

            error = self.goal.view(-1, 2, 3) - data.body_pos_w[:, self.hands]
            quats = data.root_quat_w[:, None].expand(-1, 2, -1)
            return quat_apply_inverse(quats.reshape(-1, 4), error.reshape(-1, 3)).reshape(-1, 6)
        if self.kind == "getup":
            return data.root_pos_w[:, 2:3] - self.env.scene.env_origins[:, 2:3]
        if self.kind == "motion":
            q, dq = self.reference()
            return torch.cat([q - data.joint_pos, dq - data.joint_vel], -1)
        return torch.zeros(self.env.num_envs, 0, device=self.env.device)

    def error(self):
        data = self.env.scene["robot"].data
        height = data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
        tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))
        if self.kind == "heading":
            return wrap_angle(self.goal[:, 0] - yaw_from_quat(data.root_quat_w)).abs()
        if self.kind == "waypoint":
            return (self.goal[:, :2] - data.root_pos_w[:, :2]).norm(dim=-1)
        if self.kind in ("height", "jump"):
            return (height - self.reference()).abs()
        if self.kind == "reach":
            return (self.goal.view(-1, 2, 3) - data.body_pos_w[:, self.hands]).norm(dim=-1).amax(-1)
        if self.kind == "getup":
            return (self.options["upright_height_m"] - height).clamp(min=0) + tilt
        if self.kind == "object_push":
            return (self.goal[:, :2] - self.env.scene["object"].data.root_pos_w[:, :2]).norm(dim=-1)
        if self.kind == "motion":
            q, _ = self.reference()
            return (q - data.joint_pos).square().mean(-1).sqrt()
        return torch.zeros(self.env.num_envs, device=self.env.device)

    def update(self):
        tick = self.env.episode_length_buf
        fresh = (tick != self.last_tick) & (tick > 0)
        if not fresh.any():
            return
        data = self.env.scene["robot"].data
        error = self.error()
        tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))
        stable = (error <= self.spec["success"]["tolerance"]) & (tilt < 0.35)
        if self.kind == "getup":
            height = data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
            stable &= (data.root_lin_vel_w.norm(dim=-1) < 0.3) & (
                height >= self.options["upright_height_m"]
            )
        if self.kind == "jump":
            flying = ~self.env.scene["feet_contact"].data.in_contact.any(-1)
            self.flight_time = torch.where(
                fresh & flying,
                self.flight_time + self.env.step_dt,
                torch.where(fresh, torch.zeros_like(self.flight_time), self.flight_time),
            )
            height = data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
            self.airborne |= (
                fresh
                & (self.flight_time >= 0.04)
                & (height >= self.options["minimum_flight_height_m"])
            )
            stable &= self.airborne & ~flying & (tick * self.env.step_dt >= 1.6)
        if self.kind == "object_push":
            stable &= self.env.scene["object"].data.root_lin_vel_w.norm(dim=-1) < 0.1
        self.hold = torch.where(
            fresh, torch.where(stable, self.hold + self.env.step_dt, 0), self.hold
        )
        # Acquisition memory is useful for jump/navigation; terminal failure is
        # checked independently and always overrides this event in evaluation.
        attained = self.hold + 1e-6 >= self.spec["success"]["hold_s"]
        if self.kind in ("jump", "waypoint", "object_push"):
            self.achieved |= fresh & attained
        else:
            self.achieved = torch.where(fresh, attained, self.achieved)
        self.last_tick.copy_(tick)


def goal_observation(env):
    if not hasattr(env, "task_objective"):
        env.task_objective = TaskObjective(env, env.benchmark_spec)
    return env.task_objective.observe()


def objective_reward(env):
    objective = env.task_objective
    objective.update()
    reward = torch.exp(
        -objective.error().square() / objective.options.get("reward_width", 0.25) ** 2
    )
    if objective.kind == "object_push":
        distance = (env.scene["object"].data.root_pos_w - env.scene["robot"].data.root_pos_w).norm(
            dim=-1
        )
        reward = reward + 0.2 * torch.exp(-distance)
    return reward


def objective_metrics(env):
    objective = env.task_objective
    objective.update()
    return {"objective_error": objective.error(), "objective_achieved": objective.achieved.float()}
