"""Opt-in actuator implementation using the upstream public actuator hooks."""

from dataclasses import dataclass

import torch
from genesislab.actuator.actuator import ImplicitActuator, ImplicitActuatorCfg

from humanoid.transfer import CommandDelay, sampled_pd


@dataclass(kw_only=True)
class SampledPdCfg(ImplicitActuatorCfg):
    delay_steps: tuple[int, int] = (0, 0)
    no_load_speed_rad_s: float | None = None

    def build(self, entity):
        return SampledPd(self, entity)


class SampledPd(ImplicitActuator):
    """Read live gains after DR and submit pure effort once per physics step."""

    def __init__(self, cfg, entity):
        super().__init__(cfg, entity)
        home = entity.data.default_joint_pos[:, self.joint_ids]
        self.initial = torch.cat([home, torch.zeros_like(home), torch.zeros_like(home)], -1)
        self.delay = CommandDelay(self.initial, cfg.delay_steps)
        self.last_command = self.initial.clone()
        self.physics_tick = None
        self.last_tick = 0
        self.requested_effort = torch.zeros_like(home)
        self.applied_effort = torch.zeros_like(home)

    def reset(self, env_ids=None):
        self.delay.reset(self.initial, env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.requested_effort[ids] = 0
        self.applied_effort[ids] = 0
        self.last_command[ids] = self.initial[ids]

    def apply(self):
        data, native, ids = self._entity.data, self._entity.gs_entity, self.joint_ids
        if self.physics_tick is None:
            raise RuntimeError("SampledPd must be bound to BenchmarkEnv's physics clock")
        tick = self.physics_tick()
        # GenesisLab also writes actuators during reset, without advancing time.
        # Those writes must not consume delay slots in ANY environment.
        if tick != self.last_tick:
            self.last_command = self.delay(
                torch.cat(
                    [
                        data.joint_pos_target[:, ids],
                        data.joint_vel_target[:, ids],
                        data.joint_effort_target[:, ids],
                    ],
                    -1,
                )
            )
            self.last_tick = tick
        target, velocity, feedforward = self.last_command.chunk(3, -1)
        lower, upper = native.get_dofs_force_range(self._dofs_idx)
        self.requested_effort, self.applied_effort = sampled_pd(
            target,
            data.joint_pos[:, ids],
            data.joint_vel[:, ids],
            native.get_dofs_kp(self._dofs_idx),
            native.get_dofs_kv(self._dofs_idx),
            torch.minimum(lower.abs(), upper.abs()),
            target_velocity=velocity,
            feedforward=feedforward,
            no_load_speed=self.cfg.no_load_speed_rad_s,
        )
        native.control_dofs_force(self.applied_effort, self._dofs_idx)
