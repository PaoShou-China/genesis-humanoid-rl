"""Small runtime extension: compose control, apply real force, retain diagnostics."""

from genesislab.envs import ManagerBasedRLEnv

from humanoid.control import ActionPipeline
from humanoid.disturbances import apply_push


class BenchmarkEnv(ManagerBasedRLEnv):
    """Preserve upstream reset/step contracts without copying its simulator loop."""

    def __init__(self, cfg, *, spec, **kwargs):
        super().__init__(cfg, **kwargs)
        self.benchmark_spec = spec
        for actuator in self.scene["robot"].actuators:
            if hasattr(actuator, "physics_tick"):
                actuator.physics_tick = lambda: self._sim_step_counter
                actuator.last_tick = self._sim_step_counter
        self.action_pipeline = ActionPipeline(self, spec)
        import torch

        self.previous_residual = torch.zeros(self.num_envs, 29, device=self.device)
        self.last_residual_change = torch.zeros_like(self.previous_residual)
        self.last_projection = None
        self.last_push_force = None
        self.last_push_impulse = None
        self._native_scene_step = self.scene.step
        # Scene.step is the public physics boundary. Wrap it so forces are
        # reapplied after Genesis clears them at each physics step, including
        # all decimation steps, without copying the upstream environment loop.
        self.scene.step = self._step_physics

    def _step_physics(self):
        force = apply_push(self, self.benchmark_spec.get("push"))
        self.last_push_force = force
        if self.last_push_impulse is not None:
            self.last_push_impulse += force * self.physics_dt
        return self._native_scene_step()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "previous_residual"):
            self.previous_residual[env_ids] = 0
            self.last_residual_change[env_ids] = 0
        sampled = [a for a in self.scene["robot"].actuators if hasattr(a, "delay")]
        # A single command-transport lag per robot, shared by all motor groups.
        # Keep this identical to the portable MuJoCo actuator contract.
        if sampled:
            for actuator in sampled[1:]:
                actuator.delay.lags[env_ids] = sampled[0].delay.lags[env_ids]

    def step(self, action):
        import torch

        self.last_projection = self.action_pipeline(action.to(self.device))
        correction = self.last_projection.residual_action
        if correction is not None:
            self.last_residual_change = correction - self.previous_residual
            self.previous_residual.copy_(correction)
        self.last_push_impulse = torch.zeros(self.num_envs, 3, device=self.device)
        return super().step(self.last_projection.action)
