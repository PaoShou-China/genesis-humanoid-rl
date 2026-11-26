"""Small runtime extension: compose control, apply real force, retain diagnostics."""

from genesislab.envs import ManagerBasedRLEnv

from humanoid.control import ActionPipeline
from humanoid.disturbances import apply_push


class BenchmarkEnv(ManagerBasedRLEnv):
    """Preserve upstream reset/step contracts without copying its simulator loop."""

    def __init__(self, cfg, *, spec, **kwargs):
        super().__init__(cfg, **kwargs)
        self.benchmark_spec = spec
        self.action_pipeline = ActionPipeline(self, spec)
        self.last_projection = None
        self.last_push_force = None

    def step(self, action):
        self.last_projection = self.action_pipeline(action.to(self.device))
        self.last_push_force = apply_push(self, self.benchmark_spec.get("push"))
        return super().step(self.last_projection.action)
