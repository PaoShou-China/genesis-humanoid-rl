"""Simulator-independent sampled actuator and sensor contracts.

Delay counts calls at the declared actuator period, not policy frames or solver
substeps. The optional symmetric linear speed envelope is a stress-test model,
not an identified G1 motor curve.
"""

import torch


class CommandDelay:
    """Per-environment integer transport delay with isolated partial resets."""

    def __init__(self, initial, delay_range=(0, 0)):
        self.lo, self.hi = delay_range
        self.history = initial.unsqueeze(0).repeat(self.hi + 1, 1, 1).clone()
        self.lags = torch.zeros(initial.shape[0], dtype=torch.long, device=initial.device)
        self.cursor = 0
        self.reset(initial)

    def reset(self, initial, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.history[:, ids] = initial[ids]
        self.lags[ids] = torch.randint(
            self.lo, self.hi + 1, self.lags[ids].shape, device=self.lags.device
        )

    def __call__(self, command):
        self.history[self.cursor] = command
        slots = torch.arange(command.shape[0], device=command.device)
        delayed = self.history[(self.cursor - self.lags) % len(self.history), slots].clone()
        self.cursor = (self.cursor + 1) % len(self.history)
        return delayed


def sampled_pd(
    target, q, dq, kp, kd, effort, *, target_velocity=0.0, feedforward=0.0, no_load_speed=None
):
    """Return requested and bounded effort; gains may vary by environment."""
    requested = kp * (target - q) + kd * (target_velocity - dq) + feedforward
    bound = effort
    if no_load_speed is not None:
        # Deliberately conservative symmetric stress envelope, including braking.
        bound = effort * (1 - dq.abs() / no_load_speed).clamp(0, 1)
    return requested, torch.minimum(torch.maximum(requested, -bound), bound)


def actor_observation(
    *, q, dq, home, angular_velocity, gravity, commands, last_action, linear_velocity=None
):
    """Pinned flat-task term order. Normalization belongs to the exported actor.

    Omitting linear_velocity gives the 96-D proprioceptive profile. This removes
    a truth-state input; it does not supply an IMU attitude/velocity estimator.
    """
    terms = [angular_velocity, gravity, commands, q - home, dq, last_action]
    if linear_velocity is not None:
        terms.insert(0, linear_velocity)
    return torch.cat(terms, dim=-1)
