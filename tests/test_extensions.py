"""Exercise force pulses and action projection on the actual Genesis fixture."""

import pytest
import torch


@pytest.mark.simulation
def test_real_torso_force_and_exact_requested_impulse(env):
    from humanoid.disturbances import apply_push

    env.reset(seed=123)
    force = {"start_s": 0.04, "duration_s": 0.04, "force_n": [10.0, 0.0, 0.0]}
    total = 0.0
    for step in range(6):
        requested = apply_push(env, force)
        total += float(requested[0, 0]) * env.step_dt
        assert bool(requested.any()) == (step in (2, 3))
        env.step(torch.zeros(2, 29))
    assert total == pytest.approx(0.4)


@pytest.mark.simulation
def test_force_changes_the_actual_physical_trajectory(env):
    def rollout(push):
        env.reset(seed=120)
        if push:
            env.benchmark_spec["push"] = {
                "start_s": 0.04,
                "duration_s": 0.04,
                "force_n": [30.0, 0.0, 0.0],
            }
        else:
            env.benchmark_spec.pop("push", None)
        for _ in range(6):
            env.step(torch.zeros(2, 29))
        return env.scene["robot"].data.root_lin_vel_w.clone()

    try:
        baseline = rollout(False)
        pushed = rollout(True)
        assert (pushed - baseline).abs().max() > 1e-4
    finally:
        env.benchmark_spec.pop("push", None)


@pytest.mark.simulation
def test_pipeline_applies_shield_and_rejects_nan(env):
    from humanoid.control import ActionPipeline

    env.reset(seed=124)
    spec = dict(env.benchmark_spec, shield={"type": "pd_target"})
    pipeline = ActionPipeline(env, spec)
    result = pipeline(torch.ones(2, 29) * 10)
    assert result.intervention.all()
    assert torch.isfinite(result.action).all()
    assert result.action.abs().max() <= 1.00001
    with pytest.raises(FloatingPointError):
        pipeline(torch.full((2, 29), float("nan")))
