"""Residual RL must preserve the base policy at zero and obey physical budgets."""

from copy import deepcopy

import pytest
import torch

from humanoid.config import ROOT, task_config, validate_task
from humanoid.deployment import PortableController, control_contract


def setup_controller(**changes):
    spec = task_config("H1", ROOT / "configs/tasks/h1_residual_rl.yaml")
    spec.update(changes)
    validate_task(spec)
    controller = PortableController(control_contract(spec))
    state = {
        "q": controller.values["home_rad"].clone(),
        "dq": torch.zeros(1, 29),
        "gravity": torch.tensor([[0.0, 0.0, -1.0]]),
        "angular_velocity": torch.zeros(1, 3),
    }
    return controller, state


@pytest.mark.parametrize("shield", [{}, {"type": "pd_target"}])
def test_zero_and_disabled_residual_preserve_same_base(shield):
    controller, state = setup_controller(
        controller={"type": "ankle_feedback", "ankle_bias_rad": 0.03, "knee_bias_rad": -0.02},
        shield=shield,
    )
    zero = controller.process(torch.zeros(1, 29), state)
    torch.testing.assert_close(zero.action, zero.baseline_action, rtol=0, atol=0)
    controller.spec["residual"]["enabled"] = False
    disabled = controller.process(torch.full((1, 29), 1e5), state)
    torch.testing.assert_close(disabled.action, zero.action, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0])
def test_residual_budget_is_in_radians_not_action_units(scale):
    controller, state = setup_controller(action_scale=scale)
    result = controller.process(torch.full((1, 29), 100.0), state)
    torch.testing.assert_close(result.residual_action * scale, torch.full((1, 29), 0.05))
    assert (result.effective_residual_action * scale).abs().max() <= 0.05 + 1e-7


def test_joint_mask_and_projection_preserve_residual_bound():
    controller, state = setup_controller(
        residual={"max_position_rad": 0.05, "joint_names": ["left_ankle_pitch_joint"]},
        shield={"type": "pd_target"},
    )
    state["dq"] += 5.0  # Force the proxy to intervene on the combined target.
    result = controller.process(torch.ones(1, 29), state)
    assert torch.count_nonzero(result.residual_action) == 1
    assert (result.effective_residual_action * 0.5).abs().max() <= 0.05 + 1e-7
    assert result.intervention.item()


@pytest.mark.parametrize(
    "residual",
    [
        {"max_position_rad": -0.01},
        {"max_position_rad": float("nan")},
        {"max_position_rad": 0.05, "joint_names": ["missing_joint"]},
        {"max_position_rad": 0.05, "enabled": "yes"},
    ],
)
def test_invalid_residual_configs_are_rejected(residual):
    with pytest.raises(ValueError):
        setup_controller(residual=residual)


def test_residual_does_not_change_physical_pairing():
    from evaluation.protocol import fingerprint, physical_protocol

    controller, _ = setup_controller()
    enabled = controller.spec
    disabled = deepcopy(enabled)
    disabled["residual"]["enabled"] = False
    kwargs = dict(seed=1200, episodes=8, num_envs=2)
    assert fingerprint(physical_protocol(enabled, **kwargs)) == fingerprint(
        physical_protocol(disabled, **kwargs)
    )


@pytest.mark.simulation
def test_zero_initialized_actor_and_partial_reset(env):
    from algorithms.ppo import make_runner, runner_config
    from tensordict import TensorDict

    cfg = runner_config(ROOT / "configs/algorithms/ppo_residual.yaml")
    runner = make_runner(env, cfg)
    obs, _ = env.reset(seed=301)
    with torch.inference_mode():
        action = runner.alg.get_policy()(TensorDict(obs, batch_size=[env.num_envs]))
    torch.testing.assert_close(action, torch.zeros_like(action), rtol=0, atol=0)
    original = deepcopy(env.benchmark_spec)
    try:
        env.benchmark_spec.update(
            controller={"type": "nominal_pose"}, residual={"max_position_rad": 0.05}
        )
        env.step(torch.ones(env.num_envs, 29))
        torch.testing.assert_close(
            env.previous_residual, torch.full_like(env.previous_residual, 0.1)
        )
        env.reset(env_ids=torch.tensor([0]))
        assert not env.previous_residual[0].any()
        assert env.previous_residual[1].any()
        env.step(torch.ones(env.num_envs, 29))
        torch.testing.assert_close(env.last_residual_change[1], torch.zeros(29))
        torch.testing.assert_close(env.last_residual_change[0], torch.full((29,), 0.1))
    finally:
        env.benchmark_spec.clear()
        env.benchmark_spec.update(original)
