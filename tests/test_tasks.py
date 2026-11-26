"""MDP contracts and success predicates, including tempting false positives."""

from types import SimpleNamespace

import pytest
import torch

from humanoid.config import TASKS, task_config, validate_task
from humanoid.task_objectives import TaskObjective, wrap_angle


def objective(task):
    data = SimpleNamespace(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.74], [0.0, 0.0, 0.74]]),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        root_lin_vel_w=torch.zeros(2, 3),
        joint_pos=torch.zeros(2, 29),
        joint_vel=torch.zeros(2, 29),
    )
    scene = type("Scene", (dict,), {})(
        robot=SimpleNamespace(data=data),
        feet_contact=SimpleNamespace(
            data=SimpleNamespace(in_contact=torch.ones(2, 2, dtype=torch.bool))
        ),
    )
    scene.env_origins = torch.zeros(2, 3)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        scene=scene,
    )
    return TaskObjective(env, task_config(task)), env, data


@pytest.mark.parametrize("task", list(TASKS))
def test_registered_task_configs_are_complete(task):
    spec = task_config(task)
    assert validate_task(spec)["task"] == task


def test_heading_wrap_and_no_success_time_at_reset():
    torch.testing.assert_close(
        wrap_angle(torch.tensor([2 * torch.pi - 0.02])), torch.tensor([-0.02])
    )
    obj, env, _ = objective("H6")
    obj.update()
    assert not obj.hold.any()
    for _ in range(25):
        env.episode_length_buf += 1
        obj.update()
    assert obj.achieved.all()
    obj.goal[:, 0] = 2.0
    env.episode_length_buf += 1
    obj.update()
    assert not obj.achieved.any()  # Heading must still be held at episode end.


def test_getup_cannot_pass_while_crouched_below_required_height():
    obj, env, data = objective("H10")
    data.root_pos_w[:, 2] = 0.5
    for _ in range(60):
        env.episode_length_buf += 1
        obj.update()
    assert not obj.achieved.any()
    data.root_pos_w[:, 2] = 0.66
    for _ in range(50):
        env.episode_length_buf += 1
        obj.update()
    assert obj.achieved.all()


def test_jump_requires_flight_then_landing_not_just_standing():
    obj, env, data = objective("H11")
    env.episode_length_buf[:] = 100
    for _ in range(30):
        env.episode_length_buf += 1
        obj.update()
    assert not obj.achieved.any()
    data.root_pos_w[:, 2] = 0.9
    env.scene["feet_contact"].data.in_contact[:] = False
    for _ in range(2):
        env.episode_length_buf += 1
        obj.update()
    assert obj.airborne.all()
    assert not obj.achieved.any()
    data.root_pos_w[:, 2] = 0.74
    env.scene["feet_contact"].data.in_contact[:] = True
    for _ in range(25):
        env.episode_length_buf += 1
        obj.update()
    assert obj.achieved.all()


def test_goal_partial_reset_does_not_change_other_slots():
    obj, _, _ = objective("H6")
    obj.goal[1, 0], obj.hold[1], obj.achieved[1] = 1.0, 0.5, True
    obj.reset(torch.tensor([0]))
    assert obj.goal[1, 0] == 1.0 and obj.hold[1] == 0.5 and obj.achieved[1]


def test_goal_tasks_refuse_incorrect_flat_deployment_adapter():
    from humanoid.deployment import control_contract

    with pytest.raises(ValueError, match="dedicated deployment"):
        control_contract(task_config("H6"))


def test_unknown_or_mismatched_objective_is_rejected():
    spec = task_config("H6")
    spec["objective"]["type"] = "jump"
    with pytest.raises(ValueError):
        validate_task(spec)


def test_goal_reward_width_is_not_a_physical_condition():
    from copy import deepcopy
    from evaluation.protocol import fingerprint, physical_protocol

    spec = task_config("H6")
    other = deepcopy(spec)
    other["objective"]["reward_width"] = 0.9
    kwargs = dict(seed=1, episodes=8, num_envs=2)
    assert fingerprint(physical_protocol(spec, **kwargs)) == fingerprint(
        physical_protocol(other, **kwargs)
    )
    other["objective"]["heading_range_rad"] = [-1.0, 1.0]
    assert fingerprint(physical_protocol(spec, **kwargs)) != fingerprint(
        physical_protocol(other, **kwargs)
    )
