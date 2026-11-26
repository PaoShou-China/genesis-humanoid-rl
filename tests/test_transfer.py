"""Check delay isolation, deployment parity, and actuator boundary behavior."""

from copy import deepcopy

import pytest
import torch

from humanoid.config import task_config, validate_task
from humanoid.deployment import PortableController, control_contract
from humanoid.transfer import CommandDelay, sampled_pd
from evaluation.protocol import physical_protocol, fingerprint


def test_transport_delay_and_partial_reset_do_not_leak_episodes():
    delay = CommandDelay(torch.zeros(2, 1), (2, 2))
    assert delay(torch.tensor([[1.0], [10.0]])).tolist() == [[0.0], [0.0]]
    assert delay(torch.tensor([[2.0], [20.0]])).tolist() == [[0.0], [0.0]]
    delay.reset(torch.tensor([[100.0], [0.0]]), torch.tensor([0]))
    assert delay(torch.tensor([[3.0], [30.0]])).tolist() == [[100.0], [10.0]]
    assert delay(torch.tensor([[4.0], [40.0]])).tolist() == [[100.0], [20.0]]
    assert delay(torch.tensor([[5.0], [50.0]])).tolist() == [[3.0], [30.0]]


def test_zero_delay_is_current_command_and_speed_envelope_bounds_effort():
    command = torch.tensor([[1.0, -1.0]])
    torch.testing.assert_close(CommandDelay(command * 0)(command), command)
    requested, effort = sampled_pd(
        command, 0.0, torch.tensor([[5.0, -10.0]]), 10.0, 0.0, 2.0, no_load_speed=10.0
    )
    torch.testing.assert_close(requested, torch.tensor([[10.0, -10.0]]))
    torch.testing.assert_close(effort, torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize(
    "change",
    [
        {"actuator": {"mode": "implicit", "delay_steps": [0, 1]}},
        {"actuator": {"mode": "explicit_pd", "delay_steps": [0.0, 1.0]}},
        {"actuator": {"mode": "explicit_pd", "delay_steps": [-1, 1]}},
        {"actuator": {"mode": "explicit_pd", "no_load_speed_rad_s": 0}},
        {"timing": {"physics_dt_s": 0.003, "control_dt_s": 0.02}},
        {"timing": {"physics_dt_s": float("nan")}},
        {"actor_observation": "imaginary_sensor"},
    ],
)
def test_invalid_transfer_configs_fail_before_scene_creation(change):
    spec = task_config("H1")
    spec.update(change)
    with pytest.raises(ValueError):
        validate_task(spec)


def test_transfer_conditions_change_pairing_identity():
    spec = task_config("H1")
    changed = deepcopy(spec)
    changed["actuator"] = {"mode": "explicit_pd", "delay_steps": [0, 2]}
    kwargs = dict(seed=1, num_envs=2, episodes=8)
    assert fingerprint(physical_protocol(spec, **kwargs)) != fingerprint(
        physical_protocol(changed, **kwargs)
    )


def test_contract_uses_nominal_gains_not_random_training_sample():
    spec = task_config("H1")
    spec["randomization"] = {"stiffness_scale": [0.8, 1.2]}
    reference = control_contract(spec)
    sample = deepcopy(reference["robot"])
    sample["kp"] = [x * 0.8 for x in sample["kp"]]
    assert control_contract(spec, sample)["robot"]["kp"] == reference["robot"]["kp"]


def test_incomplete_matrix_never_accepts_a_gain(tmp_path, monkeypatch):
    import json
    import sys

    from humanoid.provenance import write_json
    from scripts.transfer_summary import main

    first, second = tmp_path / "first", tmp_path / "second"
    runs = [
        {
            "engine": engine,
            "kp_scale": 16.0,
            "dt_ms": 2,
            "delay_ms": 0,
            "command": ["unused", str(folder)],
        }
        for engine, folder in (("genesis", first), ("mujoco", second))
    ]
    plan, output = tmp_path / "plan.json", tmp_path / "summary.json"
    write_json(plan, {"runs": runs})
    write_json(first / "status.json", {"evaluation_completed": True})
    write_json(first / "summary.json", {"success_rate": 1.0})
    monkeypatch.setattr(
        sys, "argv", ["transfer_summary", "--plan", str(plan), "--output", str(output)]
    )
    main()
    assert json.loads(output.read_text())["empirically_accepted_gain_candidates"] == []
    write_json(second / "status.json", {"evaluation_completed": True})
    write_json(second / "summary.json", {"success_rate": 0.5})
    main()
    assert json.loads(output.read_text())["empirically_accepted_gain_candidates"] == []
    write_json(second / "summary.json", {"success_rate": 1.0})
    main()
    assert json.loads(output.read_text())["empirically_accepted_gain_candidates"] == [16.0]


@pytest.mark.simulation
def test_portable_observation_and_action_match_genesis(env):
    env.reset(seed=202)
    data = env.scene["robot"].data
    portable = PortableController(control_contract(env.benchmark_spec))
    state = {
        "q": data.joint_pos,
        "dq": data.joint_vel,
        "gravity": data.projected_gravity_b,
        "angular_velocity": data.root_ang_vel_b,
        "linear_velocity": data.root_lin_vel_b,
    }
    commands = env.command_manager.get_command("base_velocity")
    obs = portable.observe(state, commands, env.action_manager.action)
    torch.testing.assert_close(obs, env.get_observations()["actor"])
    action = torch.rand(env.num_envs, 29) * 2 - 1
    torch.testing.assert_close(
        portable.process(action, state).action, env.action_pipeline(action).action
    )
