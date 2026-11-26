"""Run real H3/DR/shield/residual/terrain interfaces without claiming task success."""

import argparse
from pathlib import Path


def main():
    """A single variant per process bounds compiler memory and isolates failures."""
    import torch

    from humanoid.config import ROOT, task_config
    from humanoid.environments import make_env
    from humanoid.provenance import write_json
    from humanoid.robots import validate_robot
    from humanoid.native import effective_link_masses

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        choices=["push", "dr", "shield", "residual", "terrain", "transfer", "transfer_push"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    variants = {
        "push": ("H3", "h3_push"),
        "dr": ("H2", "h2_domain_randomized"),
        "shield": ("H2", "h2_shield"),
        "residual": ("H1", "h1_residual"),
        "terrain": ("H4", "h4_terrain"),
        "transfer": ("H1", "h1_transfer_delay"),
        "transfer_push": ("H3", "h3_stiff_pd"),
    }
    task, config = variants[args.variant]
    spec = task_config(task, ROOT / f"configs/tasks/{config}.yaml")
    if args.variant == "transfer_push":
        spec["actuator"] = {"mode": "explicit_pd", "delay_steps": [0, 0]}
        spec["timing"] = {"physics_dt_s": 0.002, "control_dt_s": 0.02, "substeps": 1}
    if args.variant == "transfer":
        spec["randomization"] = {"stiffness_scale": [0.9, 1.1], "damping_scale": [0.9, 1.1]}
    if args.variant in ("push", "transfer_push"):
        spec["push"] = {"start_s": 0.04, "duration_s": 0.04, "force_n": [10.0, 0.0, 0.0]}
    env = make_env(task, num_envs=2, seed=101, evaluation=True, device="cpu", config=spec)
    try:
        robot = validate_robot(env)
        env.reset(seed=101)
        impulse = 0.0
        physics_impulse = 0.0
        for _ in range(8):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(2, 29))
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert torch.isfinite(reward).all()
            impulse += float(env.last_push_force[0, 0]) * env.step_dt
            physics_impulse += float(env.last_push_impulse[0, 0])
            done = terminated | truncated
            if done.any():
                env.reset(env_ids=done.nonzero(as_tuple=False).flatten())
        if args.variant == "transfer":
            assert obs["actor"].shape == (2, 96)
            assert obs["critic"].shape == (2, 109)
            assert env.cfg.decimation == 10
            for actuator in env.scene["robot"].actuators:
                assert torch.isfinite(actuator.applied_effort).all()
                assert actuator.delay.lags.min() >= 0 and actuator.delay.lags.max() <= 5
            cursors = [a.delay.cursor for a in env.scene["robot"].actuators]
            env.scene.write_data_to_sim()
            assert cursors == [a.delay.cursor for a in env.scene["robot"].actuators]
            env.reset(env_ids=torch.tensor([0]))
            assert cursors == [a.delay.cursor for a in env.scene["robot"].actuators]
            assert not torch.equal(env.action_pipeline.kp[0], env.action_pipeline.kp[1])
            obs = env.get_observations()
            from humanoid.deployment import PortableController, control_contract
            from humanoid.provenance import write_json

            write_json(args.output.with_suffix(".contract.json"), control_contract(spec, robot))
            data = env.scene["robot"].data
            portable = PortableController(control_contract(spec, robot))
            state = {
                "q": data.joint_pos,
                "dq": data.joint_vel,
                "gravity": data.projected_gravity_b,
                "angular_velocity": data.root_ang_vel_b,
            }
            actual = portable.observe(
                state, env.command_manager.get_command("base_velocity"), env.action_manager.action
            )
            torch.testing.assert_close(actual, env.get_observations()["actor"])
        entity = env.scene["robot"].gs_entity
        masses = effective_link_masses(entity)
        if args.variant == "dr":
            assert masses.ndim == 2
            assert not torch.equal(masses[0], masses[1]), "Per-env mass randomization collapsed"
            assert env.action_pipeline.kp.ndim == 2
            assert not torch.equal(env.action_pipeline.kp[0], env.action_pipeline.kp[1])
        if args.variant in ("push", "transfer_push"):
            assert abs(impulse - 0.4) < 1e-6
            assert abs(physics_impulse - 0.4) < 1e-6
        write_json(
            args.output,
            {
                "variant": args.variant,
                "control_steps": 8,
                "physics_steps": 8 * env.cfg.decimation,
                "task_success_claimed": False,
                "actor_shape": list(obs["actor"].shape),
                "critic_shape": list(obs["critic"].shape),
                "robot": robot,
                "requested_impulse_x_ns": impulse,
                "physics_accumulated_impulse_x_ns": physics_impulse,
                "mass_by_env_kg": masses.cpu().tolist(),
                "spec": spec,
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
