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
    parser.add_argument("variant", choices=["push", "dr", "shield", "residual", "terrain"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    variants = {
        "push": ("H3", "h3_push"),
        "dr": ("H2", "h2_domain_randomized"),
        "shield": ("H2", "h2_shield"),
        "residual": ("H1", "h1_residual"),
        "terrain": ("H4", "h4_terrain"),
    }
    task, config = variants[args.variant]
    spec = task_config(task, ROOT / f"configs/tasks/{config}.yaml")
    if args.variant == "push":
        spec["push"] = {"start_s": 0.04, "duration_s": 0.04, "force_n": [10.0, 0.0, 0.0]}
    env = make_env(task, num_envs=2, seed=101, evaluation=True, device="cpu", config=spec)
    try:
        robot = validate_robot(env)
        env.reset(seed=101)
        impulse = 0.0
        for _ in range(8):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(2, 29))
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert torch.isfinite(reward).all()
            impulse += float(env.last_push_force[0, 0]) * env.step_dt
            done = terminated | truncated
            if done.any():
                env.reset(env_ids=done.nonzero(as_tuple=False).flatten())
        entity = env.scene["robot"].gs_entity
        masses = effective_link_masses(entity)
        if args.variant == "dr":
            assert masses.ndim == 2
            assert not torch.equal(masses[0], masses[1]), "Per-env mass randomization collapsed"
            assert env.action_pipeline.kp.ndim == 2
            assert not torch.equal(env.action_pipeline.kp[0], env.action_pipeline.kp[1])
        if args.variant == "push":
            assert abs(impulse - 0.4) < 1e-6
        write_json(
            args.output,
            {
                "variant": args.variant,
                "physics_steps": 8,
                "task_success_claimed": False,
                "actor_shape": list(obs["actor"].shape),
                "critic_shape": list(obs["critic"].shape),
                "robot": robot,
                "requested_impulse_x_ns": impulse,
                "mass_by_env_kg": masses.cpu().tolist(),
                "spec": spec,
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
