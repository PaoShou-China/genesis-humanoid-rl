"""Search interpretable ankle-feedback controllers on a validation-only grid."""

import argparse
import itertools
from pathlib import Path


def main():
    """Keep every candidate's first episode; never use held-out test seeds here."""
    import torch
    from genesislab.envs import ManagerBasedRLEnv

    from humanoid.environments import make_env_cfg
    from humanoid.provenance import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["feedback", "gains"], default="feedback")
    args = parser.parse_args()
    torch.set_num_threads(4)
    grid = (
        [(factor, factor**0.5, 0.0) for factor in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)]
        if args.mode == "gains"
        else list(itertools.product([0.0, 0.5, 1.5], [0.05, 0.2], [0.04, 0.12, 0.20]))
    )
    cfg = make_env_cfg("H1", num_envs=len(grid), seed=100, evaluation=True)
    cfg.events.pop("reset_joints")  # Nominal-pose diagnosis; not robustness evidence.
    if args.mode == "gains":
        cfg.sim.rigid.batch_dofs_info = True
    env = ManagerBasedRLEnv(cfg, device="cpu")
    gains = torch.tensor(grid)
    steps = torch.zeros(len(grid), dtype=torch.long)
    active = torch.ones(len(grid), dtype=torch.bool)
    peak = torch.zeros(len(grid))
    try:
        env.reset(seed=100)
        data = env.scene["robot"].data
        names = env.scene["robot"].joint_names
        if args.mode == "gains":
            entity, ids = env.scene["robot"].gs_entity, env.scene["robot"].indexing.joint_dofs_idx
            entity.set_dofs_kp(entity.get_dofs_kp(ids) * gains[:, 0, None], ids)
            entity.set_dofs_kv(entity.get_dofs_kv(ids) * gains[:, 1, None], ids)
        with torch.inference_mode():
            for step in range(env.max_episode_length):
                g = data.projected_gravity_b
                pitch = torch.atan2(g[:, 0], -g[:, 2])
                roll = torch.atan2(-g[:, 1], -g[:, 2])
                w = data.root_ang_vel_b
                delta = torch.zeros(len(grid), 29)
                for side in ("left", "right"):
                    delta[:, names.index(f"{side}_ankle_pitch_joint")] = (
                        gains[:, 0] * pitch + gains[:, 1] * w[:, 1] + gains[:, 2]
                    ).clamp(-0.4, 0.4)
                    delta[:, names.index(f"{side}_ankle_roll_joint")] = (
                        gains[:, 0] * roll + gains[:, 1] * w[:, 0]
                    ).clamp(-0.2, 0.2)
                    delta[:, names.index(f"{side}_knee_joint")] = -0.1
                if args.mode == "gains":
                    delta.zero_()
                _, _, term, trunc, _ = env.step(delta / 0.5)
                steps += active
                tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))
                peak = torch.maximum(peak, tilt * active)
                done = term | trunc
                active &= ~done
                if not active.any():
                    break
                if done.any():
                    env.reset(env_ids=done.nonzero(as_tuple=False).flatten())
                if step % 100 == 0:
                    print(f"step={step} surviving={int(active.sum())}", flush=True)
        rows = [
            {
                "kp": kp,
                "kd": kd,
                "bias": bias,
                "survival_s": float(steps[i]) * env.step_dt,
                "max_tilt_rad": float(peak[i]),
            }
            for i, (kp, kd, bias) in enumerate(grid)
        ]
        rows.sort(key=lambda row: (-row["survival_s"], row["max_tilt_rad"]))
        write_json(
            args.output, {"seed": 100, "mode": args.mode, "nominal_reset": True, "candidates": rows}
        )
        print(rows[:5], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
