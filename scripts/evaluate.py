"""Evaluate independently, preserving terminal states and complete first episodes."""

import argparse
import json
from pathlib import Path

from evaluation.statistics import summarize
from humanoid.config import task_config, TASKS, read_yaml, validate_task
from humanoid.environments import make_env
from humanoid.provenance import sha256, system_info, write_json, archive_source


def main() -> None:
    """Run fixed-horizon first episodes; reward is not used to define success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), default="H2")
    parser.add_argument("--task-config")
    parser.add_argument(
        "--evaluation-config", help="Common physical test overrides, independent of training"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--agent", choices=["trained", "zero", "feedback"], default="trained")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--viewer", action="store_true", help="Open the live viewer (requires display)"
    )
    parser.add_argument("--video-frames", type=int, default=200)
    parser.add_argument(
        "--video-start-s", type=float, help="Default: one second before a push, otherwise zero"
    )
    args = parser.parse_args()
    if min(args.episodes, args.num_envs, args.video_frames) < 1:
        parser.error("Episode, environment and video frame counts must be positive")
    if args.agent == "trained" and not args.checkpoint:
        parser.error("A trained policy requires --checkpoint")
    if args.agent != "trained" and args.checkpoint:
        parser.error("--checkpoint requires --agent trained")
    import numpy as np
    import torch
    from tensordict import TensorDict

    from evaluation.metrics import measure
    from evaluation.protocol import command_schedule, physical_protocol, fingerprint

    torch.set_num_threads(4)
    spec = task_config(args.task, args.task_config)
    if args.checkpoint:
        spec = json.loads((args.checkpoint.parent / "task.json").read_text())
        if spec["task"] != args.task:
            parser.error("Task does not match checkpoint")
        if args.task_config:
            parser.error(
                "Checkpoint already provides task.json; use --evaluation-config for physical overrides"
            )
    if args.agent == "feedback":
        if spec.get("controller", {}).get("type") != "ankle_feedback":
            parser.error("Feedback agent requires an ankle_feedback controller in --task-config")
        spec["controller"]["residual_scale"] = 0.0
    if args.evaluation_config:
        overrides = read_yaml(args.evaluation_config)
        allowed = {
            "episode_length_s",
            "commands",
            "success",
            "reset_joint_range",
            "randomization",
            "push",
            "terrain",
        }
        if set(overrides) - allowed:
            parser.error("Evaluation overrides may change physical conditions only")
        spec.update(overrides)
    validate_task(spec, args.task)
    video_start = args.video_start_s
    if video_start is None:
        video_start = max(0.0, spec.get("push", {}).get("start_s", 0.0) - 1.0)
    if not np.isfinite(video_start) or not 0 <= video_start < spec["episode_length_s"]:
        parser.error("Video start must be finite and inside the episode")
    args.output.mkdir(parents=True, exist_ok=False)
    source_sha = archive_source(args.output)
    env = None
    episodes, trace, plans = [], [], []
    writer = None
    video_frames_written = 0
    training_seed = None
    try:
        env = make_env(
            args.task,
            num_envs=min(args.num_envs, args.episodes),
            seed=args.seed,
            device=args.device,
            evaluation=True,
            config=spec,
            video=args.video,
            show_viewer=args.viewer,
        )
        if args.checkpoint:
            from algorithms.ppo import make_runner

            cfg = json.loads((args.checkpoint.parent / "runner.json").read_text())
            training_seed = cfg["seed"]
            runner = make_runner(env, cfg)
            runner.load(str(args.checkpoint), map_location=str(env.device))
            policy = runner.get_inference_policy(device=str(env.device))
        else:

            def policy(obs):
                return torch.zeros(env.num_envs, 29, device=env.device)

        if args.video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(args.output / "rollout.mp4"), fps=round(1 / env.step_dt)
            )
        batch = 0
        with torch.inference_mode():
            while len(episodes) < args.episodes:
                count = min(env.num_envs, args.episodes - len(episodes))
                obs, _ = env.reset(seed=args.seed + batch)
                active = torch.arange(env.num_envs, device=env.device) < count
                initial = measure(env, env.command_manager.get_command("base_velocity"))
                initial_com = torch.stack([initial["com_x_m"], initial["com_y_m"]], -1)
                # Independent command RNG prevents policy-dependent termination/reset
                # counts from changing later command sequences in other slots.
                command_interval = env.cfg.commands["base_velocity"].resampling_time_range[0]
                command_steps = max(1, round(command_interval / env.step_dt))
                schedule = command_schedule(
                    spec,
                    seed=args.seed + batch,
                    horizon=env.max_episode_length,
                    num_envs=env.num_envs,
                    command_steps=command_steps,
                )
                plans.append({"seed": args.seed + batch, "commands": schedule.tolist()})
                schedule = torch.tensor(schedule, dtype=torch.float32, device=env.device)
                sums, peaks = {}, {}
                steps = torch.zeros(env.num_envs, device=env.device)
                returns = torch.zeros_like(steps)
                exposed = torch.zeros_like(active)
                settled = torch.zeros_like(steps)
                recovered_at = torch.full_like(steps, -1)
                for step in range(env.max_episode_length):
                    commands = schedule[step // command_steps]
                    env.command_manager.get_term("base_velocity").set_command(commands)
                    obs = env.get_observations()
                    td = TensorDict(obs, batch_size=[env.num_envs])
                    actions = policy(td).clamp(-spec["action_clip"], spec["action_clip"])
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    metrics = measure(env, commands)
                    com_xy = torch.stack([metrics["com_x_m"], metrics["com_y_m"]], -1)
                    metrics["com_displacement_m"] = (com_xy - initial_com).norm(dim=-1)
                    for key, value in metrics.items():
                        if not torch.isfinite(value).all():
                            raise FloatingPointError(f"Non-finite metric: {key}")
                        sums[key] = sums.get(key, torch.zeros_like(value)) + value * active
                        peaks[key] = torch.maximum(
                            peaks.get(key, torch.zeros_like(value)), value * active
                        )
                    steps += active
                    returns += reward * active
                    exposed |= (metrics["external_force_n"] > 0) & active
                    push_end = sum(
                        spec.get("push", {}).get(k, 0) for k in ("start_s", "duration_s")
                    )
                    stable = (metrics["tilt_rad"] <= 0.35) & (metrics["linear_error_m_s"] <= 0.2)
                    settled = torch.where(
                        stable & active & exposed & (steps * env.step_dt >= push_end),
                        settled + 1,
                        torch.zeros_like(settled),
                    )
                    now_recovered = (settled * env.step_dt >= 0.5) & (recovered_at < 0)
                    recovered_at[now_recovered] = (
                        steps[now_recovered] * env.step_dt - push_end - 0.5
                    )
                    if batch == 0 and active[0]:
                        row = {key: float(value[0]) for key, value in metrics.items()}
                        row.update(
                            time_s=(step + 1) * env.step_dt,
                            reward=float(reward[0]),
                            root_pos_m=env.scene["robot"].data.root_pos_w[0].cpu().tolist(),
                            root_quat_wxyz=env.scene["robot"].data.root_quat_w[0].cpu().tolist(),
                            command=commands[0].cpu().tolist(),
                            applied_action=env.last_projection.action[0].cpu().tolist(),
                            joint_pos_rad=env.scene["robot"].data.joint_pos[0].cpu().tolist(),
                            foot_contact=env.scene["feet_contact"]
                            .data.in_contact[0]
                            .cpu()
                            .tolist(),
                        )
                        trace.append(row)
                        start_step = round(video_start / env.step_dt)
                        if (
                            writer is not None
                            and start_step <= step < start_step + args.video_frames
                        ):
                            root = env.scene["robot"].data.root_pos_w[0].cpu().numpy()
                            env.set_render_pose(
                                tuple(root + np.array([2.5, 2.0, 1.5])), tuple(root)
                            )
                            writer.append_data(np.asarray(env.render(), dtype=np.uint8))
                            video_frames_written += 1
                    done = terminated | truncated
                    for slot in (done & active).nonzero(as_tuple=False).flatten().tolist():
                        n = float(steps[slot])
                        row = {
                            "episode": batch * env.num_envs + slot,
                            "seed": args.seed + batch,
                            "slot": slot,
                            "terminated": int(terminated[slot]),
                            "duration_s": n * env.step_dt,
                            "return": float(returns[slot]),
                        }
                        row.update({f"mean_{k}": float(v[slot] / n) for k, v in sums.items()})
                        row.update(
                            {
                                f"max_{k}": float(v[slot])
                                for k, v in peaks.items()
                                if k in ("tilt_rad", "root_displacement_m", "com_displacement_m")
                            }
                        )
                        row["energy_j"] = float(sums["power_w"][slot] * env.step_dt)
                        row["push_exposed"] = int(exposed[slot])
                        row["impulse_magnitude_ns"] = float(
                            sums["external_force_n"][slot] * env.step_dt
                        )
                        row["recovered"] = int(
                            exposed[slot] and recovered_at[slot] >= 0 and not terminated[slot]
                        )
                        row["recovery_time_s"] = (
                            max(0.0, float(recovered_at[slot])) if row["recovered"] else None
                        )
                        passed = bool(truncated[slot]) and not bool(terminated[slot])
                        limits = spec["success"]
                        if args.task in ("H1", "H3"):
                            passed &= row["max_tilt_rad"] <= limits["max_tilt_rad"]
                            passed &= (
                                row["max_root_displacement_m"] <= limits["max_root_displacement_m"]
                            )
                            if args.task == "H3":
                                passed &= bool(row["recovered"])
                        else:
                            passed &= (
                                row["mean_linear_error_m_s"] <= limits["max_mean_linear_error_m_s"]
                            )
                            passed &= (
                                row["mean_yaw_error_rad_s"] <= limits["max_mean_yaw_error_rad_s"]
                            )
                        row["success"] = int(passed)
                        episodes.append(row)
                    active &= ~done
                    if not active.any():
                        break
                    if done.any():
                        # Reset completed slots, but never count their later episodes.
                        env.reset(env_ids=done.nonzero(as_tuple=False).flatten())
                        obs = env.get_observations()
                if active.any():
                    raise RuntimeError("Time limit failed to terminate every active episode")
                batch += 1
        episodes.sort(key=lambda row: row["episode"])
        write_json(args.output / "episodes.json", episodes)
        write_json(args.output / "trace.json", trace)
        write_json(args.output / "summary.json", summarize(episodes))
        write_json(args.output / "commands.json", plans)
        protocol = physical_protocol(
            spec, seed=args.seed, num_envs=env.num_envs, episodes=args.episodes
        )
        write_json(
            args.output / "manifest.json",
            {
                "task": spec,
                "system": system_info(),
                "evaluation_seed": args.seed,
                "num_envs": env.num_envs,
                "agent": args.agent,
                "training_seed": training_seed,
                "video_start_s": video_start,
                "video_frames": video_frames_written,
                "source_snapshot_sha256": source_sha,
                "protocol": protocol,
                "protocol_sha256": fingerprint(protocol),
                "command_plan_sha256": fingerprint(plans),
                "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "note": "First full episode per assigned slot; terminal states sampled before reset.",
            },
        )
        print(json.dumps(summarize(episodes), indent=2))
        write_json(
            args.output / "status.json", {"evaluation_completed": True, "episodes": len(episodes)}
        )
    except BaseException as error:
        write_json(
            args.output / "status.json",
            {
                "evaluation_completed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
