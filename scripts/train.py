"""Train or resume actual RSL-RL PPO on GenesisLab H1/H2."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from algorithms.ppo import make_runner, runner_config
from humanoid.config import task_config
from humanoid.environments import make_env
from humanoid.provenance import system_info, write_json
from humanoid.robots import validate_robot


def main() -> None:
    """Create a uniquely named run with resolved configuration and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["H1", "H2"], default="H2")
    parser.add_argument("--task-config")
    parser.add_argument("--ppo-config")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu for a small smoke run.")
    spec = task_config(args.task, args.task_config)
    cfg = runner_config(args.ppo_config, seed=args.seed)
    if cfg["clip_actions"] != spec["action_clip"]:
        parser.error("Runner and task action clipping must agree")
    if args.iterations is not None:
        if args.iterations < 1:
            parser.error("iterations must be positive")
        cfg["max_iterations"] = args.iterations
    if args.resume:
        previous = json.loads((args.resume.parent / "task.json").read_text())
        if previous != spec:
            parser.error("Resume task configuration differs from checkpoint task.json")
        prior_runner = json.loads((args.resume.parent / "runner.json").read_text())
        canonical_cfg = json.loads(json.dumps(cfg))
        for key in ("actor", "critic", "algorithm", "clip_actions"):
            if prior_runner[key] != canonical_cfg[key]:
                parser.error(f"Resume runner configuration differs: {key}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output or Path("logs") / args.task / f"seed{args.seed}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "task.json", spec)
    write_json(out / "runner.json", cfg)
    write_json(out / "system.json", system_info())
    write_json(out / "invocation.json", vars(args) | {"resume": str(args.resume), "output": str(out)})
    env = make_env(args.task, device=args.device, num_envs=args.num_envs, seed=args.seed, config=spec)
    try:
        write_json(out / "robot.json", validate_robot(env))
        runner = make_runner(env, cfg, str(out))
        if args.resume:
            runner.load(str(args.resume), map_location=str(env.device))
        runner.learn(num_learning_iterations=cfg["max_iterations"], init_at_random_ep_len=True)
        # Always save an explicit final checkpoint, including optimizer state.
        runner.save(str(out / "model_final.pt"))
        write_json(out / "status.json", {"training_completed": True,
            "additional_iterations_requested": cfg["max_iterations"],
            "convergence_claimed": False, "resume": str(args.resume) if args.resume else None})
        print(f"RUN_DIR={out.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
