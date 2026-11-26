"""Run a reproducible sequence of training seeds and paired evaluation conditions."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    """Run serially to bound CPU/GPU memory; save each completed job immediately."""
    from humanoid.config import TASKS, ROOT
    from humanoid.provenance import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), default="H1")
    parser.add_argument("--task-config")
    parser.add_argument("--ppo-config")
    parser.add_argument(
        "--evaluation-config", default=str(ROOT / "configs/evaluation/nominal.yaml")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Training seeds must be unique")
    jobs = []
    for seed in args.seeds:
        root = args.output / f"seed{seed}"
        train = [
            sys.executable,
            "-m",
            "scripts.train",
            "--task",
            args.task,
            "--seed",
            str(seed),
            "--num-envs",
            str(args.num_envs),
            "--device",
            args.device,
            "--output",
            str(root / "train"),
        ]
        for key in ("task_config", "ppo_config", "iterations"):
            value = getattr(args, key)
            if value is not None:
                train += ["--" + key.replace("_", "-"), str(value)]
        evaluate = [
            sys.executable,
            "-m",
            "scripts.evaluate",
            "--task",
            args.task,
            "--checkpoint",
            str(root / "train/model_final.pt"),
            "--episodes",
            str(args.episodes),
            "--num-envs",
            str(args.eval_envs),
            "--seed",
            str(args.eval_seed),
            "--device",
            args.device,
            "--evaluation-config",
            args.evaluation_config,
            "--output",
            str(root / "evaluation"),
        ]
        jobs.extend([(root, "train", train), (root, "evaluation", evaluate)])
    if args.dry_run:
        print(json.dumps([{"phase": phase, "argv": cmd} for _, phase, cmd in jobs], indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for root, phase, cmd in jobs:
        root.mkdir(exist_ok=True)
        with (root / f"{phase}.log").open("w") as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        results.append(
            {
                "seed_directory": str(root),
                "phase": phase,
                "argv": cmd,
                "returncode": completed.returncode,
            }
        )
        write_json(args.output / "jobs.json", results)
        if completed.returncode:
            raise SystemExit(f"Job failed: {root}/{phase}.log; completed runs were preserved")


if __name__ == "__main__":
    main()
