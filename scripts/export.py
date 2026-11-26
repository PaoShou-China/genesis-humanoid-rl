"""Export an actual trained policy, including observation normalization."""
import argparse
import json
from pathlib import Path


def main() -> None:
    """Load the checkpoint with its recorded network configuration, then export."""
    from algorithms.ppo import make_runner
    from humanoid.environments import make_env
    from humanoid.provenance import sha256, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    task = json.loads((args.checkpoint.parent / "task.json").read_text())
    cfg = json.loads((args.checkpoint.parent / "runner.json").read_text())
    env = make_env(task["task"], config=task, device=args.device, num_envs=1, evaluation=True)
    try:
        runner = make_runner(env, cfg)
        runner.load(str(args.checkpoint), map_location=str(env.device))
        args.output.mkdir(parents=True, exist_ok=False)
        runner.export_policy_to_jit(str(args.output), filename="policy.pt")
        write_json(args.output / "metadata.json", {"task": task, "checkpoint_sha256": sha256(args.checkpoint),
                   "action_clip": cfg["clip_actions"], "observation_group": "actor",
                   "joint_manifest": json.loads((args.checkpoint.parent / "robot.json").read_text())})
    finally:
        env.close()


if __name__ == "__main__":
    main()
