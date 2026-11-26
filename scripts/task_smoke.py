"""Run a selected task's real reset/step/PPO smoke without claiming proficiency."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppo", action="store_true")
    args = parser.parse_args()
    import torch

    from humanoid.config import ROOT, task_config
    from humanoid.environments import make_env
    from humanoid.provenance import write_json, archive_source
    from humanoid.robots import validate_robot
    from evaluation.metrics import measure

    torch.set_num_threads(4)
    spec = task_config(args.task, args.task_config)
    args.output.mkdir(parents=True, exist_ok=False)
    archive_source(args.output)
    env = None
    try:
        env = make_env(args.task, config=spec, device="cpu", num_envs=2, evaluation=not args.ppo)
        robot = validate_robot(env)
        obs, _ = env.reset(seed=1500)
        initial_root_height = env.scene["robot"].data.root_pos_w[:, 2].clone()
        for _ in range(8):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(2, 29))
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(reward).all()
            for value in measure(env, env.command_manager.get_command("base_velocity")).values():
                assert torch.isfinite(value).all()
            done = terminated | truncated
            if done.any():
                env.reset(env_ids=done.nonzero(as_tuple=False).flatten())
        if hasattr(env, "task_objective"):
            saved_goal = env.task_objective.goal[1].clone()
            env.reset(env_ids=torch.tensor([0]))
            torch.testing.assert_close(saved_goal, env.task_objective.goal[1])
            assert not env.task_objective.achieved[0]
        if args.ppo:
            from algorithms.ppo import make_runner, runner_config

            cfg = runner_config(ROOT / "configs/algorithms/ppo_smoke.yaml")
            runner = make_runner(env, cfg, str(args.output))
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
            runner.save(str(args.output / "model_smoke.pt"))
        write_json(
            args.output / "result.json",
            {
                "task": args.task,
                "spec": spec,
                "robot": robot,
                "actor_shape": list(obs["actor"].shape),
                "critic_shape": list(obs["critic"].shape),
                "control_steps": 8,
                "ppo_iterations": 2 if args.ppo else 0,
                "initial_root_height_m": initial_root_height.tolist(),
                "task_success_claimed": False,
            },
        )
        write_json(args.output / "status.json", {"smoke_completed": True})
    except BaseException as error:
        write_json(args.output / "status.json", {"smoke_completed": False, "error": str(error)})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
