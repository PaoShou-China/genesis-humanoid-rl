"""Materialize a bounded PD transfer diagnosis, without launching long runs."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys


def main():
    import yaml

    from humanoid.config import task_config, validate_task
    from humanoid.provenance import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["H1", "H3"], default="H1")
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = task_config(args.task)
    rows = []
    for gain in (8.0, 16.0):
        for dt_ms in (2, 5, 10):
            for delay_ms in (0, 10, 20):
                spec = deepcopy(base)
                spec["pd_gains"] = {"stiffness_scale": gain, "damping_scale": gain**0.5}
                spec["actuator"] = {"mode": "explicit_pd", "delay_steps": [delay_ms // dt_ms] * 2}
                spec["timing"] = {"physics_dt_s": dt_ms / 1000, "control_dt_s": 0.02, "substeps": 1}
                spec["actor_observation"] = "proprioceptive"
                validate_task(spec)
                name = f"kp{int(gain)}_dt{dt_ms}ms_delay{delay_ms}ms"
                config = output / f"{name}.yaml"
                config.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
                common = ["--episodes", str(args.episodes), "--seed", str(args.seed)]
                for engine in ("genesis", "mujoco"):
                    command = [sys.executable, "-m", "humanoid"]
                    command += (
                        [
                            "evaluate",
                            "--task",
                            args.task,
                            "--agent",
                            "zero",
                            "--num-envs",
                            "2",
                            "--device",
                            "cpu",
                        ]
                        if engine == "genesis"
                        else ["sim2sim"]
                    )
                    command += [
                        "--task-config",
                        str(config),
                        *common,
                        "--output",
                        str(output / f"{name}_{engine}"),
                    ]
                    rows.append(
                        {
                            "engine": engine,
                            "kp_scale": gain,
                            "dt_ms": dt_ms,
                            "delay_ms": delay_ms,
                            "command": command,
                        }
                    )
    write_json(
        output / "plan.json",
        {
            "purpose": "Development diagnosis; not a sealed held-out test or policy training",
            "agent": "zero pose residual; calibrated PD",
            "hardware_validated": False,
            "matching": "Same configured controller; engine reset RNG and contact semantics differ",
            "runs": rows,
        },
    )
    print(f"Prepared {len(rows)} runs; review plan.json command arrays before execution")


if __name__ == "__main__":
    main()
