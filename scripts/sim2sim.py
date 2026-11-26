"""Evaluate a flat-task control contract in real, headless MuJoCo physics."""

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def load_model(path, contract):
    """Keep the pinned morphology; add ground and use effort-only actuation."""
    import mujoco

    from humanoid.config import ROOT, read_yaml
    from humanoid.provenance import sha256

    if sha256(path) != contract["asset_xml_sha256"]:
        raise ValueError("MJCF hash differs from the audited training asset")
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(path.parent / "assets"))
    for tag in ("actuator", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(contract["actuator_dt_s"]))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    ET.SubElement(
        root.find("worldbody"),
        "geom",
        name="benchmark_ground",
        type="plane",
        size="0 0 0.1",
        friction="1 0.005 0.0001",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nq != 36 or model.nv != 35 or model.nu:
        raise ValueError("Expected floating G1 with 29 scalar joints and no hidden actuators")
    joints = read_yaml(ROOT / "configs/robots/g1_29dof.yaml")["joints"]
    for joint in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint["name"])
        if jid < 0:
            raise ValueError(f"Missing joint: {joint['name']}")
        dof = model.jnt_dofadr[jid]
        model.dof_armature[dof] = joint["armature_kg_m2"]
    return model


def main():
    """No automatic retuning, terrain substitution, or claim of hardware transfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bundle", type=Path, help="Export directory with policy.pt and contract.json"
    )
    source.add_argument("--task-config", type=Path, help="Zero-residual PD reference")
    parser.add_argument("--mjcf", type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    import mujoco
    import numpy as np
    import torch

    from evaluation.protocol import command_schedule, fingerprint
    from humanoid.config import task_config
    from humanoid.deployment import PortableController, control_contract
    from humanoid.provenance import archive_source, sha256, write_json
    from humanoid.transfer import CommandDelay, sampled_pd

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.bundle:
        contract = json.loads((args.bundle / "contract.json").read_text())
        if sha256(args.bundle / "policy.pt") != contract["policy_sha256"]:
            raise ValueError("Policy hash differs from exported contract")
        policy = torch.jit.load(str(args.bundle / "policy.pt"), map_location="cpu").eval()
    else:
        from humanoid.config import read_yaml

        cfg = read_yaml(args.task_config)
        contract = control_contract(task_config(cfg["task"], args.task_config))
        policy = lambda obs: torch.zeros(1, 29)  # noqa: E731
    controller = PortableController(contract)
    spec, values = contract["task"], controller.values
    if spec.get("actuator", {}).get("mode", "implicit") != "explicit_pd":
        parser.error(
            "MuJoCo adapter requires an explicit_pd contract; implicit equivalence is undefined"
        )
    if spec.get("randomization"):
        parser.error("MuJoCo adapter does not yet implement physical DR; use a nominal config")
    cache = Path(os.environ.get("GENESISLAB_ASSET_CACHE", Path.home() / ".cache/genesislab/assets"))
    path = args.mjcf or cache / "unitree_g1/g1.xml"
    model = load_model(path, contract)
    joint_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in controller.robot["joint_names"]
        ]
    )
    qids, vids = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso < 0:
        raise ValueError("Missing torso_link for force protocol")
    dt, period = contract["actuator_dt_s"], contract["control_dt_s"]
    decimation = round(period / dt)
    horizon = int(np.ceil(spec["episode_length_s"] / period))
    actuator_spec = spec.get("actuator", {})
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, traces, plans = [], [], []
    try:
        source_hash = archive_source(args.output)
        with torch.inference_mode():
            for episode in range(args.episodes):
                data = mujoco.MjData(model)
                data.qpos[:7] = [0, 0, 0.76, 1, 0, 0, 0]
                reset_range = spec.get("reset_joint_range", [-0.1, 0.1])
                data.qpos[qids] = values["home_rad"][0].numpy() + rng.uniform(*reset_range, 29)
                mujoco.mj_forward(model, data)
                delay = CommandDelay(
                    values["home_rad"], tuple(actuator_spec.get("delay_steps", [0, 0]))
                )
                last = torch.zeros(1, 29)
                schedule = command_schedule(
                    spec,
                    seed=args.seed + episode,
                    horizon=horizon,
                    num_envs=1,
                    command_steps=max(1, round(4 / period)),
                )
                plans.append(schedule.tolist())
                peak_tilt, peak_distance, energy, saturation, sample_count = 0.0, 0.0, 0.0, 0.0, 0
                error_sum, yaw_sum, exposed, impulse = 0.0, 0.0, False, 0.0
                recovery_hold, recovery_time = 0.0, None
                terminated = False

                def state():
                    quat = data.qpos[3:7].copy()
                    inverse = quat * np.array([1, -1, -1, -1])

                    def rotate(vector):
                        result = np.zeros(3)
                        mujoco.mju_rotVecQuat(result, np.asarray(vector, dtype=float), inverse)
                        return torch.tensor(result, dtype=torch.float32).unsqueeze(0)

                    return {
                        "q": torch.tensor(data.qpos[qids], dtype=torch.float32).unsqueeze(0),
                        "dq": torch.tensor(data.qvel[vids], dtype=torch.float32).unsqueeze(0),
                        "angular_velocity": torch.tensor(
                            data.qvel[3:6], dtype=torch.float32
                        ).unsqueeze(0),
                        "linear_velocity": rotate(data.qvel[:3]),
                        "gravity": rotate([0, 0, -1]),
                    }

                for step in range(horizon):
                    sensed = state()
                    command = torch.tensor(
                        schedule[step // max(1, round(4 / period))], dtype=torch.float32
                    )
                    obs = controller.observe(sensed, command, last)
                    if obs.shape[-1] != contract["observation_dim"]:
                        raise ValueError("Adapter observation does not match exported actor")
                    raw = policy(obs).clamp(-spec["action_clip"], spec["action_clip"])
                    projection = controller.process(raw, sensed)
                    last = projection.action
                    target = values["home_rad"] + spec["action_scale"] * last
                    for substep in range(decimation):
                        now = state()
                        requested, torque = sampled_pd(
                            delay(target),
                            now["q"],
                            now["dq"],
                            values["kp"],
                            values["kd"],
                            values["effort_limit_nm"],
                            no_load_speed=actuator_spec.get("no_load_speed_rad_s"),
                        )
                        data.qfrc_applied[:] = 0
                        data.qfrc_applied[vids] = torque[0].numpy()
                        data.xfrc_applied[:] = 0
                        push = spec.get("push")
                        if (
                            push
                            and push["start_s"] <= data.time < push["start_s"] + push["duration_s"]
                        ):
                            data.xfrc_applied[torso, :3] = push["force_n"]
                            exposed = True
                            impulse += float(np.linalg.norm(push["force_n"])) * dt
                        energy += float((torque * now["dq"]).abs().sum()) * dt
                        saturation += float((requested.abs() > torque.abs() + 1e-5).float().mean())
                        sample_count += 1
                        mujoco.mj_step(model, data)
                        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                            raise FloatingPointError("Non-finite MuJoCo state")
                    after = state()
                    tilt = float(torch.acos((-after["gravity"][0, 2]).clamp(-1, 1)))
                    distance = float(np.linalg.norm(data.qpos[:2]))
                    error = float((after["linear_velocity"][0, :2] - command[0, :2]).norm())
                    yaw = float(abs(after["angular_velocity"][0, 2] - command[0, 2]))
                    peak_tilt, peak_distance = max(peak_tilt, tilt), max(peak_distance, distance)
                    error_sum, yaw_sum = error_sum + error, yaw_sum + yaw
                    push_end = sum(
                        spec.get("push", {}).get(k, 0) for k in ("start_s", "duration_s")
                    )
                    recovery_hold = (
                        recovery_hold + period
                        if exposed and data.time >= push_end and tilt <= 0.35 and error <= 0.2
                        else 0.0
                    )
                    if recovery_hold >= 0.5 and recovery_time is None:
                        recovery_time = max(0.0, data.time - push_end - 0.5)
                    if episode == 0:
                        traces.append(
                            {
                                "time_s": float(data.time),
                                "tilt_rad": tilt,
                                "root_displacement_m": distance,
                                "linear_error_m_s": error,
                                "applied_action": last[0].tolist(),
                            }
                        )
                    terminated = tilt > 1.0 or data.qpos[2] < 0.4
                    if terminated:
                        break
                count = step + 1
                passed = not terminated and count == horizon
                limits = spec["success"]
                if spec["task"] in ("H1", "H3"):
                    passed &= peak_tilt <= limits["max_tilt_rad"]
                    passed &= peak_distance <= limits["max_root_displacement_m"]
                    if spec["task"] == "H3":
                        passed &= exposed and recovery_time is not None
                else:
                    passed &= error_sum / count <= limits["max_mean_linear_error_m_s"]
                    passed &= yaw_sum / count <= limits["max_mean_yaw_error_rad_s"]
                rows.append(
                    {
                        "episode": episode,
                        "success": int(passed),
                        "terminated": int(terminated),
                        "duration_s": float(data.time),
                        "max_tilt_rad": peak_tilt,
                        "max_root_displacement_m": peak_distance,
                        "energy_j": energy,
                        "mean_linear_error_m_s": error_sum / count,
                        "mean_yaw_error_rad_s": yaw_sum / count,
                        "mean_actuator_saturation": saturation / sample_count,
                        "push_exposed": exposed,
                        "impulse_magnitude_ns": impulse,
                        "recovery_time_s": recovery_time,
                    }
                )
        write_json(args.output / "episodes.json", rows)
        write_json(args.output / "trace.json", traces)
        write_json(args.output / "commands.json", plans)
        write_json(
            args.output / "manifest.json",
            {
                "engine": "mujoco",
                "engine_version": mujoco.__version__,
                "seed": args.seed,
                "contract": contract,
                "contract_sha256": fingerprint(contract),
                "mjcf_sha256": sha256(path),
                "source_snapshot_sha256": source_hash,
                "integrator": "implicitfast",
                "no_hardware_transfer_claim": True,
                "limitations": [
                    "Contact solvers and defaults differ from Genesis",
                    "Sequential per-episode resets; not paired Genesis RNG states",
                    "No physical randomization or sensor latency model",
                ],
            },
        )
        summary = {
            "episodes": len(rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "mean_duration_s": sum(r["duration_s"] for r in rows) / len(rows),
            "hardware_validated": False,
        }
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "status.json", {"evaluation_completed": True})
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        write_json(
            args.output / "status.json", {"evaluation_completed": False, "error": str(error)}
        )
        raise


if __name__ == "__main__":
    main()
