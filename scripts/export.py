"""Export an actual trained policy, including observation normalization."""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Export the recorded RSL-RL MLP directly; no physics scene is needed."""
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    from humanoid.provenance import sha256, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    task = json.loads((args.checkpoint.parent / "task.json").read_text())
    cfg = json.loads((args.checkpoint.parent / "runner.json").read_text())
    torch.set_num_threads(4)
    actor_cfg = cfg["actor"]
    if actor_cfg.pop("class_name") != "MLPModel" or cfg["obs_groups"]["actor"] != ["actor"]:
        parser.error("Exporter currently supports the benchmark's single-group MLP actor only")
    actor_cfg.pop("cnn_cfg", None)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = saved["actor_state_dict"]
    obs_dim = state["mlp.0.weight"].shape[1]
    robot = json.loads((args.checkpoint.parent / "robot.json").read_text())
    example = torch.zeros(2, obs_dim, device=args.device)
    obs = TensorDict({"actor": example}, batch_size=[2])
    actor = MLPModel(obs, cfg["obs_groups"], "actor", robot["num_joints"], **actor_cfg)
    actor.to(args.device).load_state_dict(state, strict=True)
    actor.eval()
    exported = torch.jit.script(actor.as_jit().cpu())
    args.output.mkdir(parents=True, exist_ok=False)
    exported.save(str(args.output / "policy.pt"))
    # Verify the actual saved artifact, not just the in-memory module.
    loaded = torch.jit.load(str(args.output / "policy.pt"))
    with torch.inference_mode():
        expected = actor(obs).cpu()
        actual = loaded(example.cpu())
        torch.testing.assert_close(actual, expected)
    contract_available = task.get("terrain", "flat") == "flat"
    if contract_available:
        from humanoid.deployment import control_contract

        contract = control_contract(task, robot)
        if contract["observation_dim"] != obs_dim:
            raise ValueError("Saved actor does not match the deployment observation contract")
        contract["policy_sha256"] = sha256(args.output / "policy.pt")
        write_json(args.output / "contract.json", contract)
    write_json(
        args.output / "metadata.json",
        {
            "task": task,
            "checkpoint_sha256": sha256(args.checkpoint),
            "export_kind": "raw_actor",
            "control_contract_available": contract_available,
            "portable_runtime": "humanoid.deployment.PortableController"
            if contract_available
            else None,
            "hardware_validated": False,
            "controller": task.get("controller", {}),
            "shield": task.get("shield", {}),
            "postprocessing_required": bool(task.get("controller") or task.get("shield")),
            "action_clip": cfg["clip_actions"],
            "observation_group": "actor",
            "observation_dim": obs_dim,
            "roundtrip_max_error": float((actual - expected).abs().max()),
            "joint_manifest": robot,
        },
    )


if __name__ == "__main__":
    main()
