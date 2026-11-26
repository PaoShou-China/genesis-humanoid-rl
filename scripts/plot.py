"""Plot measured evaluation trajectories and real PPO learning curves."""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Render six panels; no interpolation or fabricated learning curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.evaluation / "trace.json").read_text())
    if not rows:
        parser.error("Empty trace")
    t = [r["time_s"] for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].plot(t, [r["tilt_rad"] for r in rows], label="total tilt")
    w, x, y, z = np.array([r["root_quat_wxyz"] for r in rows]).T
    axes[0, 0].plot(t, np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), label="roll")
    axes[0, 0].plot(t, np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)), label="pitch")
    axes[0, 0].set(title="Root attitude", xlabel="Time [s]", ylabel="Angle [rad]")
    axes[0, 1].plot([r["com_x_m"] for r in rows], [r["com_y_m"] for r in rows])
    axes[0, 1].set(title="Whole-body CoM trajectory", xlabel="x [m]", ylabel="y [m]")
    joints = np.array([r["joint_pos_rad"] for r in rows])
    for j in (0, 3, 4, 6, 9, 10):
        axes[1, 0].plot(t, joints[:, j], label=f"joint {j}")
    axes[1, 0].set_title("Leg joint position [rad]")
    contact = np.array([r["foot_contact"] for r in rows])
    for j in range(2):
        axes[1, 1].step(t, contact[:, j] + 1.3 * j, label=f"foot {j}")
    axes[1, 1].set_title("Foot contacts (offset for visibility)")
    axes[2, 0].plot(t, [r["power_w"] for r in rows], label="absolute mechanical power [W]")
    axes[2, 0].set_title("Mechanical power")
    force = [r.get("external_force_n", 0) for r in rows]
    if max(force) > 0:
        force_axis = axes[2, 0].twinx()
        force_axis.step(t, force, color="black", linestyle="--", label="torso force [N]")
        force_axis.set_ylabel("External force [N]")
        force_axis.legend(loc="upper right", fontsize=7)
    axes[2, 1].set_title("Real training logs")
    if args.training:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(args.training)).Reload()
        tags = events.Tags()["scalars"]
        selected = ("Train/mean_reward", "Train/mean_episode_length")
        if "Train/mean_reward" not in tags:
            selected = ("Loss/value", "Loss/surrogate")
            axes[2, 1].set_title("PPO update losses (no completed training episodes)")
        for tag in tags:
            if tag in selected:
                values = events.Scalars(tag)
                target = axes[2, 1]
                if tag == "Train/mean_episode_length":
                    target = axes[2, 1].twinx()
                    target.set_ylabel("Episode length [steps]")
                target.plot(
                    [v.step for v in values],
                    [v.value for v in values],
                    label=tag,
                    color="tab:orange" if tag.endswith("length") else "tab:blue",
                )
                if target is not axes[2, 1]:
                    target.legend(loc="upper right", fontsize=7)
        axes[2, 1].set_xlabel("PPO iteration")
    else:
        axes[2, 1].plot(t, [r["linear_error_m_s"] for r in rows], label="linear speed error")
        axes[2, 1].set(
            title="Velocity tracking errors", xlabel="Time [s]", ylabel="Linear error [m/s]"
        )
        yaw_axis = axes[2, 1].twinx()
        yaw_axis.plot(
            t, [r["yaw_error_rad_s"] for r in rows], color="tab:orange", label="yaw-rate error"
        )
        yaw_axis.set_ylabel("Yaw-rate error [rad/s]")
        yaw_axis.legend(loc="upper right", fontsize=7)
    for ax in (axes[1, 0], axes[1, 1], axes[2, 0]):
        ax.set_xlabel("Time [s]")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper left" if ax in (axes[2, 0], axes[2, 1]) else "best")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
