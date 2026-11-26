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
    for key in ("linear_error_m_s", "yaw_error_rad_s", "tilt_rad"):
        axes[0, 0].plot(t, [r[key] for r in rows], label=key)
    axes[0, 0].set_title("Tracking error and tilt")
    axes[0, 1].plot([r["com_x_m"] for r in rows], [r["com_y_m"] for r in rows])
    axes[0, 1].set(title="Whole-body CoM trajectory", xlabel="x [m]", ylabel="y [m]")
    joints = np.array([r["joint_pos_rad"] for r in rows])
    for j in (0, 3, 4, 6, 9, 10):
        axes[1, 0].plot(t, joints[:, j], label=f"joint {j}")
    axes[1, 0].set_title("Leg joint position [rad]")
    contact = np.array([r["foot_contact"] for r in rows])
    for j in range(2):
        axes[1, 1].step(t, contact[:, j] + 1.3*j, label=f"foot {j}")
    axes[1, 1].set_title("Foot contacts (offset for visibility)")
    axes[2, 0].plot(t, [r["power_w"] for r in rows], label="absolute mechanical power [W]")
    axes[2, 0].set_title("Mechanical power")
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
                axes[2, 1].plot([v.step for v in values], [v.value for v in values], label=tag)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
