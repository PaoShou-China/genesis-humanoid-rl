"""Compare independent evaluation runs at the training-seed level."""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, median, stdev


def protocol_key(manifest, summary):
    """Use physical test identity; retain strict legacy comparison for old reports."""
    if "protocol_sha256" in manifest:
        return [
            manifest["protocol_sha256"],
            manifest["command_plan_sha256"],
            manifest.get("goal_plan_sha256"),
        ]
    return [
        manifest["task"],
        manifest["evaluation_seed"],
        manifest["num_envs"],
        summary["n_episodes"],
    ]


def main() -> None:
    """Require the same task, test seed and episode count for paired comparisons."""
    from humanoid.provenance import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Method labels; repeated labels group independent training seeds",
    )
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.runs):
        parser.error("Provide one method label per run")
    summaries, protocol = [], None
    for index, run in enumerate(args.runs):
        manifest = json.loads((run / "manifest.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        identity = protocol_key(manifest, summary)
        if protocol is not None and protocol != identity:
            parser.error("Evaluation protocols differ; do not compare unmatched conditions")
        protocol = identity
        summaries.append(
            {
                "run": str(run),
                "method": args.labels[index] if args.labels else str(run),
                "training_seed": manifest.get("training_seed"),
                "checkpoint": manifest["checkpoint_sha256"],
                "success_rate": summary["success_rate"],
                "fall_rate": summary["fall_rate"],
                "linear_error_m_s": summary["mean_linear_error_m_s"]["mean"],
            }
        )
    # No pooled p-values: episodes sharing a trained policy are not independent seeds.
    groups = defaultdict(list)
    for row in summaries:
        groups[row["method"]].append(row)
    aggregate = {}
    for label, rows in groups.items():
        seeds = [row["training_seed"] for row in rows]
        if len(rows) > 1 and (None in seeds or len(set(seeds)) != len(seeds)):
            parser.error(f"Method {label}: aggregation requires distinct explicit training seeds")
        values = [row["success_rate"] for row in rows]
        aggregate[label] = {
            "n_training_seeds": len(rows) if None not in seeds else None,
            "mean_success_rate": mean(values),
            "median_success_rate": median(values),
            "std_success_rate": stdev(values) if len(values) > 1 else None,
        }
    write_json(
        args.output,
        {
            "runs": summaries,
            "n_runs": len(summaries),
            "methods": aggregate,
            "note": "Run rows only. Group by method and independent training seed before statistical inference.",
        },
    )
    print("run,success_rate,fall_rate,linear_error_m_s")
    for row in summaries:
        print(
            f"{row['run']},{row['success_rate']:.4f},{row['fall_rate']:.4f},{row['linear_error_m_s']:.4f}"
        )


if __name__ == "__main__":
    main()
