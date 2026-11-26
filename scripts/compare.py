"""Compare independent evaluation runs at the training-seed level."""
import argparse
import json
from pathlib import Path


def main() -> None:
    """Require the same task, test seed and episode count for paired comparisons."""
    from humanoid.provenance import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summaries, protocol = [], None
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        identity = [manifest["task"], manifest["evaluation_seed"], manifest["num_envs"], summary["n_episodes"]]
        if protocol is not None and protocol != identity:
            parser.error("Evaluation protocols differ; do not compare unmatched conditions")
        protocol = identity
        summaries.append({"run": str(run), "checkpoint": manifest["checkpoint_sha256"],
                          "success_rate": summary["success_rate"],
                          "fall_rate": summary["fall_rate"],
                          "linear_error_m_s": summary["mean_linear_error_m_s"]["mean"]})
    # No pooled p-values: episodes sharing a trained policy are not independent seeds.
    write_json(args.output, {"runs": summaries, "n_runs": len(summaries),
               "note": "Run rows only. Group by method and independent training seed before statistical inference."})
    print("run,success_rate,fall_rate,linear_error_m_s")
    for row in summaries:
        print(f"{row['run']},{row['success_rate']:.4f},{row['fall_rate']:.4f},{row['linear_error_m_s']:.4f}")


if __name__ == "__main__":
    main()
