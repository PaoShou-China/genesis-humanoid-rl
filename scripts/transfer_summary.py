"""Summarize a transfer plan without treating missing runs as successful."""

import argparse
import json
from pathlib import Path


def main():
    from humanoid.provenance import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows, candidates = [], {}
    for run in plan["runs"]:
        folder = Path(run["command"][-1])
        row = {k: run[k] for k in ("engine", "kp_scale", "dt_ms", "delay_ms")}
        status_path, summary_path = folder / "status.json", folder / "summary.json"
        row["state"] = "missing"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            row["state"] = "completed" if status.get("evaluation_completed") else "failed"
        if row["state"] == "completed":
            if not summary_path.exists():
                raise ValueError(f"Completed run has no summary: {folder}")
            summary = json.loads(summary_path.read_text())
            row["success_rate"] = summary["success_rate"]
        rows.append(row)
        candidates.setdefault(run["kp_scale"], []).append(row)
    accepted = [
        gain
        for gain, tests in candidates.items()
        if all(r["state"] == "completed" and r["success_rate"] == 1 for r in tests)
    ]
    write_json(
        args.output,
        {
            "runs": rows,
            "completed": sum(r["state"] == "completed" for r in rows),
            "planned": len(rows),
            "empirically_accepted_gain_candidates": accepted,
            "criterion": "All planned development cases completed with episode success rate 1.0",
            "limitations": [
                "Finite development samples only; not a stability or real-world certificate",
                "No interpolation between accepted discrete gains",
                "Constant pose targets do not excite command-transport delay",
                "Source plan and result directories must be preserved together",
            ],
        },
    )
    print(
        f"Completed {sum(r['state'] == 'completed' for r in rows)}/{len(rows)} runs; "
        f"accepted discrete candidates: {accepted}"
    )


if __name__ == "__main__":
    main()
