"""Descriptive episode summaries; seed-level inference is kept separate."""

import math
from statistics import mean, median, stdev


def summarize(episodes: list[dict]) -> dict:
    """Aggregate observed episodes; sample SD is null when n=1."""
    if not episodes:
        raise ValueError("No episodes to summarize")
    result = {"n_episodes": len(episodes)}
    for key in episodes[0]:
        values = [row[key] for row in episodes]
        if key in ("episode", "seed", "slot") or not all(type(v) in (int, float) for v in values):
            continue
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite metric: {key}")
        result[key] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else None,
            "median": median(values),
        }
    result["success_rate"] = mean(row["success"] for row in episodes)
    result["failure_rate"] = 1.0 - result["success_rate"]
    result["fall_rate"] = mean(row["terminated"] for row in episodes)
    if "push_exposed" in episodes[0]:
        exposed = [row for row in episodes if row["push_exposed"]]
        result["push_exposure_rate"] = len(exposed) / len(episodes)
        result["recovery_rate"] = mean(row["recovered"] for row in exposed) if exposed else None
        recovered = [row["recovery_time_s"] for row in exposed if row["recovered"]]
        result["n_push_exposed"] = len(exposed)
        result["n_recovered"] = len(recovered)
        result["recovery_time_s_recovered"] = (
            {
                "mean": mean(recovered),
                "median": median(recovered),
                "std": stdev(recovered) if len(recovered) > 1 else None,
            }
            if recovered
            else None
        )
    return result
