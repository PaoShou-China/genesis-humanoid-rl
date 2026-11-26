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
        result[key] = {"mean": mean(values), "std": stdev(values) if len(values) > 1 else None,
                       "median": median(values)}
    result["success_rate"] = mean(row["success"] for row in episodes)
    result["failure_rate"] = 1.0 - result["success_rate"]
    result["fall_rate"] = mean(row["terminated"] for row in episodes)
    return result
