"""Metrics sanity checks guard empty results and missing sample variance."""
import pytest

from evaluation.statistics import summarize


def test_summary_is_episode_based():
    result = summarize([{"success": 1, "terminated": 0, "energy_j": 10.0},
                        {"success": 0, "terminated": 1, "energy_j": 30.0}])
    assert result["success_rate"] == 0.5
    assert result["energy_j"]["mean"] == 20
    assert result["energy_j"]["std"] == pytest.approx(14.1421356237)


def test_no_invented_uncertainty():
    assert summarize([{"success": 0, "terminated": 1}])["success"]["std"] is None
    with pytest.raises(ValueError):
        summarize([])
