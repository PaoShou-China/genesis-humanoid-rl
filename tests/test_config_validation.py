"""Reject unsupported physics claims and malformed configuration before scene build."""

from copy import deepcopy

import pytest

from humanoid.config import task_config, validate_task


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_length_s", float("nan")),
        ("action_scale", -0.1),
        ("randomization", {"unknown_mass": [1, 2]}),
        ("randomization", {"friction_ratio": [-0.5, 1]}),
        ("shield", {"type": "formal_safety_certificate"}),
        ("controller", {"type": "ankle_feedback", "residual_scale": 2}),
        ("terrain", "unknown"),
    ],
)
def test_invalid_task_fields_fail_early(field, value):
    spec = deepcopy(task_config("H2"))
    spec[field] = value
    with pytest.raises(ValueError):
        validate_task(spec)
