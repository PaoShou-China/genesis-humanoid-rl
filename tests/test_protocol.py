"""Pairing is independent of method rewards, but sensitive to physical conditions."""

from copy import deepcopy

import numpy as np

from evaluation.protocol import command_schedule, physical_protocol, fingerprint
from humanoid.config import task_config


def key(spec):
    return fingerprint(physical_protocol(spec, seed=100, num_envs=2, episodes=8))


def test_reward_and_shield_variants_share_conditions():
    original = task_config("H2")
    other = deepcopy(original)
    other["reward_profile"] = "soft_safety"
    other["shield"] = {"type": "pd_target"}
    assert key(original) == key(other)
    other["randomization"] = {"friction_ratio": [0.5, 0.6]}
    assert key(original) != key(other)


def test_plan_does_not_depend_on_external_rng_or_early_termination():
    spec = task_config("H2")
    kwargs = dict(seed=100, horizon=1000, num_envs=2, command_steps=200)
    first = command_schedule(spec, **kwargs)
    np.random.seed(938)
    np.random.random(100)
    np.testing.assert_array_equal(first, command_schedule(spec, **kwargs))


def test_h3_and_h4_have_explicit_physical_conditions():
    assert task_config("H3")["push"]["force_n"] == [30.0, 0.0, 0.0]
    assert task_config("H4")["terrain"] == "rough"
