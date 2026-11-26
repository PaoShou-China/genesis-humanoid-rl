"""Real simulation fixtures (no mocked dynamics)."""
import pytest


@pytest.fixture(scope="session")
def env():
    """Share one CPU G1 scene to amortize compilation, resetting in each test."""
    import torch

    from humanoid.environments import make_env

    torch.set_num_threads(4)
    instance = make_env("H1", num_envs=2, seed=0, device="cpu", evaluation=True)
    yield instance
    instance.close()
