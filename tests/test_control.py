"""Algebraic PD projection checks, including infeasibility and exact zero behavior."""

import torch

from humanoid.control import project_pd_target


def project(q=0.0, dq=0.0, action=1.0):
    """One joint with +/-1 rad targets and +/-2 Nm affine-PD effort."""

    def t(value):
        return torch.tensor([[value]])

    return project_pd_target(
        t(action),
        q=t(q),
        dq=t(dq),
        home=t(0.0),
        limits=torch.tensor([[[-1.0, 1.0]]]),
        kp=t(10.0),
        kd=t(1.0),
        effort=t(2.0),
        scale=1.0,
        clip=1.0,
    )


def test_torque_proxy_is_bounded_when_intersection_exists():
    result = project(dq=0.5)
    torque = 10 * result.action - 0.5
    assert torque.abs().max() <= 2.0 + 1e-6
    assert result.intervention.item()
    assert not result.infeasible.item()


def test_empty_intersection_is_reported_not_silently_certified():
    result = project(q=3.0)
    assert result.infeasible.item()
    assert -1 <= result.action.item() <= 1


def test_zero_equilibrium_needs_no_intervention():
    result = project(action=0.0)
    assert result.action.item() == 0
    assert not result.intervention.item()
