"""Asset, floating-base and physical parameter validation."""
import pytest


@pytest.mark.simulation
def test_robot_matches_asset(env):
    from humanoid.robots import validate_robot

    report = validate_robot(env)
    assert report["num_joints"] == 29
    assert report["control_dt_s"] == 0.02
    assert report["gravity_m_s2"] == [0.0, 0.0, -9.81]
