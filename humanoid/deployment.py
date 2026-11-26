"""Versioned flat-task controller contract shared by deployment adapters."""

from humanoid.config import ROOT, read_yaml, validate_task
from humanoid.transfer import actor_observation
from humanoid.control import process_action


def control_contract(spec, robot=None):
    """Record controller decisions separately from simulator-specific physics."""
    validate_task(spec)
    if spec.get("terrain", "flat") != "flat":
        raise ValueError("Deployment contract currently supports flat tasks only")
    asset = read_yaml(ROOT / "configs/robots/g1_29dof.yaml")
    joints = asset["joints"]
    gains = spec.get("pd_gains", {})
    nominal = {
        "joint_names": [j["name"] for j in joints],
        "home_rad": [j["home_rad"] for j in joints],
        "position_limits_rad": [j["position_limits_rad"] for j in joints],
        "kp": [j["kp"] * gains.get("stiffness_scale", 1) for j in joints],
        "kd": [j["kd"] * gains.get("damping_scale", 1) for j in joints],
        "effort_limit_nm": [j["effort_limit_nm"] for j in joints],
    }
    if robot:
        if robot["joint_names"] != nominal["joint_names"]:
            raise ValueError("Export joint order differs from audited G1")
        # DR samples in training robot.json are not nominal deployment gains.
        for field in ("home_rad", "position_limits_rad"):
            nominal[field] = robot[field]
    proprio = spec.get("actor_observation") == "proprioceptive"
    return {
        "schema_version": 1,
        "task": spec,
        "robot": nominal,
        "observation_dim": 96 if proprio else 99,
        "observation_terms": ([] if proprio else ["base_lin_vel_b"])
        + [
            "base_ang_vel_b",
            "projected_gravity_b",
            "commands_vx_vy_wz",
            "joint_pos_minus_home",
            "joint_vel",
            "last_processed_action",
        ],
        "quaternion_order": "wxyz",
        "angular_velocity_frame": "body",
        "normalization": "inside_policy_pt",
        "control_dt_s": spec.get("timing", {}).get("control_dt_s", 0.02),
        "actuator_dt_s": spec.get("timing", {}).get("physics_dt_s", 0.02),
        "observation_limitations": "Ground-truth attitude; no hardware IMU estimator implemented",
        "hardware_validated": False,
        "asset_xml_sha256": asset["asset_xml_sha256"],
    }


class PortableController:
    """Identical observation assembly and action processing in any adapter.

    State tensors are batched and named explicitly; adapters own sensing, timing,
    actuator dynamics and reset. No simulator truth is silently substituted for
    a missing observation.
    """

    def __init__(self, contract):
        import torch

        if contract["schema_version"] != 1:
            raise ValueError("Unsupported control contract version")
        expected = control_contract(contract["task"])
        if contract["observation_dim"] != expected["observation_dim"]:
            raise ValueError("Observation dimension does not match task profile")
        self.contract, self.spec = contract, contract["task"]
        self.robot = contract["robot"]
        self.values = {
            key: torch.tensor(self.robot[key], dtype=torch.float32).unsqueeze(0)
            for key in ("home_rad", "position_limits_rad", "kp", "kd", "effort_limit_nm")
        }

    def observe(self, state, commands, last_action):
        return actor_observation(
            q=state["q"],
            dq=state["dq"],
            home=self.values["home_rad"],
            angular_velocity=state["angular_velocity"],
            gravity=state["gravity"],
            commands=commands,
            last_action=last_action,
            linear_velocity=(
                None
                if self.spec.get("actor_observation") == "proprioceptive"
                else state["linear_velocity"]
            ),
        )

    def process(self, action, state):
        return process_action(
            action,
            spec=self.spec,
            q=state["q"],
            dq=state["dq"],
            home=self.values["home_rad"],
            limits=self.values["position_limits_rad"],
            gravity=state["gravity"],
            angular_velocity=state["angular_velocity"],
            joint_names=self.robot["joint_names"],
            kp=self.values["kp"],
            kd=self.values["kd"],
            effort=self.values["effort_limit_nm"],
        )
