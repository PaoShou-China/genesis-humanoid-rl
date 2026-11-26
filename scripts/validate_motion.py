"""Validate a real motion NPZ without unpickling or inventing demonstration data."""

import argparse
from pathlib import Path


def validate_motion(path):
    """Require timestamps/fps and explicit named joint/body arrays in SI units."""
    import numpy as np

    from humanoid.provenance import sha256

    with np.load(path, allow_pickle=False) as motion:
        required = {
            "fps",
            "joint_names",
            "body_names",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        if required - set(motion.files):
            raise ValueError(f"Missing fields: {sorted(required - set(motion.files))}")
        fps = float(motion["fps"].item())
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        joints, bodies = motion["joint_names"].tolist(), motion["body_names"].tolist()
        if not isinstance(joints, list) or not isinstance(bodies, list):
            raise ValueError("Joint/body names must be one-dimensional arrays")
        for names in (joints, bodies):
            if (
                not names
                or not all(isinstance(n, str) and n for n in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError("Names must be nonempty, unique strings")
        frames = len(motion["joint_pos"])
        if frames < 2:
            raise ValueError("At least two motion frames are required")
        shapes = {
            "joint_pos": (frames, len(joints)),
            "joint_vel": (frames, len(joints)),
            "body_pos_w": (frames, len(bodies), 3),
            "body_quat_w": (frames, len(bodies), 4),
            "body_lin_vel_w": (frames, len(bodies), 3),
            "body_ang_vel_w": (frames, len(bodies), 3),
        }
        for key, shape in shapes.items():
            if motion[key].shape != shape or not np.isfinite(motion[key]).all():
                raise ValueError(f"Invalid shape or non-finite values: {key}")
        norms = np.linalg.norm(motion["body_quat_w"], axis=-1)
        if not np.allclose(norms, 1, atol=1e-3):
            raise ValueError("Body quaternions must have unit norm; declare ordering externally")
    return {
        "file_sha256": sha256(path),
        "frames": frames,
        "fps": fps,
        "duration_s": (frames - 1) / fps,
        "joint_names": joints,
        "body_names": bodies,
        "note": "Schema verified; licensing, coordinate conventions and physical feasibility require separate review.",
    }


def main():
    """Use metadata fps; a filename such as *80fps* cannot override it."""
    from humanoid.provenance import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, validate_motion(args.path))


if __name__ == "__main__":
    main()
