"""Record concrete environment, code, and asset provenance."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def write_json(path: str | Path, value: Any) -> None:
    """Write strict JSON (NaN/Infinity are rejected)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    """Hash a file without loading it all into RAM."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def system_info() -> dict:
    """Inspect installed versions and GPU availability; never infer a GPU."""
    import torch

    packages = {}
    for name in ("torch", "genesis-world", "genesislab", "rsl-rl-lib", "tensordict", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return {
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git_commit": git.stdout.strip() or None, "git_dirty": bool(dirty.stdout.strip()),
        "genesislab_commit": "e06f383af31773fc748b1a72b5268bd84b26eb2a",
    }
