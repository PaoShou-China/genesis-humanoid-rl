"""Record concrete environment, code, and asset provenance."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


def write_json(path: str | Path, value: Any) -> None:
    """Write strict JSON (NaN/Infinity are rejected)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: str | Path) -> str:
    """Hash a file without loading it all into RAM."""
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def archive_source(output: str | Path) -> str:
    """Preserve exact source bytes, including uncommitted changes, for each run."""
    root = Path(__file__).resolve().parents[1]
    target = Path(output) / "source_snapshot.zip"
    excluded = {
        ".git",
        ".venv",
        "__pycache__",
        "reports",
        "outputs",
        "logs",
        "third_party",
        "build",
        "dist",
        ".pytest_cache",
        ".ruff_cache",
    }
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # In a wheel install root is site-packages. Never archive dependencies.
        owned = (
            "humanoid",
            "algorithms",
            "evaluation",
            "visualization",
            "configs",
            "scripts",
            "tests",
            "docs",
            ".github",
        )
        sources = [
            file for name in owned if (root / name).is_dir() for file in (root / name).rglob("*")
        ]
        sources += [root / name for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE")]
        for source in sorted(sources):
            relative = source.relative_to(root)
            if source.is_file() and not excluded.intersection(relative.parts):
                if source.suffix in (".py", ".yaml", ".toml", ".md") or source.name in (
                    "LICENSE",
                    "NOTICE",
                ):
                    archive.write(source, relative)
    return sha256(target)


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
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git_commit": git.stdout.strip() or None,
        "git_dirty": bool(dirty.stdout.strip()),
        "genesislab_commit": "e06f383af31773fc748b1a72b5268bd84b26eb2a",
    }
