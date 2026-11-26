"""Inspect the current environment before installing or upgrading anything."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys


def main() -> None:
    """Report missing dependencies without importing Genesis or building scenes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ("torch", "genesis-world", "genesislab", "rsl-rl-lib", "tensordict"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    if importlib.util.find_spec("torch"):
        import torch

        report.update(cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
                      gpu_count=torch.cuda.device_count())
    data = json.dumps(report, indent=2)
    print(data)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(data + "\n")


if __name__ == "__main__":
    main()
