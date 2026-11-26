"""One lazy command entry point; help and task discovery do not import Genesis."""

import argparse
import importlib
import sys

COMMANDS = {
    name: f"scripts.{name}"
    for name in (
        "doctor",
        "train",
        "evaluate",
        "play",
        "export",
        "compare",
        "plot",
        "sweep",
        "validate_motion",
        "sim2sim",
        "transfer_plan",
        "transfer_summary",
    )
}
COMMANDS.update(
    {
        "diagnose": "scripts.diagnose_balance",
        "smoke": "scripts.smoke_extensions",
        "videos": "scripts.compare_videos",
        "assets": "scripts.assets",
    }
)


def main():
    """Dispatch to small commands, keeping their complete native --help output."""
    parser = argparse.ArgumentParser(prog="ghrl", description="Genesis Humanoid RL Benchmark")
    parser.add_argument("command", choices=["tasks", *COMMANDS], nargs="?")
    argv = sys.argv[1:]
    if argv and argv[0] in COMMANDS:
        sys.argv = [f"ghrl {argv[0]}", *argv[1:]]
        importlib.import_module(COMMANDS[argv[0]]).main()
        return
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    elif args.command == "tasks":
        print(
            "H1  Standing balance\nH2  Velocity tracking\nH3  Torso-force push recovery\nH4  Rough-terrain locomotion"
        )
        print("H5  Motion dataset validation only; no trained tracker")
