"""Keep the installed front door usable without simulator construction."""

import subprocess
import sys


def test_task_discovery():
    result = subprocess.run(
        [sys.executable, "-m", "humanoid", "tasks"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "H4" in result.stdout
    assert "no trained tracker" in result.stdout


def test_subcommand_help_is_forwarded():
    result = subprocess.run(
        [sys.executable, "-m", "humanoid", "evaluate", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--evaluation-config" in result.stdout
