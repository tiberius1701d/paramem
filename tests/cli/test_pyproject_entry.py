"""Integration test — confirms the [project.scripts] console-script entry works.

Requires ``pip install -e .`` to have registered the ``paramem`` entry point.
CI runs ``pip install -e ".[dev]"`` which also registers console scripts.

The console script is resolved as a sibling of ``sys.executable`` inside the
active Python environment's ``bin/`` directory so the test works whether or
not the environment is activated on PATH.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _paramem_script() -> str:
    """Return the absolute path to the ``paramem`` console script.

    Derived from ``sys.executable`` so the subprocess uses the same Python
    environment as the test runner, regardless of PATH activation state.
    """
    return str(Path(sys.executable).parent / "paramem")


class TestConsoleScriptEntry:
    def test_paramem_help_via_console_script(self):
        """``paramem --help`` exits 0 after editable install."""
        paramem_bin = _paramem_script()
        result = subprocess.run(
            [paramem_bin, "--help"],
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"{paramem_bin} --help exited {result.returncode}.\n"
            f"stdout: {result.stdout.decode()}\n"
            f"stderr: {result.stderr.decode()}\n"
            "If the script is missing, run: "
            f"{sys.executable.replace('python', 'pip')} install -e . --no-deps"
        )
