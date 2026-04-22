"""Shared helpers for the PARAMEM_EXTRA_ARGS=--defer-model hold.

Kept deliberately dependency-free so ``experiments/utils/gpu_guard.py`` can
import from here without pulling the full server stack (torch, fastapi,
peft, …) into test processes.  The server side in ``paramem.server.app``
re-exports ``format_cmd_hint`` for /status.
"""

from __future__ import annotations

import os


def format_cmd_hint(argv: list[str]) -> str:
    """Reduce a process argv to a short "interp / entry" hint.

    Examples:
        ["/usr/bin/python", "-m", "paramem.server.app", ...] -> "python / paramem.server.app"
        ["python", "experiments/test8.py"]                   -> "python / test8.py"
        ["python", "--version"]                              -> "python"
        ["bash"]                                             -> "bash"
    """
    if not argv:
        return ""
    interp = os.path.basename(argv[0]) or argv[0]
    if len(argv) >= 3 and argv[1] == "-m":
        return f"{interp} / {argv[2]}"
    if len(argv) >= 2 and not argv[1].startswith("-"):
        return f"{interp} / {os.path.basename(argv[1])}"
    return interp
