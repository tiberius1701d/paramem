"""The single transport boundary for host ``systemctl --user`` interaction.

Every production call site imports this module (``from paramem.utils import
systemctl``) and calls ``systemctl.run(...)`` / ``systemctl.spawn(...)`` —
module-attribute lookup at call time, never ``from paramem.utils.systemctl
import run`` — so one test monkeypatch on this module covers every caller.
"""

from __future__ import annotations

import subprocess


def run(*args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Blocking ``systemctl --user`` call; transport only, error policy stays at call sites.

    Args:
        *args: Verb and arguments passed to ``systemctl --user`` (e.g.
            ``"restart", "paramem-server"``).
        timeout: Seconds before ``subprocess.TimeoutExpired`` is raised, or
            ``None`` for no timeout.

    Returns:
        The completed process (``check=False`` — callers inspect
        ``returncode``/``stdout``/``stderr`` themselves).
    """
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def spawn(*args: str) -> None:
    """Detached ``systemctl --user`` call; used when the caller dies mid-command (self-restart).

    Args:
        *args: Verb and arguments passed to ``systemctl --user`` (e.g.
            ``"restart", "paramem-server"``).
    """
    subprocess.Popen(["systemctl", "--user", *args], start_new_session=True)
