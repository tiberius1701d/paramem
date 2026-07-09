"""Systemd user-timer reconciliation for the scheduled backup runner.

Mirrors ``paramem/server/systemd_timer.py`` for the ``paramem-backup`` timer,
sharing its reconciliation core (``_reconcile_timer``) via a ``TimerTarget``
instead of carrying a second copy of the render/reconcile logic. Reuses
``parse_schedule`` / ``TimerSpec`` from the consolidation timer module
(Decision E — same grammar, same parser).

Unit files point at ``python -m paramem.backup --tier daily`` (oneshot service).

``daily HH:MM`` normalisation (the ``schedule: "daily 04:00"`` default in
``server.yaml``) is handled once, centrally, by
``schedule_grammar.strip_daily_prefix`` at the top of
``parse_schedule_atom`` — this module no longer carries its own copy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from paramem.server.systemd_timer import (
    TimerTarget,
    _reconcile_timer,  # noqa: PLC2701 – intentional private reuse
    parse_schedule,
)
from paramem.utils.paths import find_project_root

logger = logging.getLogger(__name__)

TIMER_NAME = "paramem-backup"
UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_PATH = UNIT_DIR / f"{TIMER_NAME}.service"
TIMER_PATH = UNIT_DIR / f"{TIMER_NAME}.timer"


# Re-export so tests can ``from paramem.backup.timer import parse_schedule``
# and verify it is the same object as the server timer's parse_schedule.
# (test_timer.py::test_reuses_parse_schedule)
__all__ = [
    "parse_schedule",
    "render_service_unit",
    "reconcile",
]


def render_service_unit(python_path: str, project_root: str, tier: str = "daily") -> str:
    """Render the systemd .service unit for the backup runner.

    Parameterised on ``tier`` so weekly / monthly / yearly timer units
    can be emitted by passing ``tier="weekly"`` etc.

    Parameters
    ----------
    python_path:
        Absolute path to the Python interpreter (``sys.executable``).
    project_root:
        Absolute path to the project root (``WorkingDirectory``).
    tier:
        Backup tier — written as ``--tier <tier>`` in ``ExecStart``.
        Defaults to ``"daily"``.

    Returns
    -------
    str
        Content of the ``.service`` unit file.
    """
    return f"""[Unit]
Description=ParaMem scheduled backup (paramem.backup runner)
After=paramem-server.service

[Service]
Type=oneshot
WorkingDirectory={project_root}
EnvironmentFile=-{project_root}/.env
ExecStart={python_path} -m paramem.backup --tier {tier}
"""


def reconcile(
    schedule: str,
    *,
    python_path: str,
    project_root: str | None = None,
    tier: str = "daily",
) -> str:
    """Reconcile the ``paramem-backup`` systemd user timer.

    Mirrors ``paramem.server.systemd_timer.reconcile``.  Differences:

    - ``schedule`` comes from ``server_config.security.backups.schedule``
      (default ``"daily 04:00"``).
    - ``"off"`` / ``""`` disables and removes the timer units.
    - Unit files point at ``python -m paramem.backup --tier <tier>`` (not
      curl).

    Parameters
    ----------
    schedule:
        Schedule string (same grammar as ``consolidation.refresh_cadence``).
    python_path:
        Absolute path to the Python interpreter.
    project_root:
        Absolute path to the project root. Defaults to ``None``, in which
        case it is resolved via ``find_project_root(Path(__file__))``
        (nearest ancestor containing ``pyproject.toml``), falling back to
        ``Path(__file__).resolve().parents[2]`` when no such ancestor exists
        (e.g. an installed package under site-packages).
    tier:
        Backup tier written into the service unit's ``ExecStart``.  Always
        ``"daily"`` in production.

    Returns
    -------
    str
        Short human-readable description of the action taken.
    """
    if project_root is None:
        _r = find_project_root(Path(__file__))
        project_root = str(_r if _r is not None else Path(__file__).resolve().parents[2])
    target = TimerTarget(
        timer_name=TIMER_NAME,
        description="ParaMem scheduled backup",
        service_path=SERVICE_PATH,
        timer_path=TIMER_PATH,
        service_content=render_service_unit(python_path, project_root, tier=tier),
    )
    return _reconcile_timer(target, schedule)
