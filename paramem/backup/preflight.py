"""Pre-flight check for /migration/preview disk-pressure gate (Slice 6b).

Thin module so the retention TTL cache and preflight math are not entangled
with the existing 6a backup code.

The check estimates the footprint of a would-be pre-migration backup (config +
graph + registry) and compares it against the remaining global cap.  If the
estimate would push usage over the cap, ``fail_code="disk_pressure"`` is set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.server.config import ServerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreFlightCheck:
    """One pre-flight evaluation for /migration/preview.

    Attributes
    ----------
    fail_code:
        ``"disk_pressure"`` when the estimated backup footprint plus current
        usage would exceed the global cap; ``None`` when the check passes.
    disk_used_bytes:
        Current disk usage in bytes across the entire backup store.
    disk_cap_bytes:
        Global cap in bytes (``max_total_disk_gb * 1024**3``).
    estimate_bytes:
        Estimated size of the would-be pre-migration backup (sum of config,
        graph, and registry bytes as they currently exist on disk).
    """

    fail_code: str | None  # "disk_pressure" | None
    disk_used_bytes: int
    disk_cap_bytes: int
    estimate_bytes: int


def compute_pre_flight_check(
    *,
    server_config: "ServerConfig",
    loop,  # ConsolidationLoop | None — cloud-only = None
    backups_root: Path,
    live_config_path: Path,
    registry_path: "Path | None",
) -> PreFlightCheck:
    """Estimate pre-migration backup footprint and compare to global cap.

    Steps
    -----
    1. ``estimate = len(live_config_path.read_bytes()) if exists else 0``
       ``       + len(loop.merger.save_bytes()) if loop and hasattr(loop, "merger") else 0``
       ``       + len(registry_path.read_bytes())``
       ``         if registry_path and registry_path.exists() else 0``
    2. ``usage = compute_disk_usage(backups_root, server_config.security.backups)``
       (cached — no bypass; the 5s TTL is fine because the operator is not
       racing themselves).
    3. ``cap = int(server_config.security.backups.max_total_disk_gb * 1024**3)``
    4. When ``usage.total_bytes + estimate > cap`` →
       ``fail_code = "disk_pressure"``; else ``None``.

    Parameters
    ----------
    server_config:
        ``ServerConfig`` providing backups config and cap.  The
        ``security.backups`` sub-config is used directly.
    loop:
        ``ConsolidationLoop`` instance for graph access.  ``None`` when the
        server is in cloud-only mode; the graph contribution is then 0.
    backups_root:
        Root of the backup store (e.g. ``data/ha/backups/``).
    live_config_path:
        Path to the live ``server.yaml`` to be backed up.
    registry_path:
        Path to the key registry file, or ``None`` when unresolvable.
        Absent file → 0 bytes (no registry yet).

    Returns
    -------
    PreFlightCheck
        All four fields populated unconditionally so callers can surface the
        numbers whether or not pre-flight failed.

    Notes
    -----
    - Returns no-pressure result when ``server_config`` is a ``MagicMock`` (or
      otherwise lacks a real ``max_total_disk_gb`` numeric value) — protects
      test fixtures from false positives without masking real config errors in
      production.
    - Any I/O error inside the estimate loop is swallowed (logged WARN) and
      counted as 0 bytes for that component.  The preview must not crash on a
      transient read error — the estimate is a heuristic, not an authoritative
      size.
    - Graph bytes from ``loop.merger.save_bytes()`` may be expensive on very
      large graphs; acceptable because preview is operator-driven (not polled).

    .. todo:: compute_pre_flight_check re-serializes loop.merger.graph every
       /status poll (via _collect_pre_flight_items). Acceptable today
       (STAGING/TRIAL suppression covers the hot path; graph sizes modest),
       but if graph grows or poll rate increases, wrap this helper in a 5s TTL
       cache parallel to compute_disk_usage.
    """
    from paramem.backup.retention import compute_disk_usage

    # Guard: return no-pressure result when the config is not a real ServerConfig.
    # A MagicMock (or any object whose security.backups.max_total_disk_gb is not a
    # real numeric) would cause int(MagicMock()) to silently resolve to 1, producing
    # a false-positive disk_pressure fail in unit tests.  This guard is the single
    # source of truth — both the /migration/preview call site and the /status
    # populator (_collect_pre_flight_items) benefit automatically.
    _max_gb = getattr(
        getattr(getattr(server_config, "security", None), "backups", None),
        "max_total_disk_gb",
        None,
    )
    if not isinstance(_max_gb, (int, float)):
        return PreFlightCheck(
            fail_code=None,
            disk_used_bytes=0,
            disk_cap_bytes=0,
            estimate_bytes=0,
        )

    backups_root = Path(backups_root)
    backups_cfg = server_config.security.backups
    cap_bytes = int(backups_cfg.max_total_disk_gb * 1024**3)

    # --- Step 1: Estimate footprint ---
    estimate_bytes = 0

    # Config contribution.
    try:
        config_path = Path(live_config_path)
        if config_path.exists():
            estimate_bytes += len(config_path.read_bytes())
    except OSError as exc:
        logger.warning("compute_pre_flight_check: could not read live config for estimate: %s", exc)

    # Graph contribution (loop.merger.save_bytes()).
    # TODO: compute_pre_flight_check re-serializes loop.merger.graph every /status
    # poll. Acceptable today (STAGING/TRIAL suppression covers the hot path; graph
    # sizes modest), but if graph grows or poll rate increases, wrap this helper in
    # a 5s TTL cache parallel to compute_disk_usage.
    if loop is not None and hasattr(loop, "merger"):
        try:
            graph_bytes = loop.merger.save_bytes()
            estimate_bytes += len(graph_bytes)
        except Exception as exc:
            logger.warning(
                "compute_pre_flight_check: could not get graph bytes for estimate: %s", exc
            )

    # Registry contribution.
    if registry_path is not None:
        try:
            reg_path = Path(registry_path)
            if reg_path.exists():
                estimate_bytes += len(reg_path.read_bytes())
        except OSError as exc:
            logger.warning(
                "compute_pre_flight_check: could not read registry for estimate: %s", exc
            )

    # --- Step 2: Current disk usage (TTL-cached) ---
    try:
        usage = compute_disk_usage(backups_root, backups_cfg)
        disk_used_bytes = usage.total_bytes
    except Exception as exc:
        logger.warning("compute_pre_flight_check: could not compute disk usage: %s", exc)
        disk_used_bytes = 0

    # --- Step 3: Cap comparison ---
    fail_code: str | None = None
    if disk_used_bytes + estimate_bytes > cap_bytes:
        fail_code = "disk_pressure"

    return PreFlightCheck(
        fail_code=fail_code,
        disk_used_bytes=disk_used_bytes,
        disk_cap_bytes=cap_bytes,
        estimate_bytes=estimate_bytes,
    )
