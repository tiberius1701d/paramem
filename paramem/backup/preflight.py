"""Pre-flight check for /migration/preview disk-pressure gate.

Thin module so the retention TTL cache and preflight math are not entangled
with the scheduled backup runner code.

The check estimates the footprint of a would-be pre-migration backup — live
config bytes, the episodic tier's BOUND slot ``graph.json`` (the simulate
venue's payload), and every tier's ``key_metadata.json`` bookkeeping — and
compares it against the remaining global cap.  If the estimate would push
usage over the cap, ``fail_code="disk_pressure"`` is set.

``fail_code`` also admits ``"check_error"`` — the caller-minted sentinel
:func:`~paramem.server.app.migration_preview` constructs when
:func:`compute_pre_flight_check` itself raises. It is declared here (rather
than left as a string only the caller knows about) because this module owns
the ``fail_code`` domain; the caller only mints the value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.server.config import ServerConfig

logger = logging.getLogger(__name__)


class PreFlightUnavailable(RuntimeError):
    """The backups disk cap could not be measured.

    Raised only when ``server_config.security.backups.max_total_disk_gb`` is
    not a real numeric (``None``, a ``MagicMock``, or ``server_config`` itself
    is ``None``). Failures reading a component of the estimate or the disk
    usage scan are not this class — they propagate as their own type
    (``OSError``, ``RuntimeError``, or :class:`pyrage.DecryptError`); see
    :func:`compute_pre_flight_check`'s ``Raises`` section.
    """


@dataclass(frozen=True)
class PreFlightCheck:
    """One pre-flight evaluation for /migration/preview.

    Attributes
    ----------
    fail_code:
        ``"disk_pressure"`` when the estimated backup footprint plus current
        usage would exceed the global cap; ``"check_error"`` when the caller
        minted this instance because :func:`compute_pre_flight_check` itself
        raised (no real measurement was taken — the three measurement fields
        are ``None`` in that case); ``None`` when the check ran and passed.
    disk_used_bytes:
        Current disk usage in bytes across the entire backup store, or
        ``None`` when ``fail_code == "check_error"`` (no measurement taken).
    disk_cap_bytes:
        Global cap in bytes (``max_total_disk_gb * 1024**3``), or ``None``
        under the same ``check_error`` condition.
    estimate_bytes:
        Estimated size of the would-be pre-migration backup (sum of config,
        graph, and registry bytes as they currently exist on disk), or
        ``None`` under the same ``check_error`` condition.
    """

    fail_code: str | None  # "disk_pressure" | "check_error" | None
    disk_used_bytes: int | None
    disk_cap_bytes: int | None
    estimate_bytes: int | None


def compute_pre_flight_check(
    *,
    server_config: "ServerConfig",
    loop,  # ConsolidationLoop | None — cloud-only = None
    backups_root: Path,
    live_config_path: Path,
    adapter_dir: "Path | None",
) -> PreFlightCheck:
    """Estimate pre-migration backup footprint and compare to global cap.

    Steps
    -----
    1. ``estimate = len(live_config_path.read_bytes()) if exists else 0``
       ``       + len(read_maybe_encrypted(bound_slot / "graph.json"))``
       ``         if loop and the episodic tier's BOUND slot carries a``
       ``         graph.json else 0 (bound_slot resolved via``
       ``         find_live_slot(episodic_root, tier_registry_sha256(episodic_root)))``
       ``       + sum(len(read_maybe_encrypted(f)) for f in every tier's``
       ``         key_metadata.json under adapter_dir that exists)``
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
        ``ConsolidationLoop`` instance for graph path access (``loop.output_dir``).
        ``None`` when the server is in cloud-only mode; the graph contribution
        is then 0.  Graph bytes are sourced from the episodic tier's BOUND
        slot ``graph.json`` under ``loop.output_dir / "episodic"`` (not from
        the in-memory ``merger.graph``, which is cleared at cycle-end and is
        empty between cycles, and not from a tier-root path — nothing writes
        one any more).
    backups_root:
        Root of the backup store (e.g. ``data/ha/backups/``).
    live_config_path:
        Path to the live ``server.yaml`` to be backed up.
    adapter_dir:
        Adapter store root, or ``None`` when unresolvable.  The registry
        contribution sums every tier's ``key_metadata.json`` under it (main
        tiers and interim slots); an absent per-tier file contributes 0
        bytes.

    Returns
    -------
    PreFlightCheck
        All four fields populated unconditionally so callers can surface the
        numbers whether or not pre-flight failed.

    Raises
    ------
    PreFlightUnavailable
        The backups cap is unavailable (``server_config`` is ``None`` or its
        ``security.backups.max_total_disk_gb`` is not a real numeric value —
        e.g. a ``MagicMock`` in a test fixture).
    OSError, RuntimeError, pyrage.DecryptError
        A component of the estimate or the disk-usage scan could not be read
        (e.g. a permission error, an age-encrypted artifact with no daily
        identity loaded — ``RuntimeError`` via
        :func:`paramem.backup.encryption.read_maybe_encrypted` — or corrupt/
        tampered age ciphertext — ``pyrage.DecryptError``, not a subclass of
        ``OSError`` or ``RuntimeError``). Not caught here — propagates to the
        caller, which mints ``check_error``.

    Notes
    -----
    - Graph bytes are read from the episodic tier's BOUND slot
      ``graph.json`` (``loop.merger.graph`` is cleared at cycle-end;
      re-serializing it would always yield an empty-graph estimate).
      Reading the on-disk file also reflects what the actual backup would
      capture.
    """
    from paramem.backup.encryption import read_maybe_encrypted
    from paramem.backup.retention import compute_disk_usage

    # Guard: raise when the cap is unavailable rather than fake a pass.  A
    # MagicMock (or any object whose security.backups.max_total_disk_gb is not
    # a real numeric) would otherwise let int(MagicMock()) resolve silently, or
    # a zeroed fake result would stage a candidate on an unmeasured estimate.
    _max_gb = getattr(
        getattr(getattr(server_config, "security", None), "backups", None),
        "max_total_disk_gb",
        None,
    )
    if not isinstance(_max_gb, (int, float)):
        raise PreFlightUnavailable(
            f"backups cap unavailable: security.backups.max_total_disk_gb={_max_gb!r}"
        )

    backups_root = Path(backups_root)
    backups_cfg = server_config.security.backups
    cap_bytes = int(backups_cfg.max_total_disk_gb * 1024**3)

    # --- Step 1: Estimate footprint ---
    estimate_bytes = 0

    # Config contribution.  Decrypted-length is the honest input — the future
    # pre-migration backup re-encrypts plaintext, so ciphertext length on disk
    # would be a double-count of age envelope overhead.
    config_path = Path(live_config_path)
    if config_path.exists():
        estimate_bytes += len(read_maybe_encrypted(config_path))

    # Graph contribution: read the episodic tier's BOUND slot graph.json on
    # disk. merger.graph is cleared at cycle-end (in the finally block), so
    # calling save_bytes() on it would re-serialize an empty graph and
    # underestimate. The on-disk file is the durable artifact that the
    # backup itself would capture, so reading it is both correct and avoids
    # a re-serialization. This term is genuinely load-bearing, not a
    # placeholder: write_bundle's per-tier capture now includes graph.json
    # whenever the tier has a bound slot carrying one (the simulate venue's
    # payload) — the train venue's bound slot carries no graph.json and this
    # term simply contributes 0 for it. Nothing writes a tier-root
    # graph.json any more — the payload lives in a timestamped slot under
    # the tier root, written via :func:`~paramem.adapters.slot.write_slot`.
    # Resolution composes the same two primitives every other bound-slot
    # reader in the package does (tier_registry_sha256 + find_live_slot) and
    # is deliberately NOT wrapped in a local try/except: a read/decrypt
    # failure here is exactly the "component of the estimate could not be
    # read" condition this function's own contract propagates (see the
    # Raises section) — this function is fail-loud by design, unlike the
    # migration comparison report's read-only display value.
    if loop is not None and hasattr(loop, "output_dir"):
        from paramem.adapters.manifest import find_live_slot, tier_registry_sha256
        from paramem.adapters.slot import payload_filename
        from paramem.memory.interim_adapter import adapter_slot_root_for_name

        _episodic_root = adapter_slot_root_for_name(Path(getattr(loop, "output_dir")), "episodic")
        _bound_slot = find_live_slot(_episodic_root, tier_registry_sha256(_episodic_root))
        if _bound_slot is not None:
            _graph_path = _bound_slot / payload_filename("simulate")
            if _graph_path.exists():
                estimate_bytes += len(read_maybe_encrypted(_graph_path))

    # Registry contribution: sum every tier's key_metadata.json.  See note
    # above on decrypted-length choice.
    if adapter_dir is not None:
        from paramem.memory.interim_adapter import iter_tier_roots

        for _tier, _tier_root in iter_tier_roots(Path(adapter_dir)):
            _km_path = _tier_root / "key_metadata.json"
            if _km_path.exists():
                estimate_bytes += len(read_maybe_encrypted(_km_path))

    # --- Step 2: Current disk usage (TTL-cached) ---
    usage = compute_disk_usage(backups_root, backups_cfg)
    disk_used_bytes = usage.total_bytes

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
