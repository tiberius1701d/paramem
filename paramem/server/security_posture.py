"""Security posture log-line selection for the server startup banner.

Isolated here so the branching logic that decides which ``SECURITY: …`` line
the server emits at startup is a pure function of its inputs — testable
without a full ``lifespan_enter`` integration. Four posture buckets:

- **age two-identity, multi-recipient ready** — daily identity loadable and
  the recovery recipient is on disk; new writes are keyed to both.
- **age daily-only** — daily loadable but no recovery pub; writes go through
  but lose the recovery safety net until `paramem generate-key` is re-run.
- **Fernet (legacy PMEM1)** — only ``PARAMEM_MASTER_KEY`` is set; the server
  still reads and writes the pre-age envelope format.
- **SECURITY: OFF** — no key material loaded; infrastructure metadata is
  plaintext on disk.
"""

from __future__ import annotations

from paramem.backup.encryption import MASTER_KEY_ENV_VAR
from paramem.backup.types import FatalConfigError


def security_posture_log_line(
    *,
    fernet_loaded: bool,
    daily_loadable: bool,
    recovery_available: bool,
) -> tuple[str, bool]:
    """Return the ``(message, is_on)`` pair for the startup SECURITY log line.

    Precedence: age daily identity wins over the legacy Fernet key when both
    are present — new writes will route through age, so operators should see
    the age posture line rather than a misleading Fernet-only one.

    Parameters
    ----------
    fernet_loaded:
        ``PARAMEM_MASTER_KEY`` is set.
    daily_loadable:
        Daily identity file exists + passphrase env var is set (scrypt unwrap
        has not been performed; loadability is a precondition probe).
    recovery_available:
        Recovery public-key file exists and is readable.

    Returns
    -------
    tuple[str, bool]
        The message to log, and ``True`` when the posture is SECURITY-ON.
    """
    if daily_loadable and recovery_available:
        return (
            "SECURITY: ON (age daily identity loaded, recovery recipient available)",
            True,
        )
    if daily_loadable:
        return (
            "SECURITY: ON (age daily identity loaded, recovery recipient missing — "
            "run `paramem generate-key` to re-enable multi-recipient writes)",
            True,
        )
    if fernet_loaded:
        return (f"SECURITY: ON ({MASTER_KEY_ENV_VAR} set)", True)
    return (
        "SECURITY: OFF (no key — all infrastructure metadata is plaintext on disk)",
        False,
    )


def assert_startup_posture(
    *,
    require_encryption: bool,
    fernet_loaded: bool,
    daily_loadable: bool,
) -> None:
    """Refuse startup when ``require_encryption`` is set but no key is loadable.

    Single uniform fail-loud gate.  Applies to every feature (snapshots,
    shards, backups, infra): when the operator opts in, a missing key at
    startup is a fatal configuration error rather than a silent degrade to
    plaintext.

    Parameters
    ----------
    require_encryption:
        Operator-set flag from ``security.require_encryption``.  ``False``
        makes this function a no-op (AUTO semantics — the default).
    fernet_loaded:
        ``PARAMEM_MASTER_KEY`` is set.
    daily_loadable:
        Daily identity file + passphrase env var are both present.

    Raises
    ------
    FatalConfigError
        When ``require_encryption=True`` and neither key path is loadable.
    """
    if not require_encryption:
        return
    if fernet_loaded or daily_loadable:
        return
    raise FatalConfigError(
        "security.require_encryption=true but no key is loadable — "
        f"set {MASTER_KEY_ENV_VAR} (legacy) or run `paramem generate-key` and "
        "set PARAMEM_DAILY_PASSPHRASE before starting the server"
    )
