"""Security posture log-line selection for the server startup banner.

Isolated here so the branching logic that decides which ``SECURITY: …`` line
the server emits at startup is a pure function of its inputs — testable
without a full ``lifespan_enter`` integration. Three posture buckets:

- **age two-identity, multi-recipient ready** — daily identity loadable and
  the recovery recipient is on disk; new writes are keyed to both.
- **age daily-only** — daily loadable but no recovery pub; writes go through
  but lose the recovery safety net until `paramem generate-key` is re-run.
- **SECURITY: OFF** — no key material loaded; infrastructure metadata is
  plaintext on disk.
"""

from __future__ import annotations

from paramem.backup.key_store import DAILY_PASSPHRASE_ENV_VAR
from paramem.backup.types import FatalConfigError


def security_posture_log_line(
    *,
    daily_loadable: bool,
    recovery_available: bool,
) -> tuple[str, bool]:
    """Return the ``(message, is_on)`` pair for the startup SECURITY log line.

    Parameters
    ----------
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
    return (
        "SECURITY: OFF (no key — all infrastructure metadata is plaintext on disk)",
        False,
    )


def assert_startup_posture(
    *,
    require_encryption: bool,
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
    daily_loadable:
        Daily identity file + passphrase env var are both present.

    Raises
    ------
    FatalConfigError
        When ``require_encryption=True`` and the daily identity is not
        loadable.
    """
    if not require_encryption:
        return
    if daily_loadable:
        return
    raise FatalConfigError(
        "security.require_encryption=true but the daily age identity is not "
        f"loadable — run `paramem generate-key` and set {DAILY_PASSPHRASE_ENV_VAR} "
        "before starting the server"
    )
