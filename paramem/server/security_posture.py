"""Security posture log-line selection for the server startup banner.

:func:`security_posture_log_line` is isolated here so the branching logic
that decides which ``SECURITY: …`` line the server emits at startup is a
pure function of its two boolean inputs — testable without a full
``lifespan_enter`` integration. Three posture buckets:

- **age two-identity, multi-recipient ready** — daily identity loadable and
  the recovery recipient is on disk; new writes are keyed to both.
- **age daily-only** — daily loadable but no recovery pub; writes go through
  but lose the recovery safety net until `paramem generate-key` is re-run.
- **SECURITY: OFF** — no key material loaded; infrastructure metadata is
  plaintext on disk.

:func:`assert_startup_posture` is not a pure function of its inputs — it
derives its own daily-key precondition probe internally against
:data:`paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT` rather than taking it
as a parameter, so the caller cannot pass a stale or half-derived value.
"""

from __future__ import annotations

import pyrage

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
) -> None:
    """Refuse startup when ``require_encryption`` is set but the daily identity is unusable.

    Single uniform fail-loud gate.  Applies to every feature (snapshots,
    shards, backups, infra): when the operator opts in, a missing OR
    unusable key at startup is a fatal configuration error rather than a
    silent degrade to plaintext. The precondition probe
    (:func:`paramem.backup.key_store.daily_identity_loadable`) is derived
    internally against :data:`paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT`;
    when it passes, this performs the real scrypt unwrap (via
    :func:`paramem.backup.key_store.load_daily_identity_cached`) so a wrong
    passphrase or a corrupt envelope is caught here, at boot, rather than
    at the first write — the same call also warms the process-wide cache
    so that first write pays no repeat unwrap cost.

    Parameters
    ----------
    require_encryption:
        Operator-set flag from ``security.require_encryption``.  ``False``
        makes this function a no-op (AUTO semantics — the default).

    Raises
    ------
    FatalConfigError
        When ``require_encryption=True`` and either the daily identity is
        not loadable (missing file / unset env var) or it is loadable but
        the unwrap fails (wrong passphrase, corrupt or tampered envelope).
    """
    if not require_encryption:
        return

    from paramem.backup import key_store as _ks

    if not _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
        raise FatalConfigError(
            f"security.require_encryption=true but the daily age identity is not loadable.\n"
            f"\n"
            f"Likely cause:\n"
            f"  security.require_encryption: true is set in server.yaml but\n"
            f"  {DAILY_PASSPHRASE_ENV_VAR} is unset or ~/.config/paramem/daily_key.age\n"
            f"  is missing at startup.\n"
            f"\n"
            f"Remediation:\n"
            f"  - export {DAILY_PASSPHRASE_ENV_VAR}=<your daily passphrase>\n"
            f"  - Confirm ~/.config/paramem/daily_key.age exists and is readable.\n"
            f"  - If no key exists yet, run: paramem generate-key\n"
            f"  - OR set security.require_encryption: false to fall back to AUTO posture.\n"
            f"\n"
            f"See SECURITY.md for a full reset procedure that\n"
            f"preserves speaker_profiles.json."
        )

    try:
        _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
    except (RuntimeError, pyrage.DecryptError, OSError, ValueError) as exc:
        raise FatalConfigError(
            f"security.require_encryption=true and ~/.config/paramem/daily_key.age "
            f"is present, but it could not be unwrapped: {exc}.\n"
            f"\n"
            f"Likely cause:\n"
            f"  {DAILY_PASSPHRASE_ENV_VAR} does not match the passphrase the key\n"
            f"  was wrapped with, or ~/.config/paramem/daily_key.age is corrupt or\n"
            f"  tampered with.\n"
            f"\n"
            f"Remediation:\n"
            f"  - Confirm {DAILY_PASSPHRASE_ENV_VAR} matches the passphrase set by\n"
            f"    `paramem generate-key` / the most recent `paramem change-passphrase`.\n"
            f"  - If the daily key file is corrupt, recover via: paramem restore --help\n"
            f"\n"
            f"See SECURITY.md for a full reset procedure that\n"
            f"preserves speaker_profiles.json."
        ) from exc
