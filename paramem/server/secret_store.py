"""Per-secret-file loader for ParaMem.

Purpose: split what used to be a single ``.env`` file into one file per secret
under ``~/.config/paramem/secrets/`` with strict permissions. Reduces blast
radius: a single file disclosure exposes only that secret, not the full set
(HA token, cloud API keys, daily passphrase, etc.).

Layout:
    ~/.config/paramem/secrets/          (mode 0700, owner-only)
        HA_TOKEN                        (mode 0600)
        ANTHROPIC_API_KEY               (mode 0600)
        PARAMEM_DAILY_PASSPHRASE        (mode 0600)
        ...

Semantics:
- Each file's name is an env var name (conventionally UPPERCASE_UNDERSCORE).
- File content is the raw secret value. Trailing newline + whitespace stripped.
- Loaded via ``os.environ.setdefault`` — shell env and ``.env`` take precedence
  (see `main()` ordering in ``app.py``). Migration path: delete the line from
  ``.env`` once the per-secret file exists.
- Directory missing → no-op (back-compat). Directory present but loose
  permissions → refuse with clear error.

This module does not handle key material itself — it just ensures secrets
reach ``os.environ`` from a safer layout than a single shared file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SECRETS_DIR = Path.home() / ".config" / "paramem" / "secrets"

# Unix mode constants.
_REQUIRED_DIR_MODE = 0o700
_REQUIRED_FILE_MODE = 0o600
_MODE_MASK = 0o777


class SecretStoreError(RuntimeError):
    """Raised on permission / layout violations that we refuse to silently accept."""


def _check_mode(path: Path, required: int) -> None:
    mode = path.stat().st_mode & _MODE_MASK
    if mode != required:
        raise SecretStoreError(
            f"{path}: mode {oct(mode)} does not match required {oct(required)}. "
            f"Run: chmod {oct(required)[2:]} {path}"
        )


def load_secrets_from_dir(secrets_dir: Path | None = None) -> list[str]:
    """Load secrets from per-file layout into ``os.environ``.

    Returns the list of variable names that were newly set (i.e. not already
    present in the environment). Variables already in ``os.environ`` are left
    alone — precedence is shell env > ``.env`` > per-secret files.

    Parameters
    ----------
    secrets_dir
        Override the default ``~/.config/paramem/secrets`` location. Useful
        for tests. When None, the default is used.

    Raises
    ------
    SecretStoreError
        When the directory exists but has loose permissions, or a file has
        loose permissions. Never silently accepts insecure state.
    """
    path = secrets_dir if secrets_dir is not None else DEFAULT_SECRETS_DIR

    if not path.exists():
        # No directory = no-op. Back-compat for deployments that keep using .env.
        return []

    if not path.is_dir():
        raise SecretStoreError(f"{path}: expected a directory, got a file")

    _check_mode(path, _REQUIRED_DIR_MODE)

    loaded: list[str] = []
    for entry in sorted(path.iterdir()):
        if not entry.is_file():
            continue
        # Skip dotfiles so users can stash notes (e.g. .README) without tripping
        # the mode check.
        if entry.name.startswith("."):
            continue

        _check_mode(entry, _REQUIRED_FILE_MODE)

        name = entry.name
        if not _is_valid_env_name(name):
            logger.warning(
                "secret_store: skipping %s — filename is not a valid env var identifier",
                entry,
            )
            continue

        if name in os.environ:
            # Shell env or .env already set this. Leave it alone.
            continue

        value = entry.read_text(encoding="utf-8").rstrip("\n").rstrip()
        os.environ[name] = value
        loaded.append(name)

    return loaded


def _is_valid_env_name(name: str) -> bool:
    """Return True iff ``name`` is a plausible env-var identifier.

    Follows the POSIX-ish rule: letters, digits, underscore; no leading digit.
    Deliberately strict to avoid filename-based surprises.
    """
    if not name:
        return False
    if name[0].isdigit():
        return False
    return all(c.isalnum() or c == "_" for c in name)


def log_startup_posture(loaded: list[str], secrets_dir: Path | None = None) -> None:
    """Emit a single startup line describing the secret-store posture.

    Call once after ``load_secrets_from_dir``. The names of loaded secrets
    are logged (without values) so operators can confirm the expected secrets
    were picked up.
    """
    path = secrets_dir if secrets_dir is not None else DEFAULT_SECRETS_DIR
    if not path.exists():
        logger.info(
            "SECRETS: per-secret store not configured (%s does not exist). "
            "Using .env / shell env only.",
            path,
        )
        return
    if loaded:
        logger.info("SECRETS: loaded %d secrets from %s: %s", len(loaded), path, ", ".join(loaded))
    else:
        logger.info(
            "SECRETS: %s exists but no new secrets loaded (all already set via shell/.env)",
            path,
        )
