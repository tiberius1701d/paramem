"""Per-user opaque bearer-token store for the ParaMem REST server.

Tokens are minted as ``secrets.token_urlsafe(32)`` values and stored by
SHA-256 hash — the plaintext token is returned exactly once at mint time
and is never persisted.

On-disk schema (one mandatory tier, no optional buckets):

.. code-block:: json

    {
        "version": 1,
        "tokens": {
            "<sha256hex>": {
                "speaker_id": "Speaker0",
                "label": "Alice iPad",
                "created": "<iso8601 utc>",
                "revoked": false
            }
        }
    }

The store is written via :func:`paramem.backup.encryption.write_infra_bytes`
(age-encrypted when a daily identity is loaded) and read via
:func:`paramem.backup.encryption.read_maybe_encrypted`.  File-I/O error
handling mirrors :class:`paramem.server.speaker.SpeakerStore._load`.

Security properties
-------------------
- The dict key is ``sha256(token)``.  Lookup is a single ``dict[key]`` access;
  no iteration, no ``hmac.compare_digest`` loop — a 256-bit hash as the key
  leaks nothing via timing.
- The store follows the deployment-wide AUTO encryption mode: plaintext when no
  daily key is loaded (Security OFF), age-encrypted when a daily key is loaded
  (Security ON).  This matches all other infra files (e.g. ``speaker_profiles.json``)
  and is consistent with ``assert_mode_consistency``'s explicit acceptance of
  "Plaintext only, no daily identity → OK (Security OFF)".
- Mixed state (plaintext file on disk while a daily key is loaded) is caught at
  startup by ``assert_mode_consistency`` via ``infra_paths()``.
- A TOCTOU guard in :meth:`_save` additionally catches key eviction between the
  pre-write loadability check and the actual write when Security is ON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes
from paramem.backup.key_store import DAILY_KEY_PATH_DEFAULT, daily_identity_loadable

logger = logging.getLogger(__name__)

_STORE_VERSION = 1


def _sha256hex(value: str) -> str:
    """Return the lowercase hex SHA-256 digest of *value* encoded as UTF-8."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class UserTokenStore:
    """Persistent per-user opaque-token store.

    Tokens are minted by :meth:`mint`, which returns the plaintext token
    exactly once.  Internal storage and all disk I/O use only the
    ``sha256(token)`` key.

    All public methods are thread-safe via an internal :class:`threading.Lock`.

    Parameters
    ----------
    store_path:
        Path to the JSON file on disk.  Parent directory is created on first
        write.  The file is envelope-encrypted when a daily age identity is
        loadable (see :func:`paramem.backup.encryption.write_infra_bytes`).
    """

    def __init__(self, store_path: Path | str) -> None:
        self.store_path = Path(store_path)
        # sha256hex → {"speaker_id": str, "label": str, "created": str, "revoked": bool}
        self._tokens: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the token store from disk.

        Missing file is silently ignored (fresh deployment).  Parse and
        I/O errors are logged and result in an empty in-memory store;
        boundary handling mirrors :class:`paramem.server.speaker.SpeakerStore._load`.
        """
        if not self.store_path.exists():
            logger.info("No user-token store found at %s", self.store_path)
            return

        try:
            data = json.loads(read_maybe_encrypted(self.store_path).decode("utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to load user-token store: %s", exc)
            return

        self._tokens = data.get("tokens", {})
        logger.info("Loaded user-token store: %d entries", len(self._tokens))

    def _save(self) -> None:
        """Atomically persist the token store to disk, encrypted when possible.

        Follows the deployment-wide AUTO encryption mode: writes plaintext when
        no daily key is loaded (Security OFF), age-encrypted when a daily key is
        loaded (Security ON).  Both states are valid; mixed state is caught at
        startup by ``assert_mode_consistency``.

        In Security ON mode a TOCTOU guard verifies the written file is an age
        envelope.  If the daily key was evicted between the pre-write check and
        the write the file is removed and a ``RuntimeError`` is raised so no
        plaintext credential silently lands on disk.

        Caller must hold ``self._lock``.

        Raises
        ------
        RuntimeError
            Only in Security ON mode: when the written file is not an age
            envelope, indicating a key-eviction race.  The file is removed
            before raising.
        """
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": _STORE_VERSION, "tokens": self._tokens},
            indent=2,
        ).encode("utf-8")
        key_was_loadable = daily_identity_loadable(DAILY_KEY_PATH_DEFAULT)
        write_infra_bytes(self.store_path, payload)
        # TOCTOU guard: if the daily key was loadable at pre-write check, verify
        # the written file is an age envelope.  A key eviction between the two
        # checks would cause a silent plaintext write — this makes it fail loudly
        # and removes the file so no plaintext credential lands on disk.
        if key_was_loadable:
            try:
                on_disk = self.store_path.read_bytes()
            except OSError:
                on_disk = b""
            if not on_disk.startswith(b"age-encryption.org"):
                try:
                    self.store_path.unlink(missing_ok=True)
                except OSError:
                    pass
                # Clear in-memory state so RAM does not stay out of sync with
                # the now-deleted on-disk file.
                self._tokens = {}
                raise RuntimeError(
                    "user-token store written in plaintext — aborting. "
                    "The daily encryption key was evicted between the pre-write "
                    "check and the write.  File removed.  "
                    "Re-set PARAMEM_DAILY_PASSPHRASE and retry."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mint(self, speaker_id: str, label: str) -> str:
        """Mint a new opaque bearer token for *speaker_id*.

        The plaintext token is returned **once** and is not stored.  Internal
        storage and all disk I/O use only the SHA-256 hash of the token.

        Parameters
        ----------
        speaker_id:
            The speaker this token authenticates (e.g. ``"Speaker0"``).
        label:
            Human-readable device or purpose label (e.g. ``"Alice iPad"``).

        Returns
        -------
        str
            The plaintext token.  Present this in
            ``Authorization: Bearer <token>`` on subsequent requests.

        Raises
        ------
        RuntimeError
            In Security ON mode only: if a key-eviction race causes the store
            to be written in plaintext (see :meth:`_save`).
        """
        token = secrets.token_urlsafe(32)
        key = _sha256hex(token)
        entry: dict = {
            "speaker_id": speaker_id,
            "label": label,
            "created": datetime.now(timezone.utc).isoformat(),
            "revoked": False,
        }
        with self._lock:
            self._tokens[key] = entry
            self._save()
        logger.info(
            "Minted token for speaker_id=%s label=%r",
            speaker_id,
            label,
        )
        return token

    def lookup(self, token: str) -> str | None:
        """Look up the ``speaker_id`` for a presented bearer token.

        Computes ``sha256(token)`` and does a direct dict lookup — no
        iteration, no constant-time loop needed (the hash key leaks nothing).

        Parameters
        ----------
        token:
            The plaintext bearer token as presented in the ``Authorization``
            header or cookie.

        Returns
        -------
        str | None
            The ``speaker_id`` if the token is known and not revoked;
            ``None`` otherwise.
        """
        key = _sha256hex(token)
        with self._lock:
            entry = self._tokens.get(key)
        if entry is None:
            return None
        if entry.get("revoked") is True:
            return None
        return entry["speaker_id"]

    def revoke_token(self, token: str) -> bool:
        """Revoke exactly the token whose plaintext value is *token*.

        Parameters
        ----------
        token:
            The plaintext bearer token to revoke.

        Returns
        -------
        bool
            ``True`` if the token was found and was not already revoked;
            ``False`` if the token was unknown or already revoked.

        Raises
        ------
        RuntimeError
            In Security ON mode only: if a key-eviction race causes the store
            to be written in plaintext (see :meth:`_save`).
        """
        key = _sha256hex(token)
        revoked = False
        with self._lock:
            entry = self._tokens.get(key)
            if entry is not None and not entry["revoked"]:
                entry["revoked"] = True
                revoked = True
                self._save()
        if revoked:
            logger.info("Revoked token (sha256 prefix %s…)", key[:8])
        return revoked

    def revoke_speaker(self, speaker_id: str) -> int:
        """Revoke all non-revoked tokens belonging to *speaker_id*.

        Parameters
        ----------
        speaker_id:
            The speaker whose tokens should all be revoked.

        Returns
        -------
        int
            Number of tokens newly marked as revoked.

        Raises
        ------
        RuntimeError
            In Security ON mode only: if a key-eviction race causes the store
            to be written in plaintext (see :meth:`_save`).
        """
        count = 0
        with self._lock:
            for entry in self._tokens.values():
                if entry["speaker_id"] == speaker_id and not entry["revoked"]:
                    entry["revoked"] = True
                    count += 1
            if count:
                self._save()
        if count:
            logger.info("Revoked %d token(s) for speaker_id=%r", count, speaker_id)
        return count

    def list(self) -> list[dict]:
        """Return a public view of all token entries.

        Each dict contains ``speaker_id``, ``label``, ``created``, and
        ``revoked``.  The SHA-256 hash and any plaintext token are never
        included.

        Returns
        -------
        list[dict]
            Snapshot of all entries; order is insertion order (Python 3.7+).
        """
        with self._lock:
            return [
                {
                    "speaker_id": e["speaker_id"],
                    "label": e["label"],
                    "created": e["created"],
                    "revoked": e["revoked"],
                }
                for e in self._tokens.values()
            ]

    def revoke_label(self, label: str) -> int:
        """Revoke all non-revoked tokens whose device label matches *label* exactly.

        Parameters
        ----------
        label:
            The exact device or purpose label to match (e.g. ``"Alice iPad"``).

        Returns
        -------
        int
            Number of tokens newly marked as revoked.

        Raises
        ------
        RuntimeError
            In Security ON mode only: if a key-eviction race causes the store
            to be written in plaintext (see :meth:`_save`).
        """
        count = 0
        with self._lock:
            for entry in self._tokens.values():
                if entry["label"] == label and not entry["revoked"]:
                    entry["revoked"] = True
                    count += 1
            if count:
                self._save()
        if count:
            logger.info("Revoked %d token(s) with label=%r", count, label)
        return count

    def has_active_tokens(self) -> bool:
        """Return ``True`` if any entry has ``revoked == False``.

        Used for the startup posture log
        (:func:`paramem.server.auth.log_startup_posture`) — indicates how many
        active tokens exist.  **Not** used for per-request auth enablement; the
        middleware keys enablement on store *presence* (``store is not None``),
        not token count, so that a wired-but-empty store remains fail-closed.
        """
        with self._lock:
            return any(not e["revoked"] for e in self._tokens.values())
