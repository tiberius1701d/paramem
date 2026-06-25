"""Per-user opaque bearer-token store for the ParaMem REST server.

Tokens are minted as ``secrets.token_urlsafe(32)`` values and stored by
SHA-256 hash — the plaintext token is returned exactly once at mint time
and is never persisted.

On-disk schema (one mandatory tier, no optional buckets):

.. code-block:: json

    {
        "version": 2,
        "tokens": {
            "<sha256hex>": {
                "speaker_id": "speaker0",
                "label": "Alice iPad",
                "created": "<iso8601 utc>",
                "revoked": false,
                "scope": "chat"
            }
        }
    }

The ``scope`` field is optional for backward compatibility with Phase-1 records.
Missing ``scope`` is read as ``"chat"`` at runtime (secure default).  New tokens
are always minted with an explicit scope.

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
from paramem.graph.name_match import is_speaker_id

logger = logging.getLogger(__name__)

_STORE_VERSION = 2
_ALLOWED_SCOPES = ("admin", "chat")
_DEFAULT_SCOPE = "chat"


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
        # sha256hex → {"speaker_id": str|None, "label": str, "created": str,
        #               "revoked": bool, "scope": str}
        self._tokens: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._mtime: int | None = None
        self._load()
        self._mtime = self._current_mtime()

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

        version = data.get("version", 1)
        self._tokens = data.get("tokens", {})
        if version < _STORE_VERSION:
            # v1 → v2: lowercase every token's speaker_id if it is a speaker id.
            # Live tokens confirmed to carry cased "Speaker0" — must rekey so that
            # probe-filter comparisons (speaker_id != bk_spk) remain valid after
            # the speaker profile store is also rekeyed to lowercase (A2 / v6).
            for entry in self._tokens.values():
                sid = entry.get("speaker_id")
                if sid is not None and is_speaker_id(sid):
                    entry["speaker_id"] = sid.lower()
            logger.info(
                "Migrated user-token store v%d → v%d (lowercase speaker_id)",
                version,
                _STORE_VERSION,
            )
            self._save()
        logger.info("Loaded user-token store: %d entries", len(self._tokens))

    def _current_mtime(self) -> int | None:
        """Return the store file's mtime_ns, or None if the file is absent.

        Boundary I/O — OSError (e.g. file not found) returns None, mirroring
        the tolerant ``_load`` "missing file ignored" stance.
        """
        try:
            return self.store_path.stat().st_mtime_ns
        except OSError:
            return None

    def _maybe_reload(self) -> None:
        """Reload the token store from disk if the file mtime has changed.

        Called at the top of :meth:`resolve` (and through delegation, at the
        top of :meth:`lookup`) so that out-of-process writes (CLI mint/revoke)
        take effect on the running server without a restart.

        Stat + compare + reload are performed inside a single critical section
        to avoid a TOCTOU race where two threads each read a changed mtime and
        both reload simultaneously.

        Self-writes via :meth:`_save` stamp ``self._mtime`` under the lock they
        already hold, so the next call here is a no-op (mtime matches).
        """
        with self._lock:
            disk_mtime = self._current_mtime()
            if disk_mtime is not None and disk_mtime != self._mtime:
                self._load()
                self._mtime = disk_mtime

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
                # Wipe self._tokens — intended fail-loud, fail-closed
                # behaviour: the server goes fail-closed (all auth fails) until
                # the store is rebuilt by a fresh CLI mint.  Leaving stale RAM
                # tokens in place would be silently dangerous after an
                # encryption-key eviction race.
                self._tokens = {}
                raise RuntimeError(
                    "user-token store written in plaintext — aborting. "
                    "The daily encryption key was evicted between the pre-write "
                    "check and the write.  File removed.  "
                    "Re-set PARAMEM_DAILY_PASSPHRASE and retry."
                )
        # Stamp the in-process mtime so the self-write does not trigger a
        # spurious reload on the next resolve() / lookup() call.
        self._mtime = self._current_mtime()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mint(self, speaker_id: str | None, label: str, scope: str = _DEFAULT_SCOPE) -> str:
        """Mint a new opaque bearer token.

        The plaintext token is returned **once** and is not stored.  Internal
        storage and all disk I/O use only the SHA-256 hash of the token.

        Parameters
        ----------
        speaker_id:
            The speaker this token authenticates.  MUST be a ``speaker{N}``
            id (e.g. ``"speaker0"``), or ``None`` for an unattributed token
            (shared device that identifies speakers by voice embedding at
            request time).  A non-``None`` id that does not match the
            ``speaker{N}`` form raises :exc:`ValueError` — all minted tokens
            must reference a canonical id so they can participate in
            graph-side speaker resolution.
        label:
            Human-readable device or purpose label (e.g. ``"Alice iPad"``).
        scope:
            Capability scope: ``"chat"`` (conversational endpoints only, the
            secure default) or ``"admin"`` (all endpoints, including operational
            ones like ``/gpu/*``, ``/consolidate``, ``/backup/*``).  Validated
            against :data:`_ALLOWED_SCOPES`; raises :exc:`ValueError` on an
            unrecognised value.

        Returns
        -------
        str
            The plaintext token.  Present this in
            ``Authorization: Bearer <token>`` on subsequent requests.

        Raises
        ------
        ValueError
            If *scope* is not one of the allowed values, or if *speaker_id*
            is a non-``None`` string that does not conform to ``speaker{N}``.
        RuntimeError
            In Security ON mode only: if a key-eviction race causes the store
            to be written in plaintext (see :meth:`_save`).
        """
        if scope not in _ALLOWED_SCOPES:
            raise ValueError(f"Invalid scope {scope!r}; must be one of {_ALLOWED_SCOPES}")
        if speaker_id is not None and not is_speaker_id(speaker_id):
            raise ValueError(
                f"mint: speaker_id={speaker_id!r} is not a canonical speaker{{N}} id. "
                "Tokens must reference a canonical speaker{N} id (e.g. 'speaker0') or "
                "pass speaker_id=None for an unattributed shared-device token."
            )
        token = secrets.token_urlsafe(32)
        key = _sha256hex(token)
        entry: dict = {
            "speaker_id": speaker_id,
            "label": label,
            "created": datetime.now(timezone.utc).isoformat(),
            "revoked": False,
            "scope": scope,
        }
        with self._lock:
            self._tokens[key] = entry
            self._save()
        logger.info(
            "Minted token for speaker_id=%s label=%r scope=%r",
            speaker_id,
            label,
            scope,
        )
        return token

    def lookup(self, token: str) -> str | None:
        """Look up the ``speaker_id`` for a presented bearer token.

        Delegates to :meth:`resolve` so the live-reload hook fires here too.
        Returns the ``speaker_id`` (a string) for an attributed, active token;
        returns ``None`` for unknown, revoked, or **unattributed** tokens
        (``speaker_id=None``).  The ``None`` return for unattributed tokens is
        intentional — :meth:`lookup` preserves its historic contract
        (``str | None`` where ``None`` means "no speaker") and is not the auth
        primitive for unattributed tokens.  :meth:`resolve` is used for that.

        Parameters
        ----------
        token:
            The plaintext bearer token as presented in the ``Authorization``
            header or cookie.

        Returns
        -------
        str | None
            The ``speaker_id`` if the token is known, not revoked, and
            attributed; ``None`` otherwise (unknown, revoked, or unattributed).
        """
        record = self.resolve(token)
        if record is None:
            return None
        _authenticated, speaker_id, _scope = record
        # speaker_id may be None for unattributed tokens — return None per
        # the existing contract (callers that only care about attributed tokens
        # see no change; the middleware switches to resolve() for unattributed).
        return speaker_id

    def resolve(self, token: str) -> tuple[bool, str | None, str] | None:
        """Resolve a presented token to ``(authenticated, speaker_id, scope)``.

        This is the single auth/validity primitive used by the middleware.
        Unlike :meth:`lookup` it distinguishes a valid unattributed token
        (``speaker_id=None``) from an unknown/revoked token (returns ``None``).

        Also triggers the mtime-based live-reload so out-of-process writes
        (CLI mint / revoke) take effect without a server restart.

        Parameters
        ----------
        token:
            The plaintext bearer token.

        Returns
        -------
        tuple[bool, str | None, str] | None
            ``(True, speaker_id_or_None, scope)`` for a valid, non-revoked
            token, where ``scope`` defaults to :data:`_DEFAULT_SCOPE` for
            records minted before the scope field existed (Phase-1 records).
            ``None`` when the token is unknown or revoked.
        """
        self._maybe_reload()
        key = _sha256hex(token)
        with self._lock:
            entry = self._tokens.get(key)
        if entry is None:
            return None
        if entry.get("revoked") is True:
            return None
        return (True, entry.get("speaker_id"), entry.get("scope", _DEFAULT_SCOPE))

    def revoke_token(self, token: str) -> bool:
        """Revoke exactly the token whose plaintext value is *token*.

        Note: assumes a freshly-loaded store (the CLI constructs a new
        ``UserTokenStore`` per invocation) and does not call
        ``_maybe_reload()`` before mutating.

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

        Note: assumes a freshly-loaded store (the CLI constructs a new
        ``UserTokenStore`` per invocation) and does not call
        ``_maybe_reload()`` before mutating.

        Skips entries whose ``speaker_id`` is ``None`` (unattributed tokens)
        so a ``None == None`` match cannot bulk-revoke all unattributed tokens.
        Use :meth:`revoke_label` to revoke unattributed tokens by their label.

        Parameters
        ----------
        speaker_id:
            The speaker whose tokens should all be revoked.  Must not be
            ``None`` — raises :exc:`ValueError` to prevent accidental
            bulk-revocation of unattributed tokens.

        Returns
        -------
        int
            Number of tokens newly marked as revoked.

        Raises
        ------
        ValueError
            If *speaker_id* is ``None``.
        RuntimeError
            In Security ON mode only: if a key-eviction race causes the store
            to be written in plaintext (see :meth:`_save`).
        """
        if speaker_id is None:
            raise ValueError(
                "revoke_speaker(None) is not allowed — it would match all "
                "unattributed tokens.  Use revoke_label() to revoke an "
                "unattributed token by its label instead."
            )
        count = 0
        with self._lock:
            for entry in self._tokens.values():
                # Skip unattributed entries (speaker_id is None) so they are
                # never accidentally matched by this method.
                if entry.get("speaker_id") is None:
                    continue
                # Comparison is intentionally EXACT (case-sensitive): normalising
                # would risk merging distinct speakers.  Use
                # ``revoke-user-token --list`` to see canonical IDs.
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

        Each dict contains ``speaker_id``, ``label``, ``created``, ``revoked``,
        and ``scope``.  The SHA-256 hash and any plaintext token are never
        included.  Pre-scope-field records (Phase-1) surface ``scope="chat"``
        (secure read-time default).

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
                    "scope": e.get("scope", _DEFAULT_SCOPE),
                }
                for e in self._tokens.values()
            ]

    def revoke_label(self, label: str) -> int:
        """Revoke all non-revoked tokens whose device label matches *label* exactly.

        Note: assumes a freshly-loaded store (the CLI constructs a new
        ``UserTokenStore`` per invocation) and does not call
        ``_maybe_reload()`` before mutating.

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
