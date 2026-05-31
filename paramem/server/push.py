"""Web Push subscription registry and send helper.

``PushSubscriptionStore`` mirrors :class:`~paramem.server.user_tokens.UserTokenStore`
in structure and encryption posture: subscriptions are keyed per speaker, written
via :func:`~paramem.backup.encryption.write_infra_bytes` (age-encrypted when a
daily key is loaded), and read via
:func:`~paramem.backup.encryption.read_maybe_encrypted`.  A TOCTOU verify-after-write
guard identical to the token store's guards against key-eviction races.

On-disk schema (``push_subscriptions.json``):

.. code-block:: json

    {
        "version": 1,
        "subscriptions": {
            "<speaker_id>": [
                {"endpoint": "https://...", "keys": {"p256dh": "...", "auth": "..."}}
            ]
        }
    }

Subscriptions are deduplicated per speaker by endpoint URL.  An endpoint that
appears more than once for the same speaker is silently ignored on ``add``.

Send helper
-----------
:func:`send_ping` posts a contentless push notification via
``httpx.Client(http2=True)`` (requires the ``h2`` package for HTTP/2).  Only
the VAPID ``Authorization`` header and a ``TTL`` header are sent — no body,
no personal content.  The push payload is intentionally empty: the service
worker's ``push`` event handler shows a generic notification badge; real
content is fetched by the client after the user taps.

Security properties
-------------------
- Subscription endpoints are public push relay URLs; no personal content is
  transmitted through the push relay.
- The store follows the deployment-wide AUTO encryption mode identical to
  ``user_tokens.json`` and ``vapid_keys.json``.
- Mixed state (plaintext file while a daily key is loaded) is caught at
  startup by ``assert_mode_consistency`` via ``infra_paths()``.
- A TOCTOU guard in :meth:`PushSubscriptionStore._save` additionally catches
  key eviction between the pre-write loadability check and the write.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import threading
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

_STORE_VERSION = 1
# TTL in seconds — how long the push relay holds the notification if the device
# is offline.  60 seconds is appropriate for a conversational assistant ping.
_PUSH_TTL = "60"


def _validate_subscription(subscription: dict) -> None:
    """Validate a Web Push subscription dict before storage.

    Raises ``ValueError`` with a descriptive message on any of:

    - ``endpoint`` is not a ``str`` or does not start with ``https://``
      (TLS required; plain HTTP endpoints are rejected).
    - ``endpoint`` host is loopback, private, link-local, or reserved
      (static SSRF guard — no DNS resolution attempted).
    - ``keys`` is missing a non-empty ``p256dh`` or ``auth`` value
      (both are W3C-required for encrypted Web Push).

    Parameters
    ----------
    subscription:
        Push subscription dict as sent by the browser's
        ``PushSubscription.toJSON()``.

    Raises
    ------
    ValueError
        When the subscription fails any of the above checks.
    """
    endpoint = subscription.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
        raise ValueError(
            f"push subscription endpoint must be a str starting with 'https://'; got {endpoint!r}"
        )

    parsed = urllib.parse.urlparse(endpoint)
    host = parsed.hostname or ""

    # Reject bare hostnames (no dot) and .local mDNS names.
    if host == "localhost" or host.endswith(".local") or (host and "." not in host):
        raise ValueError(f"push subscription endpoint host is not a public push relay: {host!r}")

    # Reject IP literals that are loopback, private, link-local, reserved, or unspecified.
    try:
        addr = ipaddress.ip_address(host)
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_unspecified
        ):
            raise ValueError(
                f"push subscription endpoint resolves to a non-public IP address: {host!r}"
            )
    except ValueError as exc:
        # Re-raise our own ValueError from the IP check above.
        if "push subscription endpoint" in str(exc):
            raise
        # host is not an IP literal — hostname path, already checked above.

    keys = subscription.get("keys")
    if not isinstance(keys, dict):
        raise ValueError("push subscription 'keys' must be a dict with 'p256dh' and 'auth'")
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")
    if not p256dh or not isinstance(p256dh, str):
        raise ValueError("push subscription 'keys.p256dh' must be a non-empty string")
    if not auth or not isinstance(auth, str):
        raise ValueError("push subscription 'keys.auth' must be a non-empty string")


class PushSubscriptionStore:
    """Persistent per-speaker Web Push subscription store.

    Subscriptions are stored keyed by ``speaker_id``.  Each speaker may hold
    multiple subscriptions (one per browser/device).  Deduplication is by
    endpoint URL — re-adding the same endpoint is a no-op.

    All public methods are thread-safe via an internal :class:`threading.Lock`.

    Parameters
    ----------
    store_path:
        Path to the JSON file on disk.  Parent directory is created on first
        write.  The file is envelope-encrypted when a daily age identity is
        loadable (see :func:`~paramem.backup.encryption.write_infra_bytes`).
    """

    def __init__(self, store_path: Path | str) -> None:
        self.store_path = Path(store_path)
        # speaker_id → list of {"endpoint": str, "keys": {"p256dh": str, "auth": str}}
        self._subscriptions: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the subscription store from disk.

        Missing file is silently ignored (fresh deployment).  Parse and I/O
        errors are logged and result in an empty in-memory store.
        """
        if not self.store_path.exists():
            logger.info("No push subscription store found at %s", self.store_path)
            return

        from paramem.backup.encryption import read_maybe_encrypted

        try:
            data = json.loads(read_maybe_encrypted(self.store_path).decode("utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to load push subscription store: %s", exc)
            return

        self._subscriptions = data.get("subscriptions", {})
        total = sum(len(v) for v in self._subscriptions.values())
        logger.info(
            "Loaded push subscription store: %d speaker(s), %d subscription(s)",
            len(self._subscriptions),
            total,
        )

    def _save(self) -> None:
        """Atomically persist the subscription store to disk, encrypted when possible.

        Follows the deployment-wide AUTO encryption mode: writes plaintext when
        no daily key is loaded (Security OFF), age-encrypted when a daily key is
        loaded (Security ON).  Both states are valid; mixed state is caught at
        startup by ``assert_mode_consistency``.

        In Security ON mode a TOCTOU guard verifies the written file is an age
        envelope.  If the daily key was evicted between the pre-write check and
        the write the file is removed and a ``RuntimeError`` is raised so no
        plaintext subscription data silently lands on disk.

        Caller must hold ``self._lock``.

        Raises
        ------
        RuntimeError
            Only in Security ON mode: when the written file is not an age
            envelope, indicating a key-eviction race.  The file is removed
            before raising.
        """
        from paramem.backup.encryption import write_infra_bytes
        from paramem.backup.key_store import DAILY_KEY_PATH_DEFAULT, daily_identity_loadable

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": _STORE_VERSION, "subscriptions": self._subscriptions},
            indent=2,
        ).encode("utf-8")
        key_was_loadable = daily_identity_loadable(DAILY_KEY_PATH_DEFAULT)
        write_infra_bytes(self.store_path, payload)
        # TOCTOU guard: identical posture to UserTokenStore._save.
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
                self._subscriptions = {}
                raise RuntimeError(
                    "push subscription store written in plaintext — aborting. "
                    "The daily encryption key was evicted between the pre-write "
                    "check and the write.  File removed.  "
                    "Re-set PARAMEM_DAILY_PASSPHRASE and retry."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, speaker_id: str, subscription: dict) -> bool:
        """Add a push subscription for *speaker_id*, deduplicating by endpoint.

        If a subscription with the same ``endpoint`` already exists for this
        speaker, the call is a no-op and returns ``False``.

        Parameters
        ----------
        speaker_id:
            The speaker who owns this subscription.
        subscription:
            Push subscription dict with at least ``endpoint`` and ``keys``
            (``p256dh``, ``auth``).

        Returns
        -------
        bool
            ``True`` if a new subscription was added; ``False`` if it already
            existed (deduplicated).

        Raises
        ------
        ValueError
            When the subscription fails endpoint or keys validation (see
            :func:`_validate_subscription`).
        RuntimeError
            In Security ON mode only: if a key-eviction race causes the store
            to be written in plaintext (see :meth:`_save`).
        """
        _validate_subscription(subscription)
        endpoint = subscription.get("endpoint", "")
        with self._lock:
            speaker_subs = self._subscriptions.setdefault(speaker_id, [])
            for existing in speaker_subs:
                if existing.get("endpoint") == endpoint:
                    logger.debug(
                        "push subscription already registered for speaker=%s endpoint=%s…",
                        speaker_id,
                        endpoint[:60],
                    )
                    return False
            speaker_subs.append(subscription)
            self._save()
        logger.info(
            "Registered push subscription for speaker=%s endpoint=%s…",
            speaker_id,
            endpoint[:60],
        )
        return True

    def list(self, speaker_id: str) -> list[dict]:
        """Return all subscriptions for *speaker_id*.

        Parameters
        ----------
        speaker_id:
            The speaker whose subscriptions to retrieve.

        Returns
        -------
        list[dict]
            Snapshot of subscription dicts (may be empty).
        """
        with self._lock:
            return list(self._subscriptions.get(speaker_id, []))

    def all(self) -> dict[str, list[dict]]:
        """Return all subscriptions keyed by speaker_id.

        Returns
        -------
        dict[str, list[dict]]
            Snapshot of the full subscriptions map.
        """
        with self._lock:
            return {k: list(v) for k, v in self._subscriptions.items()}

    def remove(self, endpoint: str) -> int:
        """Remove all subscriptions matching *endpoint* across all speakers.

        Used to prune dead/expired subscriptions (e.g. after a 404/410 response
        from the push relay).

        Parameters
        ----------
        endpoint:
            The push endpoint URL to remove.

        Returns
        -------
        int
            Number of subscriptions removed.

        Raises
        ------
        RuntimeError
            In Security ON mode only: if a key-eviction race causes the store
            to be written in plaintext (see :meth:`_save`).
        """
        removed = 0
        with self._lock:
            for speaker_id, subs in list(self._subscriptions.items()):
                before = len(subs)
                subs[:] = [s for s in subs if s.get("endpoint") != endpoint]
                removed += before - len(subs)
                if not subs:
                    del self._subscriptions[speaker_id]
            if removed:
                self._save()
        if removed:
            logger.info("Pruned %d subscription(s) for endpoint=%s…", removed, endpoint[:60])
        return removed


def send_ping(subscription: dict, vapid_handle, contact: str) -> tuple[str, int | None]:
    """Send a contentless Web Push ping to a single subscription.

    The push carries no body — only ``Authorization`` (VAPID JWT) and ``TTL``
    headers are set.  The service worker's ``push`` event shows a generic
    notification badge; real content is fetched by the client on tap.

    Transport is ``httpx.Client(http2=True)`` (requires the ``h2`` package),
    which negotiates HTTP/2 with ``web.push.apple.com`` via TLS ALPN.  HTTP/1.1
    is rejected by Apple's push endpoint at the protocol layer.

    Parameters
    ----------
    subscription:
        Push subscription dict with ``endpoint`` (and optional ``keys``).
    vapid_handle:
        A loaded :class:`py_vapid.Vapid` instance.
    contact:
        The VAPID JWT ``sub`` claim (e.g. ``"mailto:admin@localhost"``).

    Returns
    -------
    tuple[str, int | None]
        ``(http_version, status_code)`` where ``http_version`` is ``"HTTP/2"``
        when the transport is working correctly.  ``status_code`` is ``None``
        only on a network-level exception; in that case ``http_version`` carries
        a descriptive error string.

    Notes
    -----
    - A 404 or 410 response means the subscription is expired or revoked — the
      caller should prune it via
      :meth:`PushSubscriptionStore.remove`.
    - A 201 response is the success code from most push relays.
    - Network-level failures (connection refused, DNS failure, TLS error) are
      boundary-handled: they return an error string in ``http_version`` and
      ``None`` in ``status_code``.  They are NOT silently swallowed — the caller
      can inspect and decide whether to retry or log.
    """
    # SECURITY: re-validate endpoint host + drop the manual Content-Length
    # before wiring a producer (see review)
    import httpx

    from paramem.server.vapid import vapid_authorization_header

    endpoint = subscription["endpoint"]
    auth_header = vapid_authorization_header(vapid_handle, endpoint, contact)

    headers = {
        "Authorization": auth_header,
        "TTL": _PUSH_TTL,
        "Content-Length": "0",
    }

    try:
        with httpx.Client(http2=True) as client:
            response = client.post(endpoint, headers=headers)
        return (response.http_version, response.status_code)
    except httpx.HTTPError as exc:
        # Boundary error: network-level failure (connection refused, DNS failure,
        # TLS error, protocol mismatch).  NOT control-flow suppression — real
        # transport problems from a dead or unreachable endpoint.
        error_str = f"network_error:{type(exc).__name__}: {exc}"
        logger.warning("push send failed for endpoint=%s…: %s", endpoint[:60], exc)
        return (error_str, None)
