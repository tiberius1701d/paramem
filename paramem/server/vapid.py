"""VAPID keypair lifecycle and signing helper for Web Push.

The keypair is persisted as a Tier-2 infrastructure file (same encryption
posture as ``user_tokens.json``): age-encrypted when a daily identity is loaded
(Security ON), plaintext JSON otherwise (Security OFF).  Mixed state is caught
at startup by :func:`~paramem.backup.encryption.assert_mode_consistency` via
:func:`~paramem.backup.encryption.infra_paths`.

On-disk schema (``vapid_keys.json``):

.. code-block:: json

    {
        "private_key_pem": "-----BEGIN EC PRIVATE KEY-----\\n...\\n-----END EC PRIVATE KEY-----\\n"
    }

The public key is *not* persisted — it is derived on every load from the
private key.  This avoids any risk of public/private key skew on disk.

VAPID key stability
-------------------
The ``applicationServerKey`` delivered to browsers and the VAPID JWT ``aud``
claim are derived from the same EC P-256 private key.  Replacing
``vapid_keys.json`` invalidates all existing browser subscriptions; operators
should treat the keypair as effectively immutable once browsers have subscribed.

Security properties
-------------------
- The keypair is auto-generated on first startup when ``push_enabled`` is true.
- The private key PEM is written via :func:`~paramem.backup.encryption.write_infra_bytes`
  (age-encrypted when a daily key is loaded).
- ``application_server_key`` returns the unpadded base64url-encoded
  uncompressed EC point (65 bytes, ``0x04`` prefix) required by the
  ``PushManager.subscribe()`` browser API.
- The VAPID JWT is signed per-request with claims ``aud`` (endpoint origin),
  ``sub`` (configured contact URI), ``exp`` (now + 12 h).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def vapid_keys_path(data_dir: Path) -> Path:
    """Return the canonical path for the VAPID keypair file.

    Parameters
    ----------
    data_dir:
        Root of the ParaMem data directory.

    Returns
    -------
    Path
        ``<data_dir>/vapid_keys.json``
    """
    return Path(data_dir) / "vapid_keys.json"


def ensure_vapid_keypair(data_dir: Path):
    """Load or generate the VAPID keypair and return a loaded Vapid handle.

    If ``vapid_keys.json`` is absent, generates a new EC P-256 keypair via
    :class:`py_vapid.Vapid`, serialises the private key as PEM, and persists
    it via :func:`~paramem.backup.encryption.write_infra_bytes` (age-encrypted
    when a daily key is loaded, plaintext otherwise).  Idempotent: a second
    call with the same ``data_dir`` returns the same public key without
    modifying the file.

    Parameters
    ----------
    data_dir:
        Root of the ParaMem data directory.

    Returns
    -------
    py_vapid.Vapid
        A loaded Vapid instance ready for signing.

    Raises
    ------
    RuntimeError
        If the keypair file is an age envelope but the daily identity is not
        loaded (propagated from :func:`~paramem.backup.encryption.read_maybe_encrypted`).
    ImportError
        If ``py-vapid`` is not installed.
    """
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
    )
    from py_vapid import Vapid

    from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes

    path = vapid_keys_path(data_dir)

    if path.exists():
        data = json.loads(read_maybe_encrypted(path).decode("utf-8"))
        handle = Vapid.from_pem(data["private_key_pem"].encode("ascii"))
        logger.info("Loaded VAPID keypair from %s", path)
        return handle

    # Generate a new keypair.
    handle = Vapid()
    handle.generate_keys()

    private_pem = handle.private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=NoEncryption(),
    ).decode("ascii")

    payload = json.dumps({"private_key_pem": private_pem}, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_infra_bytes(path, payload)
    logger.info(
        "Generated new VAPID keypair and persisted to %s (public key: %s)",
        path,
        application_server_key(handle),
    )
    return handle


def application_server_key(handle) -> str:
    """Return the unpadded base64url public key for use as ``applicationServerKey``.

    The value is the uncompressed EC point (65 bytes, ``0x04`` prefix) encoded
    as unpadded base64url.  This is the exact format required by the browser's
    ``PushManager.subscribe({applicationServerKey: ...})`` call.

    Parameters
    ----------
    handle:
        A loaded :class:`py_vapid.Vapid` instance.

    Returns
    -------
    str
        Unpadded base64url-encoded uncompressed EC P-256 public key.
    """
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pub_bytes = handle.public_key.public_bytes(
        encoding=Encoding.X962,
        format=PublicFormat.UncompressedPoint,
    )
    return base64.urlsafe_b64encode(pub_bytes).rstrip(b"=").decode("ascii")


def vapid_authorization_header(handle, endpoint: str, contact: str) -> str:
    """Build the VAPID ``Authorization`` header value for a push request.

    Constructs claims with ``aud`` set to the endpoint's scheme+host, ``sub``
    set to *contact* (typically ``mailto:admin@...``), and ``exp`` set to
    now + 12 hours.  Signs with the loaded private key via
    :meth:`py_vapid.Vapid.sign` (RFC 8292 ``vapid t=...,k=...`` format).

    Parameters
    ----------
    handle:
        A loaded :class:`py_vapid.Vapid` instance.
    endpoint:
        The push subscription endpoint URL (used to derive ``aud``).
    contact:
        The JWT ``sub`` claim — a ``mailto:`` URI identifying the server
        operator (from ``mobile_pwa.vapid_contact`` in the server config).

    Returns
    -------
    str
        The full ``Authorization`` header value, e.g.
        ``vapid t=<jwt>,k=<pubkey>``.
    """
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    aud = f"{parsed.scheme}://{parsed.netloc}"

    claims = {
        "sub": contact,
        "aud": aud,
        "exp": int(time.time()) + 12 * 3600,
    }
    signed = handle.sign(claims)
    return signed["Authorization"]
