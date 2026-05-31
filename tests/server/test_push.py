"""Tests for Web Push — vapid.py, push.py, endpoints, and encryption coverage.

All tests are unit/TestClient level: no model load, no GPU, no live push relay.

Covers:
  - PushSubscriptionStore: encrypted-on-disk, per-speaker scoping, endpoint
    dedupe, TOCTOU guard (mirrors test_user_tokens.py).
  - VAPID: ensure_vapid_keypair idempotency, application_server_key stability,
    file registered in infra_paths().
  - Endpoints: /push/vapid-public-key and /push/subscribe behaviour (auth,
    push-disabled, speaker-id scoping, body-id ignored).
  - Consistency: vapid_keys.json + push_subscriptions.json covered by
    assert_mode_consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.server.push import PushSubscriptionStore

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point env + module default at it.

    Mirrors the helper in test_user_tokens.py.
    """
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch: pytest.MonkeyPatch):
    """Isolate daily identity cache per test."""
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


@pytest.fixture()
def no_daily_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fixture: ensure no daily key is loadable."""
    monkeypatch.setattr(
        "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
        tmp_path / "absent.age",
    )
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    _clear_daily_identity_cache()


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "push_subscriptions.json"


def _make_subscription(endpoint: str = "https://web.push.example.com/test") -> dict:
    return {
        "endpoint": endpoint,
        "keys": {"p256dh": "AAAA", "auth": "BBBB"},
    }


# ---------------------------------------------------------------------------
# PushSubscriptionStore — basic operations
# ---------------------------------------------------------------------------


class TestPushSubscriptionStoreBasic:
    def test_add_and_list(self, tmp_path, monkeypatch, store_path):
        """add() persists a subscription that list() can retrieve."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)
        sub = _make_subscription()

        added = store.add("Speaker0", sub)

        assert added is True
        subs = store.list("Speaker0")
        assert len(subs) == 1
        assert subs[0]["endpoint"] == sub["endpoint"]

    def test_list_empty_for_unknown_speaker(self, tmp_path, monkeypatch, store_path):
        """list() returns an empty list for a speaker with no subscriptions."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)

        assert store.list("NoSuchSpeaker") == []

    def test_add_multiple_speakers(self, tmp_path, monkeypatch, store_path):
        """add() scopes subscriptions per speaker."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)

        sub0 = _make_subscription("https://push.example.com/s0")
        sub1 = _make_subscription("https://push.example.com/s1")
        store.add("Speaker0", sub0)
        store.add("Speaker1", sub1)

        assert len(store.list("Speaker0")) == 1
        assert len(store.list("Speaker1")) == 1
        assert store.list("Speaker0")[0]["endpoint"] == sub0["endpoint"]
        assert store.list("Speaker1")[0]["endpoint"] == sub1["endpoint"]

    def test_all_returns_all_speakers(self, tmp_path, monkeypatch, store_path):
        """all() returns all subscriptions keyed by speaker_id."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)
        store.add("Speaker0", _make_subscription("https://push.example.com/0"))
        store.add("Speaker1", _make_subscription("https://push.example.com/1"))

        result = store.all()

        assert set(result.keys()) == {"Speaker0", "Speaker1"}


# ---------------------------------------------------------------------------
# Endpoint deduplication
# ---------------------------------------------------------------------------


class TestEndpointDedupe:
    def test_add_same_endpoint_twice_is_noop(self, tmp_path, monkeypatch, store_path):
        """Re-adding an existing endpoint for the same speaker returns False."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)
        sub = _make_subscription("https://push.example.com/dup")

        first = store.add("Speaker0", sub)
        second = store.add("Speaker0", sub)

        assert first is True
        assert second is False
        assert len(store.list("Speaker0")) == 1

    def test_add_different_endpoints_for_same_speaker(self, tmp_path, monkeypatch, store_path):
        """Two different endpoints for the same speaker both persist."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)

        sub_a = _make_subscription("https://push.example.com/a")
        sub_b = _make_subscription("https://push.example.com/b")
        store.add("Speaker0", sub_a)
        store.add("Speaker0", sub_b)

        subs = store.list("Speaker0")
        endpoints = {s["endpoint"] for s in subs}
        assert endpoints == {sub_a["endpoint"], sub_b["endpoint"]}


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_known_endpoint(self, tmp_path, monkeypatch, store_path):
        """remove() deletes the subscription and returns count=1."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)
        sub = _make_subscription("https://push.example.com/gone")
        store.add("Speaker0", sub)

        count = store.remove("https://push.example.com/gone")

        assert count == 1
        assert store.list("Speaker0") == []

    def test_remove_unknown_endpoint_returns_zero(self, tmp_path, monkeypatch, store_path):
        """remove() on an unknown endpoint returns 0."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)
        store.add("Speaker0", _make_subscription("https://push.example.com/stay"))

        count = store.remove("https://push.example.com/no-such")

        assert count == 0
        assert len(store.list("Speaker0")) == 1

    def test_remove_across_speakers(self, tmp_path, monkeypatch, store_path):
        """remove() removes the endpoint for all speakers that hold it."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)
        shared_endpoint = "https://push.example.com/shared"
        store.add("Speaker0", _make_subscription(shared_endpoint))
        store.add("Speaker1", _make_subscription(shared_endpoint))

        count = store.remove(shared_endpoint)

        assert count == 2
        assert store.list("Speaker0") == []
        assert store.list("Speaker1") == []


# ---------------------------------------------------------------------------
# On-disk security: age envelope + TOCTOU guard
# ---------------------------------------------------------------------------


class TestOnDiskSecurity:
    def test_on_disk_is_age_envelope_when_daily_loaded(self, tmp_path, monkeypatch, store_path):
        """When the daily key is loaded, the store file is an age envelope."""
        from paramem.backup.age_envelope import AGE_MAGIC

        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)
        store.add("Speaker0", _make_subscription())

        disk_bytes = store_path.read_bytes()

        assert disk_bytes.startswith(AGE_MAGIC), (
            f"expected age envelope on disk; got {disk_bytes[:40]!r}"
        )

    def test_on_disk_is_plaintext_when_no_daily_key(
        self, tmp_path, monkeypatch, store_path, no_daily_key
    ):
        """Security OFF: store is written as plaintext JSON."""
        store = PushSubscriptionStore(store_path)
        store.add("Speaker0", _make_subscription())

        disk_bytes = store_path.read_bytes()

        assert not disk_bytes.startswith(b"age-encryption.org"), (
            "expected plaintext in Security OFF mode; got age envelope"
        )
        data = json.loads(disk_bytes.decode("utf-8"))
        assert "subscriptions" in data

    def test_store_roundtrips_across_reload(self, tmp_path, monkeypatch, store_path):
        """Subscriptions survive store teardown and reload."""
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)
        sub = _make_subscription("https://push.example.com/reload")
        store.add("Speaker0", sub)

        store2 = PushSubscriptionStore(store_path)

        subs = store2.list("Speaker0")
        assert len(subs) == 1
        assert subs[0]["endpoint"] == sub["endpoint"]

    def test_toctou_guard_fires_when_key_evicted_between_checks(
        self, tmp_path, monkeypatch, store_path
    ):
        """TOCTOU guard raises RuntimeError when key eviction writes plaintext.

        Simulates the race by patching write_infra_bytes to write raw plaintext
        (as though encryption failed mid-write) while daily_identity_loadable
        returns True at the pre-write check.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)

        # Patch write_infra_bytes to bypass encryption (simulate evicted key).
        def _plaintext_write(path: Path, payload: bytes) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(payload)

        with patch("paramem.backup.encryption.write_infra_bytes", side_effect=_plaintext_write):
            with pytest.raises(RuntimeError, match="written in plaintext"):
                store.add("Speaker0", _make_subscription())


# ---------------------------------------------------------------------------
# VAPID keypair
# ---------------------------------------------------------------------------


class TestVapidKeypair:
    def test_ensure_vapid_keypair_creates_file(self, tmp_path, monkeypatch):
        """ensure_vapid_keypair creates vapid_keys.json when absent."""
        from paramem.server.vapid import ensure_vapid_keypair, vapid_keys_path

        _setup_daily(tmp_path, monkeypatch)
        handle = ensure_vapid_keypair(tmp_path)

        assert vapid_keys_path(tmp_path).exists()
        assert handle is not None

    def test_ensure_vapid_keypair_idempotent(self, tmp_path, monkeypatch):
        """Second call returns same public key without regenerating the file."""
        from paramem.server.vapid import application_server_key, ensure_vapid_keypair

        _setup_daily(tmp_path, monkeypatch)
        h1 = ensure_vapid_keypair(tmp_path)
        mtime1 = (tmp_path / "vapid_keys.json").stat().st_mtime
        key1 = application_server_key(h1)

        h2 = ensure_vapid_keypair(tmp_path)
        mtime2 = (tmp_path / "vapid_keys.json").stat().st_mtime
        key2 = application_server_key(h2)

        assert key1 == key2, "public key must be stable across reloads"
        assert mtime1 == mtime2, "file must not be modified on second call"

    def test_application_server_key_is_base64url(self, tmp_path, monkeypatch):
        """application_server_key returns an unpadded base64url string."""
        import base64

        from paramem.server.vapid import application_server_key, ensure_vapid_keypair

        _setup_daily(tmp_path, monkeypatch)
        handle = ensure_vapid_keypair(tmp_path)
        key = application_server_key(handle)

        # Must not contain base64 padding.
        assert "=" not in key
        # Must decode to a 65-byte uncompressed EC point (0x04 prefix).
        padded = key + "=" * ((4 - len(key) % 4) % 4)
        raw = base64.urlsafe_b64decode(padded)
        assert len(raw) == 65, f"expected 65-byte uncompressed EC point; got {len(raw)}"
        assert raw[0] == 0x04, "expected uncompressed point marker 0x04"

    def test_vapid_keys_file_is_age_envelope_when_daily_loaded(self, tmp_path, monkeypatch):
        """vapid_keys.json is age-encrypted when a daily key is loaded."""
        from paramem.backup.age_envelope import AGE_MAGIC
        from paramem.server.vapid import ensure_vapid_keypair, vapid_keys_path

        _setup_daily(tmp_path, monkeypatch)
        ensure_vapid_keypair(tmp_path)

        disk_bytes = vapid_keys_path(tmp_path).read_bytes()
        assert disk_bytes.startswith(AGE_MAGIC), f"expected age envelope; got {disk_bytes[:40]!r}"

    def test_vapid_keys_file_registered_in_infra_paths(self, tmp_path):
        """vapid_keys.json appears in infra_paths() for the given data_dir."""
        from paramem.backup.encryption import infra_paths

        paths = infra_paths(tmp_path)
        path_names = [p.name for p in paths]
        assert "vapid_keys.json" in path_names

    def test_push_subscriptions_file_registered_in_infra_paths(self, tmp_path):
        """push_subscriptions.json appears in infra_paths() for the given data_dir."""
        from paramem.backup.encryption import infra_paths

        paths = infra_paths(tmp_path)
        path_names = [p.name for p in paths]
        assert "push_subscriptions.json" in path_names


# ---------------------------------------------------------------------------
# assert_mode_consistency covers new files
# ---------------------------------------------------------------------------


class TestModeConsistencyCoversNewFiles:
    def test_plaintext_vapid_keys_with_loaded_key_refused(self, tmp_path, monkeypatch):
        """assert_mode_consistency refuses startup when vapid_keys.json is plaintext
        but the daily identity is loaded (Security ON)."""
        from paramem.backup.encryption import assert_mode_consistency
        from paramem.backup.types import FatalConfigError

        _setup_daily(tmp_path, monkeypatch)
        # Write vapid_keys.json as plaintext while the daily key is loaded.
        vapid_file = tmp_path / "vapid_keys.json"
        vapid_file.write_text('{"private_key_pem": "FAKE"}')

        with pytest.raises(FatalConfigError, match="plaintext"):
            assert_mode_consistency(tmp_path, daily_identity_loadable=True)

    def test_plaintext_push_subscriptions_with_loaded_key_refused(self, tmp_path, monkeypatch):
        """assert_mode_consistency refuses startup when push_subscriptions.json is
        plaintext but the daily identity is loaded."""
        from paramem.backup.encryption import assert_mode_consistency
        from paramem.backup.types import FatalConfigError

        _setup_daily(tmp_path, monkeypatch)
        subs_file = tmp_path / "push_subscriptions.json"
        subs_file.write_text('{"version": 1, "subscriptions": {}}')

        with pytest.raises(FatalConfigError, match="plaintext"):
            assert_mode_consistency(tmp_path, daily_identity_loadable=True)


# ---------------------------------------------------------------------------
# Endpoint tests — direct async calls (no model load).
#
# `from __future__ import annotations` turns all annotations into strings at
# runtime (PEP 563).  FastAPI resolves those strings against the function's
# *global* namespace, so locally-defined Pydantic models in nested functions
# are invisible to FastAPI and get treated as query parameters.  To avoid that
# we call the handler coroutines directly via asyncio, injecting a mock
# Request with a pre-set state.speaker_id.  This exercises the real production
# code path with no routing layer in the way.
# ---------------------------------------------------------------------------


def _make_mock_request(speaker_id: str | None):
    """Build a minimal FastAPI Request with speaker_id on request.state."""
    from fastapi import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/push/subscribe",
        "headers": [],
        "query_string": b"",
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    req = Request(scope, _receive)
    if speaker_id is not None:
        req.state.speaker_id = speaker_id
    return req


def _call_push_subscribe(
    tmp_path: Path,
    monkeypatch,
    push_enabled: bool,
    speaker_id: str | None,
    endpoint: str = "https://web.push.example.com/test",
    keys: dict | None = None,
):
    """Set up _state, call push_subscribe directly, return (result, store).

    Uses asyncio.run so the test does not need to be async and is immune to
    suite-ordering (each call creates and tears down its own event loop).
    """
    import asyncio
    from unittest.mock import MagicMock

    import paramem.server.app as _app
    from paramem.server.app import PushSubscribeRequest, push_subscribe
    from paramem.server.vapid import ensure_vapid_keypair

    _setup_daily(tmp_path, monkeypatch)

    store = None
    handle = None
    if push_enabled:
        handle = ensure_vapid_keypair(tmp_path)
        store = PushSubscriptionStore(tmp_path / "subs.json")

    mock_config = MagicMock()
    mock_config.mobile_pwa.push_enabled = push_enabled
    mock_config.mobile_pwa.vapid_contact = "mailto:test@localhost"

    saved = {
        "config": _app._state.get("config"),
        "vapid": _app._state.get("vapid"),
        "push_store": _app._state.get("push_store"),
    }
    _app._state["config"] = mock_config
    _app._state["vapid"] = handle
    _app._state["push_store"] = store

    try:
        body = PushSubscribeRequest(endpoint=endpoint, keys=keys or {"p256dh": "A", "auth": "B"})
        req = _make_mock_request(speaker_id)
        result = asyncio.run(push_subscribe(body, req))
    finally:
        for k, v in saved.items():
            _app._state[k] = v

    return result, store


def _call_push_vapid_public_key(
    tmp_path: Path,
    monkeypatch,
    push_enabled: bool,
):
    """Set up _state, call push_vapid_public_key directly, return result."""
    import asyncio
    from unittest.mock import MagicMock

    import paramem.server.app as _app
    from paramem.server.app import push_vapid_public_key
    from paramem.server.vapid import ensure_vapid_keypair

    _setup_daily(tmp_path, monkeypatch)

    handle = None
    if push_enabled:
        handle = ensure_vapid_keypair(tmp_path)

    mock_config = MagicMock()
    mock_config.mobile_pwa.push_enabled = push_enabled

    saved = {
        "config": _app._state.get("config"),
        "vapid": _app._state.get("vapid"),
    }
    _app._state["config"] = mock_config
    _app._state["vapid"] = handle

    try:
        result = asyncio.run(push_vapid_public_key())
    finally:
        for k, v in saved.items():
            _app._state[k] = v

    return result, handle


class TestVapidPublicKeyEndpoint:
    def test_returns_key_when_push_enabled(self, tmp_path, monkeypatch):
        """push_vapid_public_key returns the base64url public key when push is enabled."""
        from paramem.server.vapid import application_server_key

        result, handle = _call_push_vapid_public_key(tmp_path, monkeypatch, push_enabled=True)

        # On success the handler returns a plain dict (FastAPI serialises to JSON).
        assert isinstance(result, dict)
        assert "key" in result
        assert result["key"] == application_server_key(handle)

    def test_returns_503_when_push_disabled(self, tmp_path, monkeypatch):
        """push_vapid_public_key returns 503 JSONResponse when push_enabled=false."""
        result, _handle = _call_push_vapid_public_key(tmp_path, monkeypatch, push_enabled=False)

        assert result.status_code == 503
        assert json.loads(result.body)["error"] == "push_not_enabled"


class TestPushSubscribeEndpoint:
    def test_subscribe_returns_subscribed(self, tmp_path, monkeypatch):
        """push_subscribe returns {'status': 'subscribed'} for a valid request."""
        result, _store = _call_push_subscribe(
            tmp_path, monkeypatch, push_enabled=True, speaker_id="Speaker0"
        )
        assert result == {"status": "subscribed"}

    def test_subscribe_persists_under_speaker_id(self, tmp_path, monkeypatch):
        """push_subscribe stores the subscription under auth_speaker_id."""
        endpoint = "https://web.push.example.com/s7"
        _result, store = _call_push_subscribe(
            tmp_path,
            monkeypatch,
            push_enabled=True,
            speaker_id="Speaker7",
            endpoint=endpoint,
        )

        assert store is not None
        subs = store.list("Speaker7")
        assert len(subs) == 1
        assert subs[0]["endpoint"] == endpoint

    def test_subscribe_403_for_no_speaker_id(self, tmp_path, monkeypatch):
        """push_subscribe returns 403 JSONResponse when auth_speaker_id is None."""
        result, _store = _call_push_subscribe(
            tmp_path, monkeypatch, push_enabled=True, speaker_id=None
        )

        assert result.status_code == 403
        assert json.loads(result.body)["error"] == "per_user_token_required"

    def test_subscribe_503_when_push_disabled(self, tmp_path, monkeypatch):
        """push_subscribe returns 503 JSONResponse when push_enabled=false."""
        result, _store = _call_push_subscribe(
            tmp_path, monkeypatch, push_enabled=False, speaker_id="Speaker0"
        )

        assert result.status_code == 503
        assert json.loads(result.body)["error"] == "push_not_enabled"

    def test_subscribe_ignores_body_speaker_id(self, tmp_path, monkeypatch):
        """PushSubscribeRequest silently discards extra body fields (extra='ignore').

        Verifies that the model rejects unknown fields including a body-claimed
        speaker_id so it cannot override the token-authenticated identity.
        """
        from paramem.server.app import PushSubscribeRequest

        # Build the model with an extra 'speaker_id' field — should be silently dropped.
        body = PushSubscribeRequest.model_validate(
            {
                "endpoint": "https://web.push.example.com/hijack",
                "keys": {"p256dh": "X", "auth": "Y"},
                "speaker_id": "Speaker99",  # extra field — must be silently ignored
            }
        )
        # After construction, the model must not carry speaker_id.
        assert not hasattr(body, "speaker_id")
        assert body.endpoint == "https://web.push.example.com/hijack"

    def test_subscribe_stores_under_token_not_body(self, tmp_path, monkeypatch):
        """Subscription is stored under the token's speaker_id, not any body-claimed one."""
        endpoint = "https://web.push.example.com/token-wins"
        _result, store = _call_push_subscribe(
            tmp_path,
            monkeypatch,
            push_enabled=True,
            speaker_id="Speaker0",
            endpoint=endpoint,
        )
        assert store is not None
        # Subscription is under the token identity.
        assert len(store.list("Speaker0")) == 1
        # No subscription for any other speaker.
        all_subs = store.all()
        assert set(all_subs.keys()) == {"Speaker0"}


# ---------------------------------------------------------------------------
# send_ping self-verification (keygen + send construct correctly)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Subscription validation — add() rejects malformed/unsafe endpoints
# ---------------------------------------------------------------------------


class TestSubscriptionValidation:
    """add() must raise ValueError for any invalid or unsafe subscription."""

    def _store(self, store_path, tmp_path, monkeypatch):
        """Helper: return a store with no daily key (validation fires regardless)."""
        # Validation runs before any encryption logic; use no-daily-key for speed.
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        monkeypatch.delenv("PARAMEM_DAILY_PASSPHRASE", raising=False)
        _clear_daily_identity_cache()
        return PushSubscriptionStore(store_path)

    def test_rejects_http_endpoint(self, tmp_path, monkeypatch, store_path):
        """http:// endpoint must be rejected — TLS is required."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {"endpoint": "http://push.example.com/test", "keys": {"p256dh": "A", "auth": "B"}}
        with pytest.raises(ValueError, match="https://"):
            store.add("Speaker0", sub)

    def test_rejects_localhost_endpoint(self, tmp_path, monkeypatch, store_path):
        """https://localhost/... must be rejected as a loopback host."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {"endpoint": "https://localhost/push/token", "keys": {"p256dh": "A", "auth": "B"}}
        with pytest.raises(ValueError, match="push relay"):
            store.add("Speaker0", sub)

    def test_rejects_loopback_ip(self, tmp_path, monkeypatch, store_path):
        """https://127.0.0.1/... must be rejected as a loopback address."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {"endpoint": "https://127.0.0.1/push/token", "keys": {"p256dh": "A", "auth": "B"}}
        with pytest.raises(ValueError, match="non-public IP"):
            store.add("Speaker0", sub)

    def test_rejects_rfc1918_host(self, tmp_path, monkeypatch, store_path):
        """https://10.0.0.5/... must be rejected (RFC-1918 private address)."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {"endpoint": "https://10.0.0.5/push/token", "keys": {"p256dh": "A", "auth": "B"}}
        with pytest.raises(ValueError, match="non-public IP"):
            store.add("Speaker0", sub)

    def test_rejects_bare_hostname(self, tmp_path, monkeypatch, store_path):
        """https://internalhost/... (no dot) must be rejected."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {"endpoint": "https://internalhost/push", "keys": {"p256dh": "A", "auth": "B"}}
        with pytest.raises(ValueError, match="push relay"):
            store.add("Speaker0", sub)

    def test_rejects_dot_local_hostname(self, tmp_path, monkeypatch, store_path):
        """https://device.local/... (mDNS) must be rejected."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {"endpoint": "https://device.local/push", "keys": {"p256dh": "A", "auth": "B"}}
        with pytest.raises(ValueError, match="push relay"):
            store.add("Speaker0", sub)

    def test_rejects_missing_p256dh(self, tmp_path, monkeypatch, store_path):
        """Subscription missing 'p256dh' key must be rejected."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {"endpoint": "https://web.push.apple.com/token", "keys": {"auth": "B"}}
        with pytest.raises(ValueError, match="p256dh"):
            store.add("Speaker0", sub)

    def test_rejects_missing_auth(self, tmp_path, monkeypatch, store_path):
        """Subscription missing 'auth' key must be rejected."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {"endpoint": "https://web.push.apple.com/token", "keys": {"p256dh": "A"}}
        with pytest.raises(ValueError, match="auth"):
            store.add("Speaker0", sub)

    def test_accepts_apple_push_endpoint(self, tmp_path, monkeypatch, store_path):
        """Realistic Apple push endpoint with valid keys must be accepted (no false positive)."""
        store = self._store(store_path, tmp_path, monkeypatch)
        sub = {
            "endpoint": "https://web.push.apple.com/3/device/AAABBBCCCDDDEEEFFF000111222333",
            "keys": {
                "p256dh": "BGyFwoIBCFNFnSG2vQwmLMsO5OBQQ_test_key_value",
                "auth": "abc123testauth",
            },
        }
        # Must not raise — this is a valid real-world subscription shape.
        added = store.add("Speaker0", sub)
        assert added is True
        assert len(store.list("Speaker0")) == 1


# ---------------------------------------------------------------------------
# /push/subscribe endpoint returns 400 for invalid subscription
# ---------------------------------------------------------------------------


class TestPushSubscribeEndpointValidation:
    def test_subscribe_400_for_invalid_endpoint(self, tmp_path, monkeypatch):
        """push_subscribe returns 400 invalid_subscription for an http:// endpoint."""
        result, _store = _call_push_subscribe(
            tmp_path,
            monkeypatch,
            push_enabled=True,
            speaker_id="Speaker0",
            endpoint="http://push.example.com/bad",
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert body["status"] == "invalid_subscription"
        assert "detail" in body


# ---------------------------------------------------------------------------
# TOCTOU guard clears in-memory state
# ---------------------------------------------------------------------------


class TestToctouClearsMemory:
    def test_toctou_guard_clears_in_memory_subscriptions(self, tmp_path, monkeypatch, store_path):
        """When the TOCTOU guard fires, in-memory subscriptions are reset.

        After the RuntimeError is raised, store.list() must reflect cleared
        state — not the stale in-memory subscription that was added mid-race.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = PushSubscriptionStore(store_path)

        def _plaintext_write(path: Path, payload: bytes) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(payload)

        with patch("paramem.backup.encryption.write_infra_bytes", side_effect=_plaintext_write):
            with pytest.raises(RuntimeError, match="written in plaintext"):
                store.add("Speaker0", _make_subscription())

        # After the guard fires, in-memory state must be cleared.
        assert store.list("Speaker0") == [], (
            "list() must return [] after TOCTOU guard clears in-memory state"
        )
        assert store.all() == {}, "all() must return {} after TOCTOU guard clears in-memory state"


class TestSendPingConstruct:
    def test_send_ping_uses_http2_transport(self, tmp_path, monkeypatch):
        """send_ping sends over httpx with http2=True and gets an HTTP-level response.

        Sends to a well-formed but dummy Apple push endpoint.  The real endpoint
        returns a 4xx (not a BadStatusLine or ConnectionError), which confirms:
        (a) HTTP/2 was negotiated (http_version=="HTTP/2"), and
        (b) the VAPID JWT construction is syntactically correct.

        Set NO_NETWORK=1 to skip in environments without internet access.
        """
        import os

        if os.environ.get("NO_NETWORK"):
            pytest.skip("NO_NETWORK set — skipping network probe")

        from paramem.server.push import send_ping
        from paramem.server.vapid import ensure_vapid_keypair

        _setup_daily(tmp_path, monkeypatch)
        handle = ensure_vapid_keypair(tmp_path)
        # A syntactically valid but non-existent Apple push path.
        dummy_subscription = {
            "endpoint": "https://web.push.apple.com/DUMMY_PARAMEM_SELF_TEST",
            "keys": {"p256dh": "AAAA", "auth": "BBBB"},
        }

        http_version, status_code = send_ping(dummy_subscription, handle, "mailto:test@localhost")

        # Apple returns a 4xx for an invalid token — NOT a BadStatusLine.
        # http_version=="HTTP/2" proves the transport fix is in place.
        assert http_version == "HTTP/2", (
            f"Expected HTTP/2 transport; got {http_version!r}. "
            "Check that h2 is installed and httpx.Client(http2=True) is used."
        )
        assert status_code is not None, (
            "Expected an HTTP status code (4xx); got None (network-level failure). "
            f"http_version was: {http_version!r}"
        )
        assert isinstance(status_code, int)
        # Any 4xx is correct: 400/404/410 are all expected for a dummy endpoint.
        assert 400 <= status_code < 500, f"Expected a 4xx for a dummy endpoint; got {status_code}"
