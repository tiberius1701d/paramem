"""Tests for extended `paramem.server.auth.BearerTokenMiddleware`.

Covers:
- Per-user token auth (header + cookie)
- Header takes precedence over cookie
- speaker_id attached to request scope on per-user match
- Legacy shared token still authorizes (no speaker_id)
- Exempt paths bypass auth
- Revoked tokens rejected
- user_token_getter=None (pre-lifespan) does not crash
- log_startup_posture four states

CPU-only — no model load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.server.auth import BearerTokenMiddleware, log_startup_posture
from paramem.server.user_tokens import UserTokenStore

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point the env + module default at it."""
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


def _make_store(tmp_path: Path) -> UserTokenStore:
    return UserTokenStore(tmp_path / "user_tokens.json")


def _make_app(
    shared_token: str = "",
    user_token_getter=None,
    cookie_name: str = "paramem_token",
    exempt_paths=("/", "/manifest.json"),
    exempt_prefixes=(),
) -> FastAPI:
    """Build a minimal FastAPI app with BearerTokenMiddleware."""
    app = FastAPI()
    app.add_middleware(
        BearerTokenMiddleware,
        token=shared_token,
        user_token_getter=user_token_getter,
        cookie_name=cookie_name,
        exempt_paths=exempt_paths,
        exempt_prefixes=exempt_prefixes,
    )

    @app.get("/ping")
    def ping(request: Request) -> dict:
        sid = getattr(request.state, "speaker_id", None)
        return {"ok": True, "speaker_id": sid}

    @app.get("/")
    def root() -> dict:
        return {"root": True}

    @app.get("/manifest.json")
    def manifest() -> dict:
        return {"name": "ParaMem"}

    return app


# ---------------------------------------------------------------------------
# Per-user token via Authorization header
# ---------------------------------------------------------------------------


class TestPerUserTokenHeader:
    def test_valid_user_token_header_returns_200(self, tmp_path, monkeypatch):
        """A valid per-user token in Authorization header → 200."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Test")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_valid_user_token_sets_speaker_id(self, tmp_path, monkeypatch):
        """A valid per-user token attaches speaker_id to request.state."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Test")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "Speaker0"

    def test_invalid_token_returns_401(self, tmp_path, monkeypatch):
        """An invalid bearer token is rejected with 401."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Test")  # populate store so auth is enabled

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401
        assert resp.json()["error"] == "unauthorized"

    def test_revoked_token_returns_401(self, tmp_path, monkeypatch):
        """A revoked token is rejected with 401.

        A second active token is kept alive so has_active_tokens() stays True
        (useful for posture logging).  Auth stays ON regardless because the store
        is wired (fail-closed enablement).  This test verifies per-token
        revocation while auth stays ON.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Test")
        # Keep a second active token (verifies per-token revocation, not full drain).
        store.mint("Speaker0", "Device B")
        store.revoke_token(token)

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_fail_closed_when_last_token_revoked(self, tmp_path, monkeypatch):
        """Revoking the last token keeps auth ON (fail-closed), not OFF.

        A wired store with zero active tokens must reject every request — it
        must NOT silently fall open.  The PWA shell paths (/ and /manifest.json)
        remain reachable via exemptions.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Device")
        store.revoke_token(token)  # last (and only) token revoked

        assert not store.has_active_tokens()  # store is empty

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        # Protected path: must be 401, not 200 (store is wired → fail-closed).
        resp = client.get("/ping")
        assert resp.status_code == 401

        # Exempt paths still reachable.
        assert client.get("/").status_code == 200
        assert client.get("/manifest.json").status_code == 200

    def test_no_token_returns_401_when_active_tokens_exist(self, tmp_path, monkeypatch):
        """No token presented when store has active tokens → 401."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Test")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Cookie-carried token
# ---------------------------------------------------------------------------


class TestCookieToken:
    def test_cookie_token_authorizes(self, tmp_path, monkeypatch):
        """A valid token in the cookie is accepted."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Browser")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app, cookies={"paramem_token": token})

        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "Speaker0"

    def test_header_takes_precedence_over_cookie(self, tmp_path, monkeypatch):
        """When both header and cookie are present, the header token is used."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        good_token = store.mint("Speaker0", "Header device")
        bad_cookie = "not-a-valid-token"

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app, cookies={"paramem_token": bad_cookie})

        # Header carries the good token; cookie carries an invalid value.
        resp = client.get("/ping", headers={"Authorization": f"Bearer {good_token}"})
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "Speaker0"

    def test_invalid_cookie_returns_401(self, tmp_path, monkeypatch):
        """An invalid cookie token is rejected."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app, cookies={"paramem_token": "bad-cookie-token"})

        resp = client.get("/ping")
        assert resp.status_code == 401

    def test_whitespace_only_cookie_is_unauthenticated(self, tmp_path, monkeypatch):
        """A cookie whose value is all whitespace is treated as absent (fail-closed).

        An all-spaces morsel.value must not reach the token comparison — it must
        be normalised to None so the request returns 401 rather than matching
        against any stored token.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app, cookies={"paramem_token": "   "})

        resp = client.get("/ping")
        assert resp.status_code == 401, (
            "Whitespace-only cookie value must be treated as absent (401), not as a token"
        )


# ---------------------------------------------------------------------------
# Legacy shared token still works
# ---------------------------------------------------------------------------


class TestLegacySharedToken:
    def test_shared_token_still_authorizes(self):
        """The legacy shared token path continues to work after the extension."""
        app = _make_app(shared_token="legacy-secret")
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": "Bearer legacy-secret"})
        assert resp.status_code == 200

    def test_shared_token_does_not_set_speaker_id(self):
        """Shared token auth does not attach a speaker_id (unattributed)."""
        app = _make_app(shared_token="legacy-secret")
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": "Bearer legacy-secret"})
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] is None

    def test_shared_token_wrong_returns_401(self):
        """Wrong shared token returns 401."""
        app = _make_app(shared_token="correct")
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Exempt paths
# ---------------------------------------------------------------------------


class TestExemptPaths:
    def test_root_reachable_without_token(self, tmp_path, monkeypatch):
        """/ is exempt — reachable even when auth is ON."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")  # enable auth

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/")
        assert resp.status_code == 200

    def test_manifest_reachable_without_token(self, tmp_path, monkeypatch):
        """/manifest.json is exempt — reachable even when auth is ON."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/manifest.json")
        assert resp.status_code == 200

    def test_protected_path_requires_token(self, tmp_path, monkeypatch):
        """/ping is not exempt — requires a valid token when auth is ON."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# user_token_getter=None (pre-lifespan / not wired)
# ---------------------------------------------------------------------------


class TestUserTokenGetterNone:
    def test_getter_none_off_mode_does_not_crash(self):
        """user_token_getter=None with no shared token → OFF mode, no crash."""
        app = _make_app(shared_token="", user_token_getter=None)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 200  # OFF mode passes through

    def test_getter_none_shared_token_still_works(self):
        """user_token_getter=None with shared token → shared path still enforced."""
        app = _make_app(shared_token="shared-tok", user_token_getter=None)
        client = TestClient(app)

        resp_bad = client.get("/ping", headers={"Authorization": "Bearer wrong"})
        assert resp_bad.status_code == 401

        resp_ok = client.get("/ping", headers={"Authorization": "Bearer shared-tok"})
        assert resp_ok.status_code == 200

    def test_getter_returning_none_off_mode(self):
        """A getter that returns None makes the store absent → OFF if no shared token."""
        app = _make_app(shared_token="", user_token_getter=lambda: None)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 200  # OFF: no shared token, getter returns None


# ---------------------------------------------------------------------------
# Exempt prefix matching
# ---------------------------------------------------------------------------


class TestExemptPrefixes:
    def test_exempt_prefix_path_reachable_without_token(self, tmp_path, monkeypatch):
        """A path under an exempt prefix passes through without a token."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")  # enable auth

        # Add /app/ route and exempt it via prefix.
        app = FastAPI()
        app.add_middleware(
            BearerTokenMiddleware,
            token="",
            user_token_getter=lambda: store,
            exempt_paths=("/", "/manifest.json"),
            exempt_prefixes=("/app/",),
        )

        @app.get("/app/dashboard")
        def dashboard():
            return {"page": "dashboard"}

        @app.get("/chat")
        def chat():
            return {"chat": True}

        client = TestClient(app)

        # /app/dashboard is exempt (prefix /app/).
        assert client.get("/app/dashboard").status_code == 200

        # /chat is NOT exempt — requires token.
        assert client.get("/chat").status_code == 401

    def test_exempt_prefix_does_not_match_different_path(self, tmp_path, monkeypatch):
        """An exempt prefix ending with '/' does not match a path that merely
        starts with the same letters but is a different route.

        E.g. exempt_prefixes=('/app/',) must not match '/application/secret'.
        This is a documentation contract test — startswith behavior.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")

        app = FastAPI()
        app.add_middleware(
            BearerTokenMiddleware,
            token="",
            user_token_getter=lambda: store,
            exempt_prefixes=("/app/",),
        )

        @app.get("/application/secret")
        def secret():
            return {"secret": True}

        client = TestClient(app)

        # /application/secret does NOT start with /app/ — must be 401.
        assert client.get("/application/secret").status_code == 401


# ---------------------------------------------------------------------------
# ON-both mode (shared token + per-user store wired)
# ---------------------------------------------------------------------------


class TestOnBothMode:
    def test_per_user_token_authorized_in_on_both_mode(self, tmp_path, monkeypatch):
        """Shared token + per-user store: a valid per-user token authorizes and
        sets speaker_id.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        user_token = store.mint("Speaker0", "Device")

        app = _make_app(shared_token="shared-secret", user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {user_token}"})
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "Speaker0"

    def test_shared_token_authorized_in_on_both_mode(self, tmp_path, monkeypatch):
        """Shared token + per-user store: the shared token also authorizes (no
        speaker_id set).
        """
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")  # populate store

        app = _make_app(shared_token="shared-secret", user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": "Bearer shared-secret"})
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] is None  # shared token → no speaker


# ---------------------------------------------------------------------------
# Fail-clean: unauthorized requests must not reach the downstream app
# ---------------------------------------------------------------------------


class TestFailClean:
    def test_unauthorized_request_does_not_invoke_handler(self, tmp_path, monkeypatch):
        """A 401 response must not invoke the route handler downstream.

        Verifies no stale speaker_id leaks: the handler is never reached when
        auth fails, so scope["state"] is never mutated by the application.
        """
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("Speaker0", "Device")

        handler_called = []

        app = FastAPI()
        app.add_middleware(
            BearerTokenMiddleware,
            token="",
            user_token_getter=lambda: store,
        )

        @app.get("/ping")
        def ping():
            handler_called.append(True)
            return {"ok": True}

        client = TestClient(app)

        resp = client.get("/ping")  # no token → 401
        assert resp.status_code == 401
        assert handler_called == [], "handler must not be invoked on 401"


# ---------------------------------------------------------------------------
# log_startup_posture four states
# ---------------------------------------------------------------------------


class TestLogStartupPosture:
    """Tests for log_startup_posture's four output states.

    The project pattern for caplog (see tests/test_ingest_registry.py) is to
    attach caplog.handler directly to the named logger, because some installed
    packages set propagate=False which breaks the default caplog propagation path.
    """

    def _capture(self, caplog, level, fn, *args, **kwargs):
        """Attach caplog.handler to the auth logger, call fn, detach."""
        import logging

        named = logging.getLogger("paramem.server.auth")
        named.addHandler(caplog.handler)
        named.setLevel(level)
        try:
            fn(*args, **kwargs)
        finally:
            named.removeHandler(caplog.handler)

    def test_off_state(self, caplog):
        """No shared token, 0 user tokens → AUTH: OFF warning."""
        import logging

        self._capture(caplog, logging.WARNING, log_startup_posture, "", n_user_tokens=0)
        assert "AUTH: OFF" in caplog.text

    def test_on_shared_state(self, caplog):
        """Shared token set, 0 user tokens → AUTH: ON-shared info."""
        import logging

        self._capture(caplog, logging.INFO, log_startup_posture, "tok", n_user_tokens=0)
        assert "AUTH: ON-shared" in caplog.text

    def test_on_per_user_state(self, caplog):
        """Store wired + active tokens → AUTH: ON-per-user info."""
        import logging

        self._capture(
            caplog,
            logging.INFO,
            log_startup_posture,
            "",
            n_user_tokens=3,
            per_user_active=True,
        )
        assert "AUTH: ON-per-user" in caplog.text
        assert "3" in caplog.text

    def test_on_both_state(self, caplog):
        """Shared token + store wired → AUTH: ON-both info."""
        import logging

        self._capture(
            caplog,
            logging.INFO,
            log_startup_posture,
            "tok",
            n_user_tokens=2,
            per_user_active=True,
        )
        assert "AUTH: ON-both" in caplog.text

    def test_backward_compatible_no_n_user_tokens(self, caplog):
        """Calling log_startup_posture with only token (legacy signature) does not crash."""
        import logging

        self._capture(caplog, logging.INFO, log_startup_posture, "tok")
        assert "AUTH: ON-shared" in caplog.text

    def test_wired_empty_store_logs_on_per_user_not_off(self, caplog):
        """Wired store with 0 active tokens logs ON-per-user (fail-closed), not OFF.

        This is the bug the fix addresses: the middleware rejects every request
        (fail-closed) when the store is wired but empty, but the old log printed
        AUTH: OFF, contradicting runtime behavior.
        """
        import logging

        self._capture(
            caplog,
            logging.INFO,
            log_startup_posture,
            "",
            n_user_tokens=0,
            per_user_active=True,
        )
        assert "AUTH: ON-per-user" in caplog.text
        assert "AUTH: OFF" not in caplog.text
        assert "fail-closed" in caplog.text

    def test_off_state_no_store_no_shared_token(self, caplog):
        """No shared token and no store wired → AUTH: OFF (store absent = truly open)."""
        import logging

        self._capture(
            caplog,
            logging.WARNING,
            log_startup_posture,
            "",
            n_user_tokens=0,
            per_user_active=False,
        )
        assert "AUTH: OFF" in caplog.text


# ---------------------------------------------------------------------------
# _build_user_token_store helper — regression guard for Fix 1
# ---------------------------------------------------------------------------


class TestBuildUserTokenStore:
    """Unit tests for ``_build_user_token_store``.

    Verifies that per-user auth is gated on ``mobile_pwa.enabled``: a default
    deployment (enabled=False) must receive ``None`` so the middleware stays in
    OFF mode; an opted-in deployment (enabled=True) must receive a live store.

    Config loaded from the fixture; mobile_pwa.enabled toggled in code per the
    test-config-loader convention (no fixture edits).
    """

    def test_returns_none_when_mobile_pwa_disabled(self, tmp_path):
        """mobile_pwa.enabled=False → returns None (middleware stays OFF)."""
        from paramem.server.app import _build_user_token_store
        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        # Fixture pins enabled=false; assert precondition then verify helper.
        assert cfg.mobile_pwa.enabled is False
        result = _build_user_token_store(cfg)
        assert result is None

    def test_returns_store_when_mobile_pwa_enabled(self, tmp_path):
        """mobile_pwa.enabled=True → returns a UserTokenStore instance."""
        from paramem.server.app import _build_user_token_store
        from paramem.server.config import load_server_config
        from paramem.server.user_tokens import UserTokenStore

        cfg = load_server_config("tests/fixtures/server.yaml")
        cfg.mobile_pwa.enabled = True
        # Point the store path at tmp_path so no disk side-effects.
        cfg.paths.data = tmp_path

        result = _build_user_token_store(cfg)
        assert isinstance(result, UserTokenStore)


# ---------------------------------------------------------------------------
# cookie_name_getter path — Fix 4
# ---------------------------------------------------------------------------


class TestCookieNameGetter:
    """Tests for the ``cookie_name_getter`` parameter.

    Verifies that a live callable drives the effective cookie name at request
    time, and that ``None`` from the getter falls back to the static default.
    """

    def _make_app_with_getter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        cookie_name_getter,
    ) -> tuple[FastAPI, str]:
        """Build a minimal app with a per-user store and a custom getter."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Browser")

        app = FastAPI()
        app.add_middleware(
            BearerTokenMiddleware,
            token="",
            user_token_getter=lambda: store,
            cookie_name="paramem_token",
            cookie_name_getter=cookie_name_getter,
        )

        @app.get("/ping")
        def ping(request: Request) -> dict:
            sid = getattr(request.state, "speaker_id", None)
            return {"ok": True, "speaker_id": sid}

        return app, token

    def test_custom_cookie_name_getter_authorizes(self, tmp_path, monkeypatch):
        """cookie_name_getter returning 'custom_cookie' → that cookie authorizes."""
        app, token = self._make_app_with_getter(
            tmp_path, monkeypatch, cookie_name_getter=lambda: "custom_cookie"
        )
        client = TestClient(app, cookies={"custom_cookie": token})
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "Speaker0"

    def test_default_cookie_name_ignored_when_getter_returns_custom(self, tmp_path, monkeypatch):
        """When getter returns 'custom_cookie', the default 'paramem_token' cookie
        does NOT authorize (wrong name).
        """
        app, token = self._make_app_with_getter(
            tmp_path, monkeypatch, cookie_name_getter=lambda: "custom_cookie"
        )
        client = TestClient(app, cookies={"paramem_token": token})
        resp = client.get("/ping")
        # Token sent under wrong cookie name → 401.
        assert resp.status_code == 401

    def test_getter_returning_none_falls_back_to_static_default(self, tmp_path, monkeypatch):
        """cookie_name_getter returning None falls back to the static cookie_name."""
        app, token = self._make_app_with_getter(
            tmp_path, monkeypatch, cookie_name_getter=lambda: None
        )
        # Token sent under the static default 'paramem_token' → authorized.
        client = TestClient(app, cookies={"paramem_token": token})
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "Speaker0"


# ---------------------------------------------------------------------------
# Scope surfaced on request.state
# ---------------------------------------------------------------------------


def _make_scope_app(
    shared_token: str = "",
    user_token_getter=None,
) -> FastAPI:
    """Minimal app that exposes request.state.scope in the /ping response."""
    app = FastAPI()
    app.add_middleware(
        BearerTokenMiddleware,
        token=shared_token,
        user_token_getter=user_token_getter,
    )

    @app.get("/ping")
    def ping(request: Request) -> dict:
        sid = getattr(request.state, "speaker_id", None)
        scope = getattr(request.state, "scope", None)
        return {"ok": True, "speaker_id": sid, "scope": scope}

    return app


class TestScopeOnRequestState:
    def test_per_user_admin_token_sets_admin_scope(self, tmp_path, monkeypatch):
        """A per-user admin token sets request.state.scope = 'admin'."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Admin", scope="admin")

        app = _make_scope_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["scope"] == "admin"
        assert resp.json()["speaker_id"] == "Speaker0"

    def test_per_user_chat_token_sets_chat_scope(self, tmp_path, monkeypatch):
        """A per-user chat token sets request.state.scope = 'chat'."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Phone", scope="chat")

        app = _make_scope_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["scope"] == "chat"

    def test_shared_token_sets_admin_scope(self):
        """The shared env token sets request.state.scope = 'admin'."""
        app = _make_scope_app(shared_token="shared-secret")
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": "Bearer shared-secret"})
        assert resp.status_code == 200
        assert resp.json()["scope"] == "admin"
        # Shared token does not set speaker_id (unattributed).
        assert resp.json()["speaker_id"] is None

    def test_unattributed_chat_token_sets_chat_scope_no_speaker(self, tmp_path, monkeypatch):
        """An unattributed chat token → scope='chat', speaker_id absent on state."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint(None, "Shared Kitchen Tablet", scope="chat")

        app = _make_scope_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["scope"] == "chat"
        assert resp.json()["speaker_id"] is None

    def test_off_mode_stamps_admin_scope(self):
        """OFF mode (no token, no store) → handler runs, scope stamped 'admin'.

        OFF is the deliberate open posture: the middleware stamps
        admin scope on pass-through requests so the ``require_admin`` gate stays
        a pure ``scope == 'admin'`` check and every endpoint stays reachable
        without credentials.
        """
        app = _make_scope_app(shared_token="", user_token_getter=None)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 200  # OFF mode passes through
        assert resp.json()["scope"] == "admin"
