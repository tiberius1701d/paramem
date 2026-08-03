"""Tests for `paramem.server.auth` — bearer-token middleware.

Re-spec (shared-token retirement, C): the middleware's ``token=`` constructor
parameter and its constant-time shared-token comparison branch are deleted —
every credential now lives exclusively in a
:class:`~paramem.server.user_tokens.UserTokenStore`.  ``OFF``/``ON`` are
keyed on store presence only.  These tests build the middleware with a
``user_token_getter`` (backed by a real ``UserTokenStore`` on ``tmp_path``,
or ``None`` for OFF) instead of a literal shared-token string.

This file merges the former ``test_auth_middleware.py`` (extended coverage:
cookie carrier, exempt paths/prefixes, cookie-name getter, scope-on-state,
fail-clean, getter-None) into the one canonical suite.  Classes that tested
the retired shared-secret constructor param (``TestLegacySharedToken``,
``TestOnBothMode``) are replaced by :class:`TestRetiredSharedTokenSemantics`,
one discriminating pin for the retirement.  ``TestLogStartupPosture`` is
reconciled with the migration-guard coverage below it — the retired
ON-shared/ON-both states and the legacy single-positional-arg call form are
dropped, not re-specced, since neither exists in the current signature or
behavior.
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


def _make_app(user_token_getter=None) -> FastAPI:
    """Build a minimal FastAPI app with BearerTokenMiddleware (ping/echo only)."""
    app = FastAPI()
    app.add_middleware(BearerTokenMiddleware, user_token_getter=user_token_getter)

    @app.get("/ping")
    def ping() -> dict:
        return {"ok": True}

    @app.post("/echo")
    def echo(payload: dict) -> dict:
        return payload

    return app


def _make_full_app(
    user_token_getter=None,
    cookie_name: str = "paramem_token",
    cookie_name_getter=None,
    exempt_paths=("/", "/manifest.json"),
    exempt_prefixes=(),
) -> FastAPI:
    """Build a FastAPI app with BearerTokenMiddleware exposing request.state.

    ``/ping`` surfaces ``speaker_id`` and ``scope`` from ``request.state`` so
    tests can assert what the middleware stamped on the ASGI scope.  ``/`` and
    ``/manifest.json`` back the default exempt-path tests.
    """
    app = FastAPI()
    app.add_middleware(
        BearerTokenMiddleware,
        user_token_getter=user_token_getter,
        cookie_name=cookie_name,
        cookie_name_getter=cookie_name_getter,
        exempt_paths=exempt_paths,
        exempt_prefixes=exempt_prefixes,
    )

    @app.get("/ping")
    def ping(request: Request) -> dict:
        sid = getattr(request.state, "speaker_id", None)
        scope = getattr(request.state, "scope", None)
        return {"ok": True, "speaker_id": sid, "scope": scope}

    @app.get("/")
    def root() -> dict:
        return {"root": True}

    @app.get("/manifest.json")
    def manifest() -> dict:
        return {"name": "ParaMem"}

    return app


class TestBearerTokenMiddleware:
    def test_disabled_when_no_store_wired(self) -> None:
        client = TestClient(_make_app(user_token_getter=None))
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_disabled_when_getter_returns_none(self) -> None:
        client = TestClient(_make_app(user_token_getter=lambda: None))
        response = client.post("/echo", json={"x": 1})
        assert response.status_code == 200

    def test_rejects_without_header(self, tmp_path) -> None:
        store = UserTokenStore(tmp_path / "user_tokens.json")
        store.mint("speaker0", "Device", scope="chat")
        client = TestClient(_make_app(user_token_getter=lambda: store))
        response = client.get("/ping")
        assert response.status_code == 401
        body = response.json()
        assert body["error"] == "unauthorized"
        assert "missing" in body["detail"].lower()
        assert response.headers.get("WWW-Authenticate", "").startswith("Bearer")

    def test_rejects_malformed_header(self, tmp_path) -> None:
        store = UserTokenStore(tmp_path / "user_tokens.json")
        store.mint("speaker0", "Device", scope="chat")
        client = TestClient(_make_app(user_token_getter=lambda: store))
        response = client.get("/ping", headers={"Authorization": "Token secret-abc"})
        assert response.status_code == 401
        assert "malformed" in response.json()["detail"].lower()

    def test_rejects_wrong_token(self, tmp_path) -> None:
        store = UserTokenStore(tmp_path / "user_tokens.json")
        store.mint("speaker0", "Device", scope="chat")
        client = TestClient(_make_app(user_token_getter=lambda: store))
        response = client.get("/ping", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    def test_accepts_correct_token(self, tmp_path) -> None:
        store = UserTokenStore(tmp_path / "user_tokens.json")
        token = store.mint("speaker0", "Device", scope="chat")
        client = TestClient(_make_app(user_token_getter=lambda: store))
        response = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_accepts_correct_token_with_trailing_space(self, tmp_path) -> None:
        # The middleware strips trailing whitespace on the presented token.
        store = UserTokenStore(tmp_path / "user_tokens.json")
        token = store.mint("speaker0", "Device", scope="chat")
        client = TestClient(_make_app(user_token_getter=lambda: store))
        response = client.get("/ping", headers={"Authorization": f"Bearer {token}   "})
        assert response.status_code == 200

    def test_wired_but_empty_store_is_fail_closed(self, tmp_path) -> None:
        """A wired store with zero active tokens 401s every request — it
        does not fall back to OFF behavior."""
        store = UserTokenStore(tmp_path / "user_tokens.json")
        client = TestClient(_make_app(user_token_getter=lambda: store))
        response = client.get("/ping")
        assert response.status_code == 401

    def test_revoked_token_rejected(self, tmp_path) -> None:
        store = UserTokenStore(tmp_path / "user_tokens.json")
        token = store.mint("speaker0", "Device", scope="chat")
        store.revoke_token(token)
        client = TestClient(_make_app(user_token_getter=lambda: store))
        response = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Per-user token via Authorization header
# ---------------------------------------------------------------------------


class TestPerUserTokenHeader:
    def test_valid_user_token_header_returns_200(self, tmp_path, monkeypatch):
        """A valid per-user token in Authorization header → 200."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("speaker0", "Test")

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_valid_user_token_sets_speaker_id(self, tmp_path, monkeypatch):
        """A valid per-user token attaches speaker_id to request.state."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Test")

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "speaker0"

    def test_invalid_token_returns_401(self, tmp_path, monkeypatch):
        """An invalid bearer token is rejected with 401."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("speaker0", "Test")  # populate store so auth is enabled

        app = _make_full_app(user_token_getter=lambda: store)
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
        token = store.mint("speaker0", "Test")
        # Keep a second active token (verifies per-token revocation, not full drain).
        store.mint("speaker0", "Device B")
        store.revoke_token(token)

        app = _make_full_app(user_token_getter=lambda: store)
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
        token = store.mint("speaker0", "Device")
        store.revoke_token(token)  # last (and only) token revoked

        assert not store.has_active_tokens()  # store is empty

        app = _make_full_app(user_token_getter=lambda: store)
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
        store.mint("speaker0", "Test")

        app = _make_full_app(user_token_getter=lambda: store)
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

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app, cookies={"paramem_token": token})

        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "speaker0"

    def test_header_takes_precedence_over_cookie(self, tmp_path, monkeypatch):
        """When both header and cookie are present, the header token is used."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        good_token = store.mint("Speaker0", "Header device")
        bad_cookie = "not-a-valid-token"

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app, cookies={"paramem_token": bad_cookie})

        # Header carries the good token; cookie carries an invalid value.
        resp = client.get("/ping", headers={"Authorization": f"Bearer {good_token}"})
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "speaker0"

    def test_invalid_cookie_returns_401(self, tmp_path, monkeypatch):
        """An invalid cookie token is rejected."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("speaker0", "Device")

        app = _make_full_app(user_token_getter=lambda: store)
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
        store.mint("speaker0", "Device")

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app, cookies={"paramem_token": "   "})

        resp = client.get("/ping")
        assert resp.status_code == 401, (
            "Whitespace-only cookie value must be treated as absent (401), not as a token"
        )


# ---------------------------------------------------------------------------
# Retired shared-token semantics (replaces TestLegacySharedToken/TestOnBothMode)
# ---------------------------------------------------------------------------


class TestRetiredSharedTokenSemantics:
    """Replaces the former ``TestLegacySharedToken`` and ``TestOnBothMode``
    classes, which exercised the deleted ``token=`` shared-secret constructor
    param and its OFF/ON-both matrix.  There is no shared-secret credential
    anymore — every accepted token is a ``UserTokenStore`` entry.  This is the
    one discriminating pin for the retirement: a value shaped like the old
    shared secret is just an unregistered string now.  It is rejected once a
    store is wired (fail-closed ON), and passes through untouched — stamped
    with the non-admin ``"chat"`` scope — only when no store is wired at all
    (auth OFF).
    """

    def test_unregistered_value_401s_when_wired_else_passes_chat_scope_off(
        self, tmp_path, monkeypatch
    ):
        """One old-shared-secret-shaped value, checked against both postures."""
        old_style_value = "legacy-secret"

        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("speaker0", "Device")  # any active token wires the store ON

        on_client = TestClient(_make_full_app(user_token_getter=lambda: store))
        on_resp = on_client.get("/ping", headers={"Authorization": f"Bearer {old_style_value}"})
        assert on_resp.status_code == 401

        off_client = TestClient(_make_full_app(user_token_getter=None))
        off_resp = off_client.get("/ping", headers={"Authorization": f"Bearer {old_style_value}"})
        assert off_resp.status_code == 200
        assert off_resp.json()["scope"] == "chat"


# ---------------------------------------------------------------------------
# Exempt paths
# ---------------------------------------------------------------------------


class TestExemptPaths:
    def test_root_reachable_without_token(self, tmp_path, monkeypatch):
        """/ is exempt — reachable even when auth is ON."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("speaker0", "Device")  # enable auth

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/")
        assert resp.status_code == 200

    def test_manifest_reachable_without_token(self, tmp_path, monkeypatch):
        """/manifest.json is exempt — reachable even when auth is ON."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("speaker0", "Device")

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/manifest.json")
        assert resp.status_code == 200

    def test_protected_path_requires_token(self, tmp_path, monkeypatch):
        """/ping is not exempt — requires a valid token when auth is ON."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("speaker0", "Device")

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# user_token_getter=None / getter returning None (pre-lifespan / not wired)
# ---------------------------------------------------------------------------


class TestUserTokenGetterNone:
    def test_getter_none_off_mode_does_not_crash(self):
        """user_token_getter=None → OFF mode, no crash."""
        app = _make_full_app(user_token_getter=None)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 200  # OFF mode passes through

    def test_getter_returning_none_off_mode(self):
        """A getter that returns None makes the store absent → OFF mode."""
        app = _make_full_app(user_token_getter=lambda: None)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 200  # OFF: getter returns None


# ---------------------------------------------------------------------------
# Exempt prefix matching
# ---------------------------------------------------------------------------


class TestExemptPrefixes:
    def test_exempt_prefix_path_reachable_without_token(self, tmp_path, monkeypatch):
        """A path under an exempt prefix passes through without a token."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        store.mint("speaker0", "Device")  # enable auth

        # Add /app/ route and exempt it via prefix.
        app = FastAPI()
        app.add_middleware(
            BearerTokenMiddleware,
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
        store.mint("speaker0", "Device")

        app = FastAPI()
        app.add_middleware(
            BearerTokenMiddleware,
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
        store.mint("speaker0", "Device")

        handler_called = []

        app = FastAPI()
        app.add_middleware(BearerTokenMiddleware, user_token_getter=lambda: store)

        @app.get("/ping")
        def ping():
            handler_called.append(True)
            return {"ok": True}

        client = TestClient(app)

        resp = client.get("/ping")  # no token → 401
        assert resp.status_code == 401
        assert handler_called == [], "handler must not be invoked on 401"


# ---------------------------------------------------------------------------
# log_startup_posture — two output states (OFF / ON), current signature
# ---------------------------------------------------------------------------


class TestLogStartupPosture:
    """Tests for ``log_startup_posture``'s two states (OFF / ON), keyed on
    store presence per the current ``(n_user_tokens, per_user_active)``
    signature.

    Re-spec: the retired shared-token model's three/four-way ON-shared /
    ON-per-user / ON-both matrix and its legacy single-positional-arg call
    form (``log_startup_posture("tok")``) are gone — dropped, not
    re-specced, since neither exists in the current signature or behavior.
    A plain OFF-state assertion is not repeated here — it is already covered
    by :class:`TestLogStartupPostureMigrationGuard` below.
    """

    def test_on_with_zero_tokens_is_fail_closed(self, caplog):
        """Store wired, 0 active tokens → AUTH: ON fail-closed info, not OFF.

        This is the bug the fix addresses: the middleware rejects every
        request (fail-closed) when the store is wired but empty; the log
        must say ON, not OFF, or the two would contradict runtime behavior.
        """
        import logging

        with caplog.at_level(logging.INFO, logger="paramem.server.auth"):
            log_startup_posture(n_user_tokens=0, per_user_active=True)

        assert "AUTH: ON" in caplog.text
        assert "fail-closed" in caplog.text
        assert "AUTH: OFF" not in caplog.text

    def test_on_with_active_tokens(self, caplog):
        """Store wired with active tokens → AUTH: ON info with the count."""
        import logging

        with caplog.at_level(logging.INFO, logger="paramem.server.auth"):
            log_startup_posture(n_user_tokens=3, per_user_active=True)

        assert "AUTH: ON" in caplog.text
        assert "3" in caplog.text


class TestLogStartupPostureMigrationGuard:
    """Fail-open migration guard: a stale ``PARAMEM_API_TOKEN`` from the
    retired shared-token model must not silently leave a deployment open
    with no warning.  ``log_startup_posture`` is this module's ONE read of
    the env var (see the module docstring) — presence-only, never used as a
    credential.
    """

    def test_off_with_stale_token_env_var_warns_loudly(self, monkeypatch, caplog) -> None:
        import logging

        from paramem.server.auth import log_startup_posture

        monkeypatch.setenv("PARAMEM_API_TOKEN", "old-shared-secret")
        with caplog.at_level(logging.WARNING, logger="paramem.server.auth"):
            log_startup_posture(n_user_tokens=0, per_user_active=False)

        messages = [r.message % r.args if r.args else r.message for r in caplog.records]
        assert any("AUTH: OFF" in m for m in messages)
        assert any("PARAMEM_API_TOKEN" in m and "no longer" in m.lower() for m in messages)

    def test_off_without_token_env_var_no_migration_warning(self, monkeypatch, caplog) -> None:
        import logging

        from paramem.server.auth import log_startup_posture

        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        with caplog.at_level(logging.WARNING, logger="paramem.server.auth"):
            log_startup_posture(n_user_tokens=0, per_user_active=False)

        messages = [r.message % r.args if r.args else r.message for r in caplog.records]
        assert any("AUTH: OFF" in m for m in messages)
        assert not any("PARAMEM_API_TOKEN" in m for m in messages)

    def test_on_with_stale_token_env_var_no_migration_warning(self, monkeypatch, caplog) -> None:
        """A wired store (ON, regardless of token count) is already
        protected — the migration guard only fires on the OFF branch."""
        import logging

        from paramem.server.auth import log_startup_posture

        monkeypatch.setenv("PARAMEM_API_TOKEN", "old-shared-secret")
        with caplog.at_level(logging.WARNING, logger="paramem.server.auth"):
            log_startup_posture(n_user_tokens=1, per_user_active=True)

        messages = [r.message % r.args if r.args else r.message for r in caplog.records]
        assert not any("PARAMEM_API_TOKEN" in m for m in messages)


# ---------------------------------------------------------------------------
# _build_user_token_store helper — per-user auth gated on mobile_pwa.enabled
# OR a prior mint's on-disk file
# ---------------------------------------------------------------------------


class TestBuildUserTokenStore:
    """Unit tests for ``paramem.server.app._build_user_token_store``.

    The store is wired when EITHER ``config.mobile_pwa.enabled`` is True OR
    the store's on-disk file already exists (a prior ``mint-user-token``
    run) — see its docstring. All three branches (either condition alone,
    neither) are covered here.
    """

    def _config(self, tmp_path, *, mobile_pwa_enabled: bool):
        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        cfg.mobile_pwa.enabled = mobile_pwa_enabled
        cfg.paths.data = tmp_path
        return cfg

    def test_pwa_enabled_wires_store(self, tmp_path) -> None:
        from paramem.server.app import _build_user_token_store

        cfg = self._config(tmp_path, mobile_pwa_enabled=True)
        result = _build_user_token_store(cfg)
        assert isinstance(result, UserTokenStore)

    def test_prior_mint_file_wires_store_even_when_pwa_disabled(self, tmp_path) -> None:
        """The file-existence branch: mobile_pwa.enabled=False, but
        user_tokens.json already exists on disk from a prior
        ``mint-user-token`` run — the server picks it up on this boot
        regardless of the PWA setting."""
        from paramem.server.app import _build_user_token_store

        cfg = self._config(tmp_path, mobile_pwa_enabled=False)
        # A real (empty) v2 store file, as a prior mint would have written.
        (tmp_path / "user_tokens.json").write_text('{"version": 2, "tokens": {}}')

        result = _build_user_token_store(cfg)
        assert isinstance(result, UserTokenStore)

    def test_neither_condition_stays_off(self, tmp_path) -> None:
        """Fresh install: mobile_pwa.enabled=False and no store file on disk
        yet → None (auth-OFF until the first mint)."""
        from paramem.server.app import _build_user_token_store

        cfg = self._config(tmp_path, mobile_pwa_enabled=False)
        assert not (tmp_path / "user_tokens.json").exists()

        result = _build_user_token_store(cfg)
        assert result is None


# ---------------------------------------------------------------------------
# cookie_name_getter path
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

        app = _make_full_app(
            user_token_getter=lambda: store,
            cookie_name="paramem_token",
            cookie_name_getter=cookie_name_getter,
        )
        return app, token

    def test_custom_cookie_name_getter_authorizes(self, tmp_path, monkeypatch):
        """cookie_name_getter returning 'custom_cookie' → that cookie authorizes."""
        app, token = self._make_app_with_getter(
            tmp_path, monkeypatch, cookie_name_getter=lambda: "custom_cookie"
        )
        client = TestClient(app, cookies={"custom_cookie": token})
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["speaker_id"] == "speaker0"

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
        assert resp.json()["speaker_id"] == "speaker0"


# ---------------------------------------------------------------------------
# Scope surfaced on request.state
# ---------------------------------------------------------------------------


class TestScopeOnRequestState:
    def test_per_user_admin_token_sets_admin_scope(self, tmp_path, monkeypatch):
        """A per-user admin token sets request.state.scope = 'admin'."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("Speaker0", "Admin", scope="admin")

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["scope"] == "admin"
        assert resp.json()["speaker_id"] == "speaker0"

    def test_per_user_chat_token_sets_chat_scope(self, tmp_path, monkeypatch):
        """A per-user chat token sets request.state.scope = 'chat'."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("speaker0", "Phone", scope="chat")

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["scope"] == "chat"

    def test_unattributed_chat_token_sets_chat_scope_no_speaker(self, tmp_path, monkeypatch):
        """An unattributed chat token → scope='chat', speaker_id absent on state."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint(None, "Shared Kitchen Tablet", scope="chat")

        app = _make_full_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/ping", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["scope"] == "chat"
        assert resp.json()["speaker_id"] is None

    def test_off_mode_stamps_chat_scope(self):
        """OFF mode (no store) → handler runs, scope stamped 'chat'.

        Fail-closed admin: OFF mode is open for use (no token check performed),
        but the request is stamped with the non-admin 'chat' scope — the same
        sentinel a per-user chat token carries — so ``require_admin`` denies
        admin endpoints until a credential is configured, while unguarded
        (chat/voice) endpoints stay reachable without credentials.
        """
        app = _make_full_app(user_token_getter=None)
        client = TestClient(app)

        resp = client.get("/ping")
        assert resp.status_code == 200  # OFF mode passes through (no scope gate here)
        assert resp.json()["scope"] == "chat"
