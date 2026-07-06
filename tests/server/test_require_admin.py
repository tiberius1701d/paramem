"""Tests for the ``require_admin`` FastAPI dependency and per-scope route gating.

CPU-only — no model load.  Uses a minimal FastAPI app with
``BearerTokenMiddleware`` to verify that:

- chat-scope tokens are denied on admin endpoints (403 admin_scope_required).
- admin-scope tokens are accepted on admin endpoints.
- chat-scope tokens reach /chat, /voice (path-level), /push/* without 403.
- auth-OFF mode allows admin endpoints (server open by design — no credential).
- The real app route table has ``require_admin`` on every audited admin path.
- /admin/assign-orphans is reachable by a per-user admin token (ON-per-user
  mode, shared env token unset).
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
from paramem.server.app import require_admin
from paramem.server.auth import BearerTokenMiddleware
from paramem.server.user_tokens import UserTokenStore

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_CHAT_SCOPE_PATHS = {
    "/chat",
    "/voice",
    "/push/vapid-public-key",
    "/push/subscribe",
    "/status",
}
# These paths are exempt from auth entirely (PWA shell, manifest, liveness).
_EXEMPT_PATHS = {"/", "/app", "/manifest.json", "/health"}


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
) -> FastAPI:
    """Minimal FastAPI app with BearerTokenMiddleware + require_admin."""
    from fastapi import Depends

    app = FastAPI()
    app.add_middleware(
        BearerTokenMiddleware,
        token=shared_token,
        user_token_getter=user_token_getter,
    )

    @app.get("/admin-only", dependencies=[Depends(require_admin)])
    async def admin_only():
        return {"ok": True, "admin": True}

    @app.post("/admin-post", dependencies=[Depends(require_admin)])
    async def admin_post():
        return {"ok": True}

    @app.post("/chat")
    async def chat_endpoint(request: Request):
        return {"ok": True, "scope": getattr(request.state, "scope", None)}

    return app


# ---------------------------------------------------------------------------
# Chat-scope token denied on admin endpoint
# ---------------------------------------------------------------------------


class TestChatScopeOnAdminRoute:
    def test_chat_token_403_on_admin_get(self, tmp_path, monkeypatch):
        """A chat-scope per-user token → 403 on admin GET endpoint."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("speaker0", "Device", scope="chat")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403
        body = resp.json()
        # FastAPI wraps HTTPException detail in {"detail": ...}
        assert "admin_scope_required" in str(body)

    def test_chat_token_403_on_admin_post(self, tmp_path, monkeypatch):
        """A chat-scope per-user token → 403 on admin POST endpoint."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("speaker0", "Device", scope="chat")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.post("/admin-post", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_off_mode_allows_admin_endpoint(self, tmp_path, monkeypatch):
        """Auth OFF (no token, no store) → admin endpoint reachable (open by design).

        OFF mode is fully open: no credential is configured.
        The middleware stamps ``scope == "admin"`` on every OFF-mode pass-through,
        so admin endpoints stay reachable as they were before the scope gate.
        """
        # Build an app with no shared token and no store → OFF mode.
        app = _make_app(shared_token="", user_token_getter=None)
        client = TestClient(app)

        # OFF mode: middleware stamps admin scope; require_admin allows.
        resp = client.get("/admin-only")
        assert resp.status_code == 200
        assert resp.json()["admin"] is True


# ---------------------------------------------------------------------------
# Admin-scope token accepted on admin endpoint
# ---------------------------------------------------------------------------


class TestAdminScopeOnAdminRoute:
    def test_admin_token_200_on_admin_get(self, tmp_path, monkeypatch):
        """An admin-scope per-user token → 200 on admin GET endpoint."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("speaker0", "Admin Device", scope="admin")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["admin"] is True

    def test_shared_token_200_on_admin_get(self):
        """The shared env token (always admin) → 200 on admin GET endpoint."""
        app = _make_app(shared_token="shared-secret")
        client = TestClient(app)

        resp = client.get("/admin-only", headers={"Authorization": "Bearer shared-secret"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Chat-scope token reaches /chat (not 403)
# ---------------------------------------------------------------------------


class TestChatScopeReachesChat:
    def test_chat_token_reaches_chat_endpoint(self, tmp_path, monkeypatch):
        """A chat-scope per-user token can reach /chat without 403."""
        _setup_daily(tmp_path, monkeypatch)
        store = _make_store(tmp_path)
        token = store.mint("speaker0", "Phone", scope="chat")

        app = _make_app(user_token_getter=lambda: store)
        client = TestClient(app)

        resp = client.post("/chat", headers={"Authorization": f"Bearer {token}"})
        # Endpoint exists and returns 200 (not 403).
        assert resp.status_code == 200
        assert resp.json()["scope"] == "chat"


# ---------------------------------------------------------------------------
# Route-table introspection: every admin path carries require_admin
# ---------------------------------------------------------------------------

_ADMIN_PATHS = {
    "/gpu/acquire",
    "/gpu/release",
    "/refresh-ha",
    "/admin/assign-orphans",
    "/scheduled-tick",
    "/consolidate",
    "/consolidate/housekeeping",
    "/ingest-sessions",
    "/ingest-sessions/cancel",
    "/migration/preview",
    "/migration/cancel",
    "/migration/confirm",
    "/migration/status",
    "/migration/diff",
    "/migration/accept",
    "/migration/rollback",
    "/backup/list",
    "/backup/create",
    "/backup/restore",
    "/backup/prune",
    "/debug/probe",
    "/debug/recall",
    "/debug/dump",
    "/calibrate/extract",
    "/calibrate/anonymize",
    "/calibrate/plausibility",
    "/integrity",
    "/speaker/forget",
}


class TestRouteTableIntrospection:
    """Verify the live app route table has require_admin on every audited path.

    Uses FastAPI route introspection (``route.dependant.dependencies``) so
    a future unguarded admin route fails this test immediately, without a
    model load or a TestClient request.
    """

    def _dep_callables(self, route) -> list:
        """Extract the callable from each dependency on a route."""
        deps = []
        try:
            for d in route.dependant.dependencies:
                deps.append(d.call)
        except AttributeError:
            pass
        return deps

    def test_all_admin_paths_carry_require_admin(self):
        """Every path in _ADMIN_PATHS must carry require_admin in its dependencies."""
        from paramem.server.app import app

        route_map: dict[str, list] = {}
        for route in app.routes:
            if hasattr(route, "path") and hasattr(route, "dependant"):
                route_map[route.path] = self._dep_callables(route)

        missing = []
        for path in _ADMIN_PATHS:
            deps = route_map.get(path, [])
            if require_admin not in deps:
                missing.append(path)

        assert not missing, (
            f"Admin paths missing require_admin dependency: {missing!r}\n"
            "Add dependencies=[Depends(require_admin)] to each listed route."
        )

    def test_chat_scope_paths_do_not_carry_require_admin(self):
        """Chat-scope paths must NOT carry require_admin."""
        from paramem.server.app import app

        route_map: dict[str, list] = {}
        for route in app.routes:
            if hasattr(route, "path") and hasattr(route, "dependant"):
                route_map[route.path] = self._dep_callables(route)

        wrongly_gated = []
        for path in _CHAT_SCOPE_PATHS:
            deps = route_map.get(path, [])
            if require_admin in deps:
                wrongly_gated.append(path)

        assert not wrongly_gated, (
            f"Chat-scope paths should NOT have require_admin: {wrongly_gated!r}"
        )


# ---------------------------------------------------------------------------
# /admin/assign-orphans reachable by per-user admin token (ON-per-user mode)
# ---------------------------------------------------------------------------


class TestAssignOrphansPerUserAdmin:
    """Verify /admin/assign-orphans is reachable with a per-user admin token
    when the shared env token (PARAMEM_API_TOKEN) is unset.

    This tests the intended loosening: per-user admin tokens can reach
    admin endpoints without requiring the shared token to be set.

    Uses route-table introspection only (no model load / TestClient against
    the full app) — the dependency presence guarantees the auth path.
    """

    def test_assign_orphans_has_require_admin_not_inline_check(self):
        """POST /admin/assign-orphans must have require_admin dependency.

        Verifies the inline _api_token check was removed and replaced by the
        dependency, so a per-user admin token can also reach it.
        """
        from paramem.server.app import app

        route_map = {}
        for route in app.routes:
            if hasattr(route, "path") and hasattr(route, "dependant"):
                route_map[route.path] = route

        route = route_map.get("/admin/assign-orphans")
        assert route is not None, "/admin/assign-orphans route not found in app"
        deps = [d.call for d in route.dependant.dependencies]
        assert require_admin in deps, "/admin/assign-orphans must carry require_admin dependency"


# ---------------------------------------------------------------------------
# /health — unauthenticated liveness probe
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """/health must return 200 + {"status": "ok"} without any Authorization header.

    Uses the minimal test app wired with BearerTokenMiddleware in ON-shared
    mode (a shared token is set) so that any path NOT in exempt_paths would
    ordinarily be rejected with 401.  Verifying 200 on /health confirms the
    exemption is in effect.
    """

    def _make_guarded_app(self) -> FastAPI:
        """Minimal FastAPI app with BearerTokenMiddleware in ON-shared mode."""
        from fastapi import FastAPI

        mini_app = FastAPI()
        mini_app.add_middleware(
            BearerTokenMiddleware,
            token="secret-token",
            exempt_paths=("/health",),
        )

        @mini_app.get("/health")
        async def health():
            return {"status": "ok"}

        @mini_app.get("/protected")
        async def protected():
            return {"ok": True}

        return mini_app

    def test_health_200_without_token(self):
        """/health returns 200 + {"status": "ok"} with no Authorization header."""
        app = self._make_guarded_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_200_even_when_token_configured(self):
        """/health is reachable even when a shared token is set (auth ON mode)."""
        app = self._make_guarded_app()
        client = TestClient(app)

        # Confirm auth IS active by verifying protected endpoint requires token.
        resp_protected = client.get("/protected")
        assert resp_protected.status_code == 401, (
            "Protected endpoint should be 401 without token — auth is not active"
        )

        # Now confirm /health bypasses that requirement.
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        assert resp_health.json() == {"status": "ok"}

    def test_real_app_health_exempt_in_middleware(self):
        """/health is listed in the real app's BearerTokenMiddleware exempt_paths."""
        from paramem.server.app import app as real_app

        # FastAPI stores pending middleware entries in app.user_middleware before
        # the middleware stack is first built.  Each entry has .cls and .kwargs.
        mw_entries = getattr(real_app, "user_middleware", [])
        btm = None
        for entry in mw_entries:
            cls = entry.cls if hasattr(entry, "cls") else None
            if cls is BearerTokenMiddleware:
                kwargs = entry.kwargs if hasattr(entry, "kwargs") else {}
                btm_exempt = kwargs.get("exempt_paths", ())
                assert "/health" in btm_exempt, (
                    f"/health not found in BearerTokenMiddleware exempt_paths: {btm_exempt!r}"
                )
                btm = entry
                break

        assert btm is not None, "BearerTokenMiddleware not found in app.user_middleware"

    def test_health_not_in_admin_paths(self):
        """/health must NOT appear in _ADMIN_PATHS (it requires no auth at all)."""
        assert "/health" not in _ADMIN_PATHS

    def test_health_in_exempt_paths(self):
        """/health must appear in _EXEMPT_PATHS."""
        assert "/health" in _EXEMPT_PATHS
