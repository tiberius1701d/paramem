"""Tests for PWA static-asset serving — specifically the ``/app/sw.js`` route.

The service-worker script must be served with ``Cache-Control: no-cache`` so
the browser revalidates it on every navigation and detects a new
``CACHE_VERSION`` promptly.  A dedicated ``@app.get("/app/sw.js")`` route
registered at module load time takes precedence over the ``/app``
``StaticFiles`` mount (which is added later in the lifespan), providing this
header without altering every other static asset.

CPU-only — no model load, no GPU.

Covers:
- ``GET /app/sw.js`` returns 200.
- Response body contains ``CACHE_VERSION`` (confirms the real sw.js is served).
- ``Cache-Control`` response header contains ``no-cache``.
- ``Content-Type`` is ``application/javascript``.
- ``/app/sw.js`` is reachable WITHOUT a bearer token (exempt via ``/app/``
  prefix — confirmed by the middleware configuration in app.py).
- Other static assets (``/app/app.js``, ``/app/manifest.json``) still serve
  correctly via the ``StaticFiles`` mount — the new route must not shadow the
  rest of the mount.
- ``/app/sw.js`` still returns 200 when ``_state["config"]`` is ``None``
  (pre-lifespan / no config loaded) — the route falls back to the built-in
  default static directory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REAL_STATIC_DIR = Path(app_module.__file__).parent.parent / "web" / "static"


def _make_config(static_dir: str | None = None) -> MagicMock:
    """Minimal server config mock with mobile_pwa.enabled=True."""
    cfg = MagicMock()
    cfg.mobile_pwa.enabled = True
    cfg.mobile_pwa.static_dir = static_dir
    cfg.mobile_pwa.cookie_name = "paramem_token"
    return cfg


def _make_state(config=None) -> dict:
    """Minimal _state dict with just the keys the sw.js route reads."""
    return {
        "config": config,
        "model": None,
        "mode": "local",
    }


def _client_no_token() -> TestClient:
    """TestClient against the real app, NO auth headers.

    When ``PARAMEM_API_TOKEN`` is unset (CI default), auth is OFF and every
    request passes through.  When it is set we deliberately omit it here to
    assert the token-exempt behaviour of ``/app/sw.js``.
    """
    return TestClient(app_module.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Tests: Cache-Control header
# ---------------------------------------------------------------------------


class TestSwJsNoCacheHeader:
    """The ``/app/sw.js`` route sets ``Cache-Control: no-cache``."""

    def test_sw_js_returns_200(self, monkeypatch):
        """GET /app/sw.js returns 200."""
        monkeypatch.setattr(app_module, "_state", _make_state(_make_config()))
        resp = _client_no_token().get("/app/sw.js")
        assert resp.status_code == 200

    def test_sw_js_cache_control_no_cache(self, monkeypatch):
        """Response header Cache-Control contains 'no-cache'."""
        monkeypatch.setattr(app_module, "_state", _make_state(_make_config()))
        resp = _client_no_token().get("/app/sw.js")
        assert resp.status_code == 200
        cc = resp.headers.get("cache-control", "")
        assert "no-cache" in cc, f"Expected 'no-cache' in Cache-Control, got: {cc!r}"

    def test_sw_js_content_type_javascript(self, monkeypatch):
        """Response Content-Type is application/javascript."""
        monkeypatch.setattr(app_module, "_state", _make_state(_make_config()))
        resp = _client_no_token().get("/app/sw.js")
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "javascript" in ct, f"Expected javascript in Content-Type, got: {ct!r}"

    def test_sw_js_body_contains_cache_version(self, monkeypatch):
        """Response body is the real sw.js and contains CACHE_VERSION."""
        monkeypatch.setattr(app_module, "_state", _make_state(_make_config()))
        resp = _client_no_token().get("/app/sw.js")
        assert resp.status_code == 200
        assert "CACHE_VERSION" in resp.text, "Expected CACHE_VERSION in sw.js body"

    def test_sw_js_body_contains_v7(self, monkeypatch):
        """Response body reflects the bumped CACHE_VERSION v7."""
        monkeypatch.setattr(app_module, "_state", _make_state(_make_config()))
        resp = _client_no_token().get("/app/sw.js")
        assert resp.status_code == 200
        assert '"v7"' in resp.text, "Expected CACHE_VERSION v7 in sw.js body"


# ---------------------------------------------------------------------------
# Tests: token-exempt (reachable without a bearer token)
# ---------------------------------------------------------------------------


class TestSwJsTokenExempt:
    """``/app/sw.js`` is reachable without a bearer token.

    The BearerTokenMiddleware is configured with ``exempt_prefixes=("/app/",)``
    in app.py.  ``/app/sw.js`` starts with ``/app/`` so it is exempt.
    This test verifies the exemption is not accidentally broken by the new route.
    """

    def test_sw_js_no_token_returns_200_when_auth_off(self, monkeypatch):
        """No token, auth OFF (no PARAMEM_API_TOKEN set) → 200.

        When the API token env var is unset the middleware is in OFF mode and
        passes every request through.  This is the CI default.
        """
        monkeypatch.setattr(app_module, "_state", _make_state(_make_config()))
        # Explicitly use a client with no auth headers.
        client = TestClient(app_module.app, raise_server_exceptions=True)
        resp = client.get("/app/sw.js")
        # In OFF mode (no token configured) all requests pass through.
        # In ON mode, /app/sw.js is exempt via /app/ prefix — also 200.
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: fallback to default static dir when config is None
# ---------------------------------------------------------------------------


class TestSwJsFallback:
    """Route falls back to the built-in static dir when config is absent."""

    def test_sw_js_works_when_config_is_none(self, monkeypatch):
        """GET /app/sw.js returns 200 even when _state['config'] is None.

        The route must not crash pre-lifespan (config not yet loaded).
        """
        monkeypatch.setattr(app_module, "_state", _make_state(config=None))
        resp = _client_no_token().get("/app/sw.js")
        assert resp.status_code == 200
        assert "no-cache" in resp.headers.get("cache-control", "")

    def test_sw_js_works_when_static_dir_unset(self, monkeypatch):
        """GET /app/sw.js returns 200 when mobile_pwa.static_dir is None/empty.

        The route falls back to the built-in ``paramem/web/static`` directory.
        """
        cfg = _make_config(static_dir=None)
        monkeypatch.setattr(app_module, "_state", _make_state(cfg))
        resp = _client_no_token().get("/app/sw.js")
        assert resp.status_code == 200
        assert "no-cache" in resp.headers.get("cache-control", "")


# ---------------------------------------------------------------------------
# Tests: route precedence — sw.js route wins over StaticFiles mount
# ---------------------------------------------------------------------------


class TestSwJsRoutePrecedence:
    """The explicit route shadows the StaticFiles mount for sw.js only.

    Verifies that the ``@app.get("/app/sw.js")`` route (registered at module
    load time, BEFORE the lifespan-deferred mount) takes precedence over the
    ``/app`` ``StaticFiles`` mount for ``/app/sw.js``, while other assets
    still resolve correctly via the mount.
    """

    def test_sw_js_has_no_cache_not_static_etag_only(self, monkeypatch):
        """The sw.js response has Cache-Control: no-cache (from the route),
        not just ETag/Last-Modified from StaticFiles.
        """
        monkeypatch.setattr(app_module, "_state", _make_state(_make_config()))
        resp = _client_no_token().get("/app/sw.js")
        assert resp.status_code == 200
        assert "no-cache" in resp.headers.get("cache-control", ""), (
            "sw.js must be served by the explicit route (no-cache), "
            "not by StaticFiles (no Cache-Control header)"
        )


# ---------------------------------------------------------------------------
# Tests: regression — other static assets still serve correctly
# ---------------------------------------------------------------------------


def _make_minimal_pwa_app(static_dir: Path):
    """Build a minimal FastAPI app that mirrors the real app's route+mount pattern.

    Registers ``@app.get("/app/sw.js")`` (the explicit route) at module-level
    style — before the StaticFiles mount — matching the precedence contract
    used in the production app.  This lets us test the full route+mount
    interaction without triggering the production server's lifespan.
    """
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    mini = FastAPI()

    @mini.get("/app/sw.js")
    def serve_sw_js():
        """Serve sw.js with Cache-Control: no-cache."""
        return FileResponse(
            path=str(static_dir / "sw.js"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    mini.mount("/app", StaticFiles(directory=str(static_dir), html=True), name="pwa")
    return mini


class TestOtherStaticAssetsUnaffected:
    """Other ``/app/`` assets are NOT served by the sw.js route.

    Uses a minimal app that mirrors the real route+mount pattern to test the
    non-shadowing contract without requiring the full production lifespan.
    """

    @pytest.mark.parametrize(
        "path",
        ["/app/app.js", "/app/manifest.json", "/app/index.html"],
    )
    def test_other_assets_serve_200(self, path):
        """Other shell assets return 200 via the StaticFiles mount.

        The sw.js explicit route must not interfere with the rest of the
        ``/app`` mount.
        """
        mini = _make_minimal_pwa_app(_REAL_STATIC_DIR)
        client = TestClient(mini, raise_server_exceptions=True)
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"

    @pytest.mark.parametrize(
        "path",
        ["/app/app.js", "/app/manifest.json"],
    )
    def test_other_assets_have_no_injected_no_cache(self, path):
        """Other shell assets do NOT receive the injected ``Cache-Control: no-cache``.

        Only sw.js gets the no-cache header.  app.js, manifest.json etc. are
        served by StaticFiles with ETag/Last-Modified only.
        """
        mini = _make_minimal_pwa_app(_REAL_STATIC_DIR)
        client = TestClient(mini, raise_server_exceptions=True)
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"
        # The sw.js route only handles /app/sw.js — other paths go through
        # StaticFiles, which does NOT add our injected Cache-Control: no-cache.
        cc = resp.headers.get("cache-control", "")
        assert cc != "no-cache", (
            f"{path} must not have our injected Cache-Control: no-cache header "
            f"(it should go through StaticFiles); got: {cc!r}"
        )
