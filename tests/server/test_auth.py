"""Tests for `paramem.server.auth` — WP3 bearer-token middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paramem.server.auth import BearerTokenMiddleware, load_token_from_env


def _make_app(token: str) -> FastAPI:
    app = FastAPI()
    app.add_middleware(BearerTokenMiddleware, token=token)

    @app.get("/ping")
    def ping() -> dict:
        return {"ok": True}

    @app.post("/echo")
    def echo(payload: dict) -> dict:
        return payload

    return app


class TestBearerTokenMiddleware:
    def test_disabled_when_token_empty(self) -> None:
        client = TestClient(_make_app(""))
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_disabled_when_token_none_spaces(self) -> None:
        # load_token_from_env strips whitespace before handing to middleware,
        # but defensive: the middleware treats empty as disabled.
        client = TestClient(_make_app(""))
        response = client.post("/echo", json={"x": 1})
        assert response.status_code == 200

    def test_rejects_without_header(self) -> None:
        client = TestClient(_make_app("secret-abc"))
        response = client.get("/ping")
        assert response.status_code == 401
        body = response.json()
        assert body["error"] == "unauthorized"
        assert "missing" in body["detail"].lower()
        assert response.headers.get("WWW-Authenticate", "").startswith("Bearer")

    def test_rejects_malformed_header(self) -> None:
        client = TestClient(_make_app("secret-abc"))
        response = client.get("/ping", headers={"Authorization": "Token secret-abc"})
        assert response.status_code == 401
        assert "malformed" in response.json()["detail"].lower()

    def test_rejects_wrong_token(self) -> None:
        client = TestClient(_make_app("secret-abc"))
        response = client.get("/ping", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    def test_accepts_correct_token(self) -> None:
        client = TestClient(_make_app("secret-abc"))
        response = client.get("/ping", headers={"Authorization": "Bearer secret-abc"})
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_accepts_correct_token_with_trailing_space(self) -> None:
        # The middleware strips trailing whitespace on the presented token.
        client = TestClient(_make_app("secret-abc"))
        response = client.get("/ping", headers={"Authorization": "Bearer secret-abc   "})
        assert response.status_code == 200

    def test_enabled_flag(self) -> None:
        mw_on = BearerTokenMiddleware(None, token="x")
        mw_off = BearerTokenMiddleware(None, token="")
        assert mw_on.enabled is True
        assert mw_off.enabled is False

    def test_constant_time_comparison_path_works_for_equal_length(self) -> None:
        # hmac.compare_digest only accepts equal-length inputs for timing
        # guarantees on some Python versions; verify a near-miss same-length token
        # is correctly rejected.
        client = TestClient(_make_app("abcdef"))
        response = client.get("/ping", headers={"Authorization": "Bearer abcdez"})
        assert response.status_code == 401


class TestLoadTokenFromEnv:
    def test_returns_empty_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        assert load_token_from_env() == ""

    def test_returns_stripped_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARAMEM_API_TOKEN", "  sekret  ")
        assert load_token_from_env() == "sekret"

    def test_returns_empty_for_whitespace_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARAMEM_API_TOKEN", "   ")
        assert load_token_from_env() == ""
