"""Tests for bearer-token resolution in ``paramem.cli.http_client``.

Covers:
1. env-only resolution: ``PARAMEM_API_TOKEN`` in env → returned.
2. secret-file resolution: temp HOME with secret file, env unset → returned.
3. repo ``.env`` resolution + precedence: secret file wins over ``.env``.
4. ``allow_files=False`` → ``None`` even with files present.
5. Quote-stripping: leading/trailing ``"`` and ``'`` are removed.
6. ``get_json``/``post_json`` attach/omit ``Authorization`` based on resolved token.

All tests use ``monkeypatch`` for env and HOME; no real ``~/.config`` or repo
``.env`` is ever touched.  The ``PARAMEM_CLI_NO_TOKEN_FILES=1`` flag set by
``tests/conftest.py`` is respected for tests that call the real
``get_json``/``post_json`` without overriding it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.cli import http_client
from paramem.cli.http_client import _repo_env_path, resolve_token

# ---------------------------------------------------------------------------
# 1. Env-variable resolution
# ---------------------------------------------------------------------------


class TestResolveTokenEnv:
    def test_returns_env_value_when_set(self, monkeypatch):
        """``PARAMEM_API_TOKEN`` in env → that value returned."""
        monkeypatch.setenv("PARAMEM_API_TOKEN", "env-tok-123")
        assert resolve_token() == "env-tok-123"

    def test_strips_whitespace_from_env(self, monkeypatch):
        """Leading/trailing whitespace in the env var is stripped."""
        monkeypatch.setenv("PARAMEM_API_TOKEN", "  tok  ")
        assert resolve_token() == "tok"

    def test_returns_none_when_env_empty(self, monkeypatch):
        """Empty env var → None (not an empty string)."""
        monkeypatch.setenv("PARAMEM_API_TOKEN", "")
        # conftest sets PARAMEM_CLI_NO_TOKEN_FILES=1 so files are not read.
        assert resolve_token() is None

    def test_env_takes_precedence_over_allow_files(self, monkeypatch, tmp_path):
        """Env value is returned before any file walk even when allow_files=True."""
        monkeypatch.setenv("PARAMEM_API_TOKEN", "env-wins")
        # Even with allow_files=True the env value is returned first.
        assert resolve_token(allow_files=True) == "env-wins"


# ---------------------------------------------------------------------------
# 2. Secret-file resolution
# ---------------------------------------------------------------------------


class TestResolveTokenSecretFile:
    def test_reads_secret_file_when_env_unset(self, monkeypatch, tmp_path):
        """Secret file at ``~/.config/paramem/secrets/PARAMEM_API_TOKEN`` → value returned."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        # Redirect HOME to a temp directory.
        monkeypatch.setenv("HOME", str(tmp_path))
        secret_dir = tmp_path / ".config" / "paramem" / "secrets"
        secret_dir.mkdir(parents=True)
        (secret_dir / "PARAMEM_API_TOKEN").write_text("file-secret-tok\n", encoding="utf-8")
        assert resolve_token(allow_files=True) == "file-secret-tok"

    def test_empty_secret_file_falls_through(self, monkeypatch, tmp_path):
        """Secret file exists but is empty → fall through to next source.

        The repo .env walk is neutralised by patching _repo_env_path to return
        None, so this test exercises only the secret-file branch falling through.
        """
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        secret_dir = tmp_path / ".config" / "paramem" / "secrets"
        secret_dir.mkdir(parents=True)
        (secret_dir / "PARAMEM_API_TOKEN").write_text("   ", encoding="utf-8")
        # No repo .env walk (neutralised) → falls through to None.
        with patch.object(http_client, "_repo_env_path", return_value=None):
            result = resolve_token(allow_files=True)
        assert result is None


# ---------------------------------------------------------------------------
# 3. Repo ``.env`` resolution and precedence (via _repo_env_path helper)
# ---------------------------------------------------------------------------


class TestResolveTokenRepoEnv:
    def _make_repo_tree(self, tmp_path: Path, *, token_value: str) -> Path:
        """Build a minimal repo tree: pyproject.toml + .env with PARAMEM_API_TOKEN.

        Creates *tmp_path* if it does not yet exist.
        """
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        (tmp_path / ".env").write_text(f"PARAMEM_API_TOKEN={token_value}\n", encoding="utf-8")
        # Return a synthetic __file__-like start path inside the repo.
        sub = tmp_path / "paramem" / "cli"
        sub.mkdir(parents=True)
        return sub / "http_client.py"

    def test_repo_env_path_finds_pyproject(self, tmp_path):
        """``_repo_env_path`` returns the ``.env`` path beside ``pyproject.toml``."""
        start = self._make_repo_tree(tmp_path, token_value="x")
        result = _repo_env_path(start)
        assert result == tmp_path / ".env"

    def test_repo_env_path_returns_none_without_pyproject(self, tmp_path):
        """``_repo_env_path`` returns None when no ``pyproject.toml`` is found."""
        start = tmp_path / "some" / "deep" / "path.py"
        start.parent.mkdir(parents=True)
        assert _repo_env_path(start) is None

    def test_reads_repo_env_when_no_secret_file(self, monkeypatch, tmp_path):
        """Repo ``.env`` is read when env var and secret file are both absent."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        # No secret file — redirect HOME to tmp_path so ~/.config doesn't exist.
        monkeypatch.setenv("HOME", str(tmp_path))
        start = self._make_repo_tree(tmp_path / "repo", token_value="repo-env-tok")
        env_path = _repo_env_path(start)
        assert env_path is not None
        # Call resolve_token with an explicit start path via _repo_env_path directly
        # to avoid monkeypatching __file__.  Verify _repo_env_path returns the right path,
        # then verify resolve_token reads it when triggered from the same tree.
        assert env_path.is_file()
        # Read the .env ourselves to confirm format, then test resolve_token via
        # a controlled temp HOME (no secret file) and patched __file__.
        with patch.object(http_client, "_repo_env_path", return_value=env_path):
            result = resolve_token(allow_files=True)
        assert result == "repo-env-tok"

    def test_secret_file_wins_over_repo_env(self, monkeypatch, tmp_path):
        """Secret file takes precedence over repo ``.env``."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        # Write the secret file.
        secret_dir = tmp_path / ".config" / "paramem" / "secrets"
        secret_dir.mkdir(parents=True)
        (secret_dir / "PARAMEM_API_TOKEN").write_text("secret-file-tok\n", encoding="utf-8")
        # Write a repo .env with a different value.
        repo = tmp_path / "repo"
        repo.mkdir()
        start = self._make_repo_tree(repo, token_value="repo-tok-should-not-win")
        env_path = _repo_env_path(start)
        with patch.object(http_client, "_repo_env_path", return_value=env_path):
            result = resolve_token(allow_files=True)
        assert result == "secret-file-tok"


# ---------------------------------------------------------------------------
# 4. allow_files=False
# ---------------------------------------------------------------------------


class TestResolveTokenAllowFilesFalse:
    def test_returns_none_with_no_env_and_files_disabled(self, monkeypatch, tmp_path):
        """``allow_files=False`` → None even when files with tokens exist."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        # Write a secret file that would otherwise be found.
        secret_dir = tmp_path / ".config" / "paramem" / "secrets"
        secret_dir.mkdir(parents=True)
        (secret_dir / "PARAMEM_API_TOKEN").write_text("should-not-be-read\n", encoding="utf-8")
        assert resolve_token(allow_files=False) is None

    def test_env_cli_no_token_files_flag_disables_files(self, monkeypatch, tmp_path):
        """``PARAMEM_CLI_NO_TOKEN_FILES=1`` env flag disables file lookups."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        monkeypatch.setenv("PARAMEM_CLI_NO_TOKEN_FILES", "1")
        monkeypatch.setenv("HOME", str(tmp_path))
        secret_dir = tmp_path / ".config" / "paramem" / "secrets"
        secret_dir.mkdir(parents=True)
        (secret_dir / "PARAMEM_API_TOKEN").write_text("should-not-be-read\n", encoding="utf-8")
        # allow_files=None → reads PARAMEM_CLI_NO_TOKEN_FILES → False → no files.
        assert resolve_token() is None


# ---------------------------------------------------------------------------
# 5. Quote-stripping
# ---------------------------------------------------------------------------


class TestResolveTokenQuoteStripping:
    def test_strips_double_quotes_from_env_file_value(self, monkeypatch, tmp_path):
        """Double-quoted values in ``.env`` are unquoted."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (repo / ".env").write_text('PARAMEM_API_TOKEN="quoted-tok"\n', encoding="utf-8")
        env_path = _repo_env_path(repo / "sub" / "start.py")
        with patch.object(http_client, "_repo_env_path", return_value=env_path):
            result = resolve_token(allow_files=True)
        assert result == "quoted-tok"

    def test_strips_single_quotes_from_env_file_value(self, monkeypatch, tmp_path):
        """Single-quoted values in ``.env`` are unquoted."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (repo / ".env").write_text("PARAMEM_API_TOKEN='single-tok'\n", encoding="utf-8")
        env_path = _repo_env_path(repo / "sub" / "start.py")
        with patch.object(http_client, "_repo_env_path", return_value=env_path):
            result = resolve_token(allow_files=True)
        assert result == "single-tok"

    def test_strips_quotes_from_secret_file_value(self, monkeypatch, tmp_path):
        """Quoted value in secret file is unquoted."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        secret_dir = tmp_path / ".config" / "paramem" / "secrets"
        secret_dir.mkdir(parents=True)
        (secret_dir / "PARAMEM_API_TOKEN").write_text('"quoted-secret"\n', encoding="utf-8")
        assert resolve_token(allow_files=True) == "quoted-secret"


# ---------------------------------------------------------------------------
# 6. get_json / post_json attach/omit Authorization header
# ---------------------------------------------------------------------------


class TestHttpClientAuthHeader:
    """Verify that get_json/post_json send or omit Authorization based on resolve_token."""

    def _make_mock_client(self, status_code: int = 200, json_body: dict | None = None):
        """Build a MagicMock httpx.Client context manager with a stub response."""
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_body or {"ok": True}
        resp.text = ""
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = resp
        mock_client.post.return_value = resp
        return mock_client

    def test_get_json_attaches_header_when_token_resolved(self, monkeypatch):
        """get_json sends Authorization when resolve_token returns a token."""
        monkeypatch.setenv("PARAMEM_API_TOKEN", "bearer-tok")
        captured: dict = {}

        def fake_get(url, *, headers=None):
            captured["headers"] = headers or {}
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"ok": True}
            return resp

        mock_client = self._make_mock_client()
        mock_client.get = fake_get

        with patch("paramem.cli.http_client.httpx.Client", return_value=mock_client):
            http_client.get_json("http://127.0.0.1:8420/status")

        assert captured["headers"].get("Authorization") == "Bearer bearer-tok"

    def test_get_json_omits_header_when_no_token(self, monkeypatch):
        """get_json omits Authorization when resolve_token returns None.

        conftest sets PARAMEM_CLI_NO_TOKEN_FILES=1 and pops PARAMEM_API_TOKEN,
        so resolve_token() returns None → no header.
        """
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        captured: dict = {}

        def fake_get(url, *, headers=None):
            captured["headers"] = headers or {}
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"ok": True}
            return resp

        mock_client = self._make_mock_client()
        mock_client.get = fake_get

        with patch("paramem.cli.http_client.httpx.Client", return_value=mock_client):
            http_client.get_json("http://127.0.0.1:8420/status")

        assert "Authorization" not in captured["headers"]

    def test_post_json_attaches_header_when_token_resolved(self, monkeypatch):
        """post_json sends Authorization when resolve_token returns a token."""
        monkeypatch.setenv("PARAMEM_API_TOKEN", "post-bearer-tok")
        captured: dict = {}

        def fake_post(url, *, json=None, headers=None):
            captured["headers"] = headers or {}
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"ok": True}
            return resp

        mock_client = self._make_mock_client()
        mock_client.post = fake_post

        with patch("paramem.cli.http_client.httpx.Client", return_value=mock_client):
            http_client.post_json("http://127.0.0.1:8420/consolidate")

        assert captured["headers"].get("Authorization") == "Bearer post-bearer-tok"

    def test_post_json_omits_header_when_no_token(self, monkeypatch):
        """post_json omits Authorization when resolve_token returns None."""
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        captured: dict = {}

        def fake_post(url, *, json=None, headers=None):
            captured["headers"] = headers or {}
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"ok": True}
            return resp

        mock_client = self._make_mock_client()
        mock_client.post = fake_post

        with patch("paramem.cli.http_client.httpx.Client", return_value=mock_client):
            http_client.post_json("http://127.0.0.1:8420/consolidate")

        assert "Authorization" not in captured["headers"]

    def test_explicit_token_arg_overrides_resolve(self, monkeypatch):
        """Passing an explicit token string skips resolve_token and uses it directly."""
        # env is absent — without the explicit arg, token would be None.
        monkeypatch.delenv("PARAMEM_API_TOKEN", raising=False)
        captured: dict = {}

        def fake_post(url, *, json=None, headers=None):
            captured["headers"] = headers or {}
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"ok": True}
            return resp

        mock_client = self._make_mock_client()
        mock_client.post = fake_post

        with patch("paramem.cli.http_client.httpx.Client", return_value=mock_client):
            http_client.post_json("http://127.0.0.1:8420/consolidate", token="override-tok")

        assert captured["headers"].get("Authorization") == "Bearer override-tok"
