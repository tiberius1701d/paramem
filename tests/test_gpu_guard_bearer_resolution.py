"""Unit tests for the GPU_GUARD_HTTP_BEARER resolution helper.

Tests the pure :func:`experiments.utils.gpu_guard._resolve_http_bearer`
function directly so that:

- No real ``os.environ`` is mutated.
- The conftest ``PARAMEM_API_TOKEN`` pop (conftest.py:60) is not bypassed.
- No GPU or gpu_guard import side-effects are triggered.

gpu_guard is provided by lab-tools (separate repo, not on PyPI).  CI skips
tests in ``test_gpu_guard_shim.py`` via ``pytest.importorskip``; here we do
the same so the whole module is skipped cleanly when gpu_guard is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gpu_guard")

# Import only the pure helper — not the module top-level so we avoid the
# add_default_consumer / set_default_notifier side-effects in collection.
from experiments.utils.gpu_guard import _resolve_http_bearer  # noqa: E402


class TestResolveBearerPrecedence:
    """Verify the four-step resolution order."""

    def test_gpu_guard_http_bearer_already_set_wins(self, tmp_path):
        """(1) GPU_GUARD_HTTP_BEARER in env is returned unchanged."""
        env = {
            "GPU_GUARD_HTTP_BEARER": "existing-bearer",
            "PARAMEM_API_TOKEN": "should-not-be-used",
        }
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("PARAMEM_API_TOKEN=dotenv-token\n")

        result = _resolve_http_bearer(env, dotenv_path)

        assert result == "existing-bearer"

    def test_paramem_api_token_ambient_copied_when_no_bearer(self, tmp_path):
        """(2) PARAMEM_API_TOKEN in ambient env → returned as bearer."""
        env = {"PARAMEM_API_TOKEN": "ambient-token"}
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("PARAMEM_API_TOKEN=dotenv-token\n")

        result = _resolve_http_bearer(env, dotenv_path)

        assert result == "ambient-token"

    def test_dotenv_fallback_when_no_ambient_token(self, tmp_path):
        """(3) .env PARAMEM_API_TOKEN used when neither bearer nor ambient token present."""
        env: dict[str, str] = {}
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("PARAMEM_API_TOKEN=dotenv-only-token\n")

        result = _resolve_http_bearer(env, dotenv_path)

        assert result == "dotenv-only-token"

    def test_none_returned_when_nothing_available(self, tmp_path):
        """(4) No token anywhere → None (gpu_guard omits the header)."""
        env: dict[str, str] = {}
        dotenv_path = tmp_path / ".env"  # does not exist

        result = _resolve_http_bearer(env, dotenv_path)

        assert result is None

    def test_none_returned_when_dotenv_missing_key(self, tmp_path):
        """(4b) .env exists but has no PARAMEM_API_TOKEN → None."""
        env: dict[str, str] = {}
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("SOME_OTHER_VAR=foo\n")

        result = _resolve_http_bearer(env, dotenv_path)

        assert result is None

    def test_none_returned_when_dotenv_file_absent(self, tmp_path):
        """(4c) .env path does not exist → None, no error."""
        env: dict[str, str] = {}
        dotenv_path = tmp_path / "nonexistent.env"

        result = _resolve_http_bearer(env, dotenv_path)

        assert result is None


class TestResolveBearerNeverSetsParamemApiToken:
    """PARAMEM_API_TOKEN must never be the return value or stored in env."""

    def test_result_is_not_paramem_api_token_key(self, tmp_path):
        """The helper returns the token value, never the key name."""
        env = {"PARAMEM_API_TOKEN": "secret-value"}
        dotenv_path = tmp_path / ".env"

        result = _resolve_http_bearer(env, dotenv_path)

        # Value is returned, not the key name
        assert result == "secret-value"
        assert result != "PARAMEM_API_TOKEN"

    def test_env_dict_unchanged(self, tmp_path):
        """The helper does not mutate the env dict passed to it."""
        env = {"PARAMEM_API_TOKEN": "secret-value"}
        original_keys = set(env.keys())
        dotenv_path = tmp_path / ".env"

        _resolve_http_bearer(env, dotenv_path)

        assert set(env.keys()) == original_keys
        assert "GPU_GUARD_HTTP_BEARER" not in env
