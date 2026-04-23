"""Tests for `paramem.server.secret_store` — WP4 per-secret file layout."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from paramem.server.secret_store import (
    SecretStoreError,
    load_secrets_from_dir,
)


@pytest.fixture
def secrets_dir(tmp_path: Path) -> Path:
    """Return a fresh 0700 secrets directory under tmp_path."""
    d = tmp_path / "secrets"
    d.mkdir(mode=0o700)
    # mkdir on some filesystems ignores mode; force it.
    d.chmod(0o700)
    return d


def _write_secret(secrets_dir: Path, name: str, value: str, mode: int = 0o600) -> Path:
    p = secrets_dir / name
    p.write_text(value, encoding="utf-8")
    p.chmod(mode)
    return p


class TestLoadSecretsFromDir:
    def test_missing_directory_is_noop(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        assert load_secrets_from_dir(missing) == []

    def test_directory_with_wrong_mode_is_rejected(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_SECRET", raising=False)
        _write_secret(secrets_dir, "TEST_SECRET", "value")
        secrets_dir.chmod(0o755)  # too permissive

        with pytest.raises(SecretStoreError, match="does not match required"):
            load_secrets_from_dir(secrets_dir)

    def test_file_with_wrong_mode_is_rejected(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_SECRET", raising=False)
        _write_secret(secrets_dir, "TEST_SECRET", "value", mode=0o644)

        with pytest.raises(SecretStoreError, match="does not match required"):
            load_secrets_from_dir(secrets_dir)

    def test_loads_secret_into_environ(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_SECRET_A", raising=False)
        _write_secret(secrets_dir, "TEST_SECRET_A", "alpha")

        loaded = load_secrets_from_dir(secrets_dir)

        assert loaded == ["TEST_SECRET_A"]
        assert os.environ["TEST_SECRET_A"] == "alpha"

    def test_strips_trailing_newline_and_whitespace(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_SECRET_B", raising=False)
        _write_secret(secrets_dir, "TEST_SECRET_B", "beta  \n")

        load_secrets_from_dir(secrets_dir)

        assert os.environ["TEST_SECRET_B"] == "beta"

    def test_does_not_override_existing_env(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_SECRET_C", "shell-value")
        _write_secret(secrets_dir, "TEST_SECRET_C", "file-value")

        loaded = load_secrets_from_dir(secrets_dir)

        assert loaded == []  # nothing new set
        assert os.environ["TEST_SECRET_C"] == "shell-value"

    def test_skips_dotfiles(self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_SECRET_D", raising=False)
        # .README exists but has loose mode — should be skipped without error.
        readme = secrets_dir / ".README"
        readme.write_text("stash your notes here", encoding="utf-8")
        readme.chmod(0o644)

        _write_secret(secrets_dir, "TEST_SECRET_D", "delta")

        loaded = load_secrets_from_dir(secrets_dir)

        assert loaded == ["TEST_SECRET_D"]

    def test_skips_invalid_env_names(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Leading digit → invalid POSIX identifier.
        monkeypatch.delenv("1BAD", raising=False)
        _write_secret(secrets_dir, "1BAD", "value")

        loaded = load_secrets_from_dir(secrets_dir)

        # File is not loaded; env var is NOT set. (The code also logs a warning
        # to stderr; we don't assert on that because the project's logging
        # isn't configured to propagate into caplog uniformly.)
        assert loaded == []
        assert "1BAD" not in os.environ

    def test_skips_subdirectories(self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_SECRET_E", raising=False)
        (secrets_dir / "subdir").mkdir(mode=0o700)
        _write_secret(secrets_dir, "TEST_SECRET_E", "epsilon")

        loaded = load_secrets_from_dir(secrets_dir)

        assert loaded == ["TEST_SECRET_E"]

    def test_path_is_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "notadir"
        f.write_text("")

        with pytest.raises(SecretStoreError, match="expected a directory"):
            load_secrets_from_dir(f)

    def test_multiple_secrets_loaded_sorted(
        self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_SECRET_F", raising=False)
        monkeypatch.delenv("TEST_SECRET_G", raising=False)
        # Write G then F; loader should return them in sorted order.
        _write_secret(secrets_dir, "TEST_SECRET_G", "gamma")
        _write_secret(secrets_dir, "TEST_SECRET_F", "foxtrot")

        loaded = load_secrets_from_dir(secrets_dir)

        assert loaded == ["TEST_SECRET_F", "TEST_SECRET_G"]
        assert os.environ["TEST_SECRET_F"] == "foxtrot"
        assert os.environ["TEST_SECRET_G"] == "gamma"


class TestModeSemantics:
    """Sanity check that our constants match the documented invariants."""

    def test_file_mode_is_0600(self, secrets_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_SECRET_H", raising=False)
        p = _write_secret(secrets_dir, "TEST_SECRET_H", "hotel")

        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    def test_directory_mode_is_0700(self, secrets_dir: Path) -> None:
        assert stat.S_IMODE(secrets_dir.stat().st_mode) == 0o700
