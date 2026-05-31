"""Tests for ``paramem revoke-user-token``.

Covers:
- ``--list``: exit 0, prints token table, empty store handled gracefully.
- ``--speaker``: revokes all tokens for the speaker; store reflects revoked=True;
  no-match returns exit 1.
- ``--label``: revokes tokens by exact label; store reflects revoked=True;
  no-match returns exit 1.
- ``--yes`` skips the confirmation prompt.
- On-disk store reflects revoked=True after revocation (Security OFF).
- When daily key is loaded (Security ON), on-disk store stays an age envelope
  after revocation.
- Missing config returns exit 1 with a clear error message.
- CLI registration: subcommand appears in the top-level parser; all flags accepted;
  ``--help`` exits 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.cli import revoke_user_token
from paramem.server.user_tokens import UserTokenStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point env + module default at it.

    Mirrors the helper in ``tests/server/test_user_tokens.py``.
    """
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no daily key is loadable (Security OFF) for tests that do not
    call _setup_daily explicitly.  Mirrors test_mint_user_token.py."""
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
        Path("/nonexistent/daily_key.age"),
    )
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


def _make_config_yaml(tmp_path: Path, data_dir: Path) -> Path:
    """Write a minimal server.yaml pointing at *data_dir*.

    Mirrors the helper used in test_mint_user_token.py.
    """
    cfg_path = tmp_path / "server.yaml"
    cfg_path.write_text(
        f"model: mistral\npaths:\n  data: {data_dir}\n  simulate: {data_dir}/simulate\n",
        encoding="utf-8",
    )
    return cfg_path


def _make_args(
    config: Path,
    *,
    speaker: str | None = None,
    label: str | None = None,
    list_tokens: bool = False,
    yes: bool = True,
) -> argparse.Namespace:
    """Build a Namespace matching the parser's output for revoke-user-token.

    ``yes=True`` by default so tests do not need to mock stdin.
    """
    return argparse.Namespace(
        speaker=speaker,
        label=label,
        list_tokens=list_tokens,
        config=str(config),
        yes=yes,
    )


def _mint_token(data_dir: Path, speaker_id: str, label: str = "") -> str:
    """Mint a token directly into the store at *data_dir* and return the plaintext token."""
    store = UserTokenStore(data_dir / "user_tokens.json")
    return store.mint(speaker_id, label)


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------


class TestListTokens:
    def test_list_exits_zero_and_prints_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--list prints the token table and exits 0."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "phone")

        rc = revoke_user_token.run(_make_args(cfg, list_tokens=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "Speaker0" in out
        assert "phone" in out

    def test_list_empty_store_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--list on an empty store prints a friendly message and exits 0."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = revoke_user_token.run(_make_args(cfg, list_tokens=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "No tokens" in out

    def test_list_does_not_modify_store(self, tmp_path: Path) -> None:
        """--list must not write to the store."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "tablet")
        store_path = data_dir / "user_tokens.json"
        mtime_before = store_path.stat().st_mtime

        revoke_user_token.run(_make_args(cfg, list_tokens=True))

        assert store_path.stat().st_mtime == mtime_before, (
            "--list must not modify the on-disk store"
        )


# ---------------------------------------------------------------------------
# --speaker
# ---------------------------------------------------------------------------


class TestRevokeBySpeaker:
    def test_revokes_all_tokens_for_speaker(self, tmp_path: Path) -> None:
        """--speaker revokes every active token for that speaker."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        t1 = _mint_token(data_dir, "Speaker0", "phone")
        t2 = _mint_token(data_dir, "Speaker0", "tablet")
        t_other = _mint_token(data_dir, "Speaker1", "desktop")

        rc = revoke_user_token.run(_make_args(cfg, speaker="Speaker0"))

        assert rc == 0
        store = UserTokenStore(data_dir / "user_tokens.json")
        assert store.lookup(t1) is None, "t1 must be revoked"
        assert store.lookup(t2) is None, "t2 must be revoked"
        assert store.lookup(t_other) == "Speaker1", "other speaker's token must be untouched"

    def test_on_disk_store_reflects_revoked_true(self, tmp_path: Path) -> None:
        """After --speaker revocation, the on-disk JSON has revoked=True entries."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "device")

        revoke_user_token.run(_make_args(cfg, speaker="Speaker0"))

        raw = json.loads((data_dir / "user_tokens.json").read_text(encoding="utf-8"))
        entries = list(raw["tokens"].values())
        assert all(e["revoked"] is True for e in entries if e["speaker_id"] == "Speaker0"), (
            "on-disk entries for Speaker0 must have revoked=True"
        )

    def test_no_match_returns_exit_1(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """--speaker with an unknown speaker_id exits 1 with a clear message."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "device")

        rc = revoke_user_token.run(_make_args(cfg, speaker="Speaker99"))

        assert rc == 1
        err = capsys.readouterr().err
        assert "Speaker99" in err or "No active tokens" in err

    def test_prints_restart_note(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Successful --speaker revocation prints a restart-required note."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "phone")

        revoke_user_token.run(_make_args(cfg, speaker="Speaker0"))

        out = capsys.readouterr().out
        assert "restart" in out.lower(), "output must mention that a server restart is needed"

    def test_exits_zero_on_success(self, tmp_path: Path) -> None:
        """--speaker with a matching speaker exits 0."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "device")

        rc = revoke_user_token.run(_make_args(cfg, speaker="Speaker0"))

        assert rc == 0


# ---------------------------------------------------------------------------
# --label
# ---------------------------------------------------------------------------


class TestRevokeByLabel:
    def test_revokes_matching_label_only(self, tmp_path: Path) -> None:
        """--label revokes only tokens whose label matches exactly."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        t_phone = _mint_token(data_dir, "Speaker0", "phone")
        t_tablet = _mint_token(data_dir, "Speaker0", "tablet")
        t_other = _mint_token(data_dir, "Speaker1", "phone")

        rc = revoke_user_token.run(_make_args(cfg, label="phone"))

        assert rc == 0
        store = UserTokenStore(data_dir / "user_tokens.json")
        assert store.lookup(t_phone) is None, "Speaker0 phone must be revoked"
        assert store.lookup(t_other) is None, "Speaker1 phone must also be revoked"
        assert store.lookup(t_tablet) == "Speaker0", "tablet must be untouched"

    def test_on_disk_store_reflects_revoked_true(self, tmp_path: Path) -> None:
        """After --label revocation, the on-disk JSON has revoked=True for matching entries."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "phone")

        revoke_user_token.run(_make_args(cfg, label="phone"))

        raw = json.loads((data_dir / "user_tokens.json").read_text(encoding="utf-8"))
        entries = list(raw["tokens"].values())
        assert all(e["revoked"] is True for e in entries if e["label"] == "phone"), (
            "on-disk entries with label='phone' must have revoked=True"
        )

    def test_no_match_returns_exit_1(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """--label with no matching label exits 1 with a clear message."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "tablet")

        rc = revoke_user_token.run(_make_args(cfg, label="no-such-label"))

        assert rc == 1
        err = capsys.readouterr().err
        assert "no-such-label" in err or "No active tokens" in err

    def test_exits_zero_on_success(self, tmp_path: Path) -> None:
        """--label with a matching label exits 0."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        _mint_token(data_dir, "Speaker0", "phone")

        rc = revoke_user_token.run(_make_args(cfg, label="phone"))

        assert rc == 0


# ---------------------------------------------------------------------------
# Security ON: age-encrypted store stays encrypted after revocation
# ---------------------------------------------------------------------------


class TestEncryptedStore:
    def test_store_stays_age_envelope_after_speaker_revoke(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When daily key is loaded, the store is still an age envelope after revocation."""
        from paramem.backup.age_envelope import AGE_MAGIC

        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        store = UserTokenStore(data_dir / "user_tokens.json")
        store.mint("Speaker0", "device")

        revoke_user_token.run(_make_args(cfg, speaker="Speaker0"))

        disk_bytes = (data_dir / "user_tokens.json").read_bytes()
        assert disk_bytes.startswith(AGE_MAGIC), (
            "store must remain an age envelope after revocation when Security ON"
        )

    def test_store_stays_age_envelope_after_label_revoke(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When daily key is loaded, the store is still an age envelope after label revocation."""
        from paramem.backup.age_envelope import AGE_MAGIC

        _setup_daily(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        store = UserTokenStore(data_dir / "user_tokens.json")
        store.mint("Speaker0", "phone")

        revoke_user_token.run(_make_args(cfg, label="phone"))

        disk_bytes = (data_dir / "user_tokens.json").read_bytes()
        assert disk_bytes.startswith(AGE_MAGIC), (
            "store must remain an age envelope after label revocation when Security ON"
        )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_missing_config_returns_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Exit 1 with a clear error when config file is not found."""
        missing_cfg = tmp_path / "nonexistent.yaml"

        rc = revoke_user_token.run(_make_args(missing_cfg, speaker="Speaker0"))

        assert rc == 1
        err = capsys.readouterr().err
        assert "config file not found" in err or "not found" in err

    def test_abort_on_no_confirmation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --yes, a 'n' answer at the prompt aborts and returns exit 1."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        t = _mint_token(data_dir, "Speaker0", "phone")

        # Simulate user entering 'n'.
        monkeypatch.setattr("builtins.input", lambda _: "n")
        args = _make_args(cfg, speaker="Speaker0", yes=False)

        rc = revoke_user_token.run(args)

        assert rc == 1
        # Token must still be active.
        store = UserTokenStore(data_dir / "user_tokens.json")
        assert store.lookup(t) == "Speaker0", "token must not be revoked after abort"

    def test_confirm_yes_proceeds_without_prompt(self, tmp_path: Path) -> None:
        """--yes bypasses the confirmation prompt and revokes successfully."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        t = _mint_token(data_dir, "Speaker0", "phone")
        args = _make_args(cfg, speaker="Speaker0", yes=True)

        rc = revoke_user_token.run(args)

        assert rc == 0
        store = UserTokenStore(data_dir / "user_tokens.json")
        assert store.lookup(t) is None, "token must be revoked when --yes is passed"


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class TestCliRegistration:
    def test_subcommand_registered_in_parser(self) -> None:
        """revoke-user-token must appear as a valid subcommand in the top-level parser."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["revoke-user-token", "--speaker", "Speaker0"])
        assert args.command == "revoke-user-token"
        assert args.speaker == "Speaker0"

    def test_label_flag_accepted(self) -> None:
        """--label flag is accepted by the parser."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["revoke-user-token", "--label", "phone"])
        assert args.label == "phone"

    def test_list_flag_accepted(self) -> None:
        """--list flag is accepted by the parser."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["revoke-user-token", "--list"])
        assert args.list_tokens is True

    def test_yes_flag_accepted(self) -> None:
        """--yes flag is accepted by the parser."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["revoke-user-token", "--speaker", "Speaker0", "--yes"])
        assert args.yes is True

    def test_config_flag_accepted(self) -> None:
        """--config flag is accepted by the parser."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["revoke-user-token", "--list", "--config", "/tmp/cfg.yaml"])
        assert args.config == "/tmp/cfg.yaml"

    def test_help_exits_zero(self) -> None:
        """revoke-user-token --help exits 0."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["revoke-user-token", "--help"])
        assert exc_info.value.code == 0

    def test_speaker_and_label_are_mutually_exclusive(self) -> None:
        """--speaker and --label cannot be combined."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["revoke-user-token", "--speaker", "Speaker0", "--label", "phone"])
