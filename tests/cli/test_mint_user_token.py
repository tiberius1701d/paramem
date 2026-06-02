"""Tests for ``paramem mint-user-token``.

Covers:
- Happy path: exit 0, token store created with exactly one entry, token
  round-trips via UserTokenStore.lookup().
- QR output: something QR-like (or the text fallback) reaches stdout.
- Deep-link: QR/printed output encodes /app#token=<t>&url=<encoded-server-url>.
- Text fallback is printed (speaker_id, server_url, token, deeplink).
- CLI registration: subcommand appears in the top-level parser.
- Config not found: exit 1 with clear error.
- PNG write: QR saved to specified path when --png is given.
- No --server-url: WARNING emitted, no crash, token still printed.
- No daily key needed: store written in plaintext (Security OFF) — no
  PARAMEM_DAILY_PASSPHRASE required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from paramem.backup.key_store import DAILY_PASSPHRASE_ENV_VAR, _clear_daily_identity_cache
from paramem.cli import mint_user_token
from paramem.server.user_tokens import UserTokenStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no daily key is loadable (Security OFF) for all tests in this
    module.  The store must be written in plaintext so no passphrase is
    required — mint-user-token must not depend on the daily encryption key."""
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    # Point the default daily-key path at a nonexistent file so
    # daily_identity_loadable() returns False.
    monkeypatch.setattr(
        "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
        Path("/nonexistent/daily_key.age"),
    )
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


def _make_config_yaml(tmp_path: Path, data_dir: Path) -> Path:
    """Write a minimal server.yaml pointing at *data_dir*.

    Mirrors the helper used in test_encrypt_infra_cli.py.
    """
    cfg_path = tmp_path / "server.yaml"
    cfg_path.write_text(
        f"model: mistral\npaths:\n  data: {data_dir}\n  simulate: {data_dir}/simulate\n",
        encoding="utf-8",
    )
    return cfg_path


def _make_args(
    config: Path,
    speaker_id: str | None = "Speaker0",
    *,
    label: str = "",
    server_url: str = "",
    png: str | None = None,
    scope: str = "chat",
    unattributed: bool = False,
    force_admin: bool = False,
) -> argparse.Namespace:
    """Build a Namespace matching the parser's output for mint-user-token."""
    return argparse.Namespace(
        speaker_id=speaker_id,
        label=label,
        server_url=server_url,
        config=str(config),
        png=png,
        scope=scope,
        unattributed=unattributed,
        force_admin=force_admin,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_exits_zero_and_creates_store(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        """run() exits 0 and creates user_tokens.json under the data dir."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, "Speaker0"))

        assert rc == 0
        store_path = data_dir / "user_tokens.json"
        assert store_path.exists(), "user_tokens.json must be created"

    def test_store_contains_exactly_one_entry(self, tmp_path: Path) -> None:
        """The token store contains exactly one entry after a single mint."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        mint_user_token.run(_make_args(cfg, "Speaker0"))

        store = UserTokenStore(data_dir / "user_tokens.json")
        entries = store.list()
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"

    def test_token_round_trips_via_lookup(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        """The printed token looks up back to the expected speaker_id."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, "Speaker0"))
        assert rc == 0

        # Extract token from the "token      : <value>" line in stdout.
        out = capsys.readouterr().out
        token_line = next(ln for ln in out.splitlines() if ln.startswith("token"))
        token = token_line.split(":", 1)[1].strip()
        assert token, "Printed token must be non-empty"

        store = UserTokenStore(data_dir / "user_tokens.json")
        resolved = store.lookup(token)
        assert resolved == "Speaker0", f"lookup() must return 'Speaker0', got {resolved!r}"

    def test_store_entry_has_expected_speaker_and_label(self, tmp_path: Path) -> None:
        """The store entry has the correct speaker_id and label."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        mint_user_token.run(_make_args(cfg, "Speaker1", label="My Phone"))

        store = UserTokenStore(data_dir / "user_tokens.json")
        entries = store.list()
        assert entries[0]["speaker_id"] == "Speaker1"
        assert entries[0]["label"] == "My Phone"
        assert entries[0]["revoked"] is False

    def test_store_hash_not_plaintext_token(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        """The token store file must not contain the plaintext token."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        mint_user_token.run(_make_args(cfg, "Speaker0"))
        out = capsys.readouterr().out

        token_line = next(ln for ln in out.splitlines() if ln.startswith("token"))
        token = token_line.split(":", 1)[1].strip()

        store_bytes = (data_dir / "user_tokens.json").read_text(encoding="utf-8")
        assert token not in store_bytes, "Plaintext token must not appear in the store file"


# ---------------------------------------------------------------------------
# QR output and text fallback
# ---------------------------------------------------------------------------


class TestQrAndTextOutput:
    def test_stdout_contains_qr_or_text_fallback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """stdout contains either ANSI QR characters or the text token fallback."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        mint_user_token.run(_make_args(cfg, "Speaker0"))
        out = capsys.readouterr().out

        # Either ANSI escape codes (QR terminal output) or the text fallback are
        # present — both indicate the QR step ran without error.
        has_ansi = "\x1b[" in out
        has_token_line = any(ln.startswith("token") for ln in out.splitlines())
        assert has_ansi or has_token_line, f"Expected QR or text token line in stdout, got: {out!r}"

    def test_text_fallback_lines_present(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """speaker_id, server_url, and token lines all appear in stdout."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        mint_user_token.run(_make_args(cfg, "Speaker2", server_url="https://example.ts.net"))
        out = capsys.readouterr().out
        lines = out.splitlines()

        assert any("Speaker2" in ln for ln in lines), "speaker_id must appear in stdout"
        assert any("https://example.ts.net" in ln for ln in lines), (
            "server_url must appear in stdout"
        )
        assert any(ln.startswith("token") for ln in lines), "token line must appear in stdout"

    def test_deeplink_has_token_and_url_params(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """The printed deep-link contains /app#token=<t>&url=<encoded-server-url>."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        mint_user_token.run(_make_args(cfg, "Speaker0", server_url="https://example.ts.net"))
        out = capsys.readouterr().out

        # Token from text fallback line.
        token_line = next(ln for ln in out.splitlines() if ln.startswith("token"))
        token = token_line.split(":", 1)[1].strip()
        assert token, "Printed token must be non-empty"

        # Deep-link from the deeplink line.
        deeplink_line = next(ln for ln in out.splitlines() if ln.startswith("deeplink"))
        deeplink = deeplink_line.split(":", 1)[1].strip()

        assert "/app#token=" in deeplink, "Deep-link must contain /app#token="
        assert token in deeplink, "Deep-link must contain the minted token"
        assert "url=" in deeplink, "Deep-link must contain url= parameter"
        assert "https%3A%2F%2Fexample.ts.net" in deeplink, (
            "server_url must be percent-encoded in the deep-link"
        )


# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------


class TestPngOutput:
    def test_png_created_when_path_given(self, tmp_path: Path) -> None:
        """--png writes a PNG file at the specified path when --server-url is also given."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        png_path = tmp_path / "qr.png"

        rc = mint_user_token.run(
            _make_args(cfg, "Speaker0", server_url="https://example.ts.net", png=str(png_path))
        )

        assert rc == 0
        assert png_path.exists(), "PNG file must be created when --png is given"
        # Minimal PNG magic-bytes check.
        header = png_path.read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", "Written file must be a valid PNG"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_missing_config_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        """Exit 1 with a clear error when config file is not found."""
        missing_cfg = tmp_path / "nonexistent.yaml"

        rc = mint_user_token.run(_make_args(missing_cfg, "Speaker0"))

        assert rc == 1
        err = capsys.readouterr().err
        assert "config file not found" in err or "not found" in err

    def test_no_server_url_emits_warning_no_crash(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Omitting --server-url emits a WARNING to stderr, exits 0, and still prints the token.

        No QR is emitted and no deep-link line appears in stdout — the warning
        explains that --server-url is required for a scannable QR.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, "Speaker0", server_url=""))
        assert rc == 0

        captured = capsys.readouterr()
        assert "WARNING" in captured.err, (
            "A WARNING must appear in stderr when --server-url is omitted"
        )
        assert "--server-url" in captured.err or "server-url" in captured.err, (
            "The warning must mention --server-url"
        )

        # Token is still printed as a text fallback.
        lines = captured.out.splitlines()
        assert any(ln.startswith("token") for ln in lines), (
            "Plaintext token must still be printed even without --server-url"
        )
        # No deep-link line when server_url is empty.
        assert not any(ln.startswith("deeplink") for ln in lines), (
            "No deeplink line must appear when --server-url is omitted"
        )

    def test_no_server_url_with_png_emits_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--png combined with no --server-url emits a warning that --png is ignored."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)
        png_path = tmp_path / "qr.png"

        rc = mint_user_token.run(_make_args(cfg, "Speaker0", server_url="", png=str(png_path)))
        assert rc == 0
        assert not png_path.exists(), "PNG must NOT be written when --server-url is omitted"

        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "--png" in err or "png" in err.lower(), (
            "The warning must mention that --png is ignored"
        )


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class TestCliRegistration:
    def test_subcommand_registered_in_parser(self) -> None:
        """mint-user-token must appear as a valid subcommand in the top-level parser."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["mint-user-token", "Speaker0"])
        assert args.command == "mint-user-token"
        assert args.speaker_id == "Speaker0"

    def test_all_flags_accepted(self) -> None:
        """All declared flags are accepted by the parser without error."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "mint-user-token",
                "Speaker0",
                "--label",
                "My Tablet",
                "--server-url",
                "https://example.ts.net",
                "--config",
                "/tmp/cfg.yaml",
                "--png",
                "/tmp/qr.png",
            ]
        )
        assert args.speaker_id == "Speaker0"
        assert args.label == "My Tablet"
        assert args.server_url == "https://example.ts.net"
        assert args.config == "/tmp/cfg.yaml"
        assert args.png == "/tmp/qr.png"

    def test_help_exits_zero(self) -> None:
        """mint-user-token --help exits 0."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["mint-user-token", "--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Scope and unattributed
# ---------------------------------------------------------------------------


class TestScopeAndUnattributed:
    def test_scope_admin_stored_in_record(self, tmp_path: Path) -> None:
        """--scope admin → stored record has scope == 'admin'."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, "Speaker0", scope="admin"))
        assert rc == 0

        store = UserTokenStore(data_dir / "user_tokens.json")
        entries = store.list()
        assert entries[0]["scope"] == "admin"

    def test_default_scope_is_chat(self, tmp_path: Path) -> None:
        """Default scope (no --scope) → stored record has scope == 'chat'."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, "Speaker0"))
        assert rc == 0

        store = UserTokenStore(data_dir / "user_tokens.json")
        entries = store.list()
        assert entries[0]["scope"] == "chat"

    def test_unattributed_mints_null_speaker_chat_token(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--unattributed → speaker_id=None, scope='chat' in store; prints <unattributed>."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, speaker_id=None, scope="chat", unattributed=True))
        assert rc == 0

        store = UserTokenStore(data_dir / "user_tokens.json")
        entries = store.list()
        assert len(entries) == 1
        assert entries[0]["speaker_id"] is None
        assert entries[0]["scope"] == "chat"

        out = capsys.readouterr().out
        assert "<unattributed>" in out, "Text fallback must print <unattributed> for no speaker_id"

    def test_unattributed_with_speaker_id_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--unattributed combined with a positional SPEAKER_ID → exit 1."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, speaker_id="Speaker0", unattributed=True))
        assert rc == 1
        err = capsys.readouterr().err
        assert "cannot be combined" in err or "SPEAKER_ID" in err

    def test_no_speaker_id_no_unattributed_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Neither positional SPEAKER_ID nor --unattributed → exit 1."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, speaker_id=None, unattributed=False))
        assert rc == 1
        err = capsys.readouterr().err
        assert "SPEAKER_ID" in err or "required" in err

    def test_scope_admin_unattributed_refused_without_force_admin(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--scope admin --unattributed without --force-admin → exit 1 with warning."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(
            _make_args(cfg, speaker_id=None, scope="admin", unattributed=True, force_admin=False)
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "force-admin" in err or "--force-admin" in err

    def test_scope_admin_unattributed_with_force_admin_succeeds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--scope admin --unattributed --force-admin → mint succeeds + warning to stderr."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(
            _make_args(cfg, speaker_id=None, scope="admin", unattributed=True, force_admin=True)
        )
        assert rc == 0
        err = capsys.readouterr().err
        # A warning must be printed.
        assert "WARNING" in err or "unattributed admin" in err.lower()

        store = UserTokenStore(data_dir / "user_tokens.json")
        entries = store.list()
        assert entries[0]["speaker_id"] is None
        assert entries[0]["scope"] == "admin"

    def test_scope_line_in_text_fallback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """The text fallback includes a 'scope' line."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        mint_user_token.run(_make_args(cfg, "Speaker0", scope="admin"))
        out = capsys.readouterr().out
        assert any("scope" in ln for ln in out.splitlines()), (
            "Text fallback must include a 'scope' line"
        )


# ---------------------------------------------------------------------------
# Parser: new flags accepted by argparse
# ---------------------------------------------------------------------------


class TestNonTtyWarning:
    """Verify that a WARNING is printed to stderr when stdout is not a TTY."""

    def test_warning_emitted_to_stderr_when_not_a_tty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """run() emits a WARNING to stderr when sys.stdout is not a TTY.

        Under pytest, sys.stdout.isatty() is always False (capsys pipes
        stdout), so the warning fires unconditionally in tests — the test
        verifies the warning text reaches stderr.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config_yaml(tmp_path, data_dir)

        rc = mint_user_token.run(_make_args(cfg, "Speaker0"))
        assert rc == 0

        err = capsys.readouterr().err
        assert "WARNING" in err, "A WARNING must be printed to stderr when stdout is not a TTY"
        assert "plaintext" in err.lower(), (
            "The warning must mention that the plaintext token is being written"
        )


class TestNewParserFlags:
    def test_scope_flag_accepted(self) -> None:
        """--scope is accepted by the parser."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["mint-user-token", "Speaker0", "--scope", "admin"])
        assert args.scope == "admin"

    def test_unattributed_flag_accepted(self) -> None:
        """--unattributed is accepted as a flag (no positional required by parser)."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["mint-user-token", "--unattributed"])
        assert args.unattributed is True
        assert args.speaker_id is None

    def test_force_admin_flag_accepted(self) -> None:
        """--force-admin is accepted as a flag."""
        from paramem.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            ["mint-user-token", "--unattributed", "--scope", "admin", "--force-admin"]
        )
        assert args.force_admin is True
