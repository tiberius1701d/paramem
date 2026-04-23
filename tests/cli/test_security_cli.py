"""Tests for the security-track CLI commands.

Covers ``paramem generate-key``, ``paramem encrypt-infra``,
``paramem decrypt-infra``, and ``paramem dump``.  All tests are
pure-Python / filesystem-only — no REST, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    PMEM1_MAGIC,
    _clear_cipher_cache,
    is_pmem1_envelope,
    read_maybe_encrypted,
    write_infra_bytes,
)
from paramem.cli import decrypt_infra, dump, encrypt_infra, generate_key


def _make_key() -> str:
    return Fernet.generate_key().decode()


def _clean_env() -> None:
    os.environ.pop(MASTER_KEY_ENV_VAR, None)
    _clear_cipher_cache()


@pytest.fixture(autouse=True)
def _env_isolation():
    _clean_env()
    yield
    _clean_env()


# ---------------------------------------------------------------------------
# generate-key
# ---------------------------------------------------------------------------


class TestGenerateKey:
    def test_prints_key_line_to_stdout(self, capsys):
        rc = generate_key.run(argparse.Namespace())
        assert rc == 0
        out = capsys.readouterr()
        assert "PARAMEM_MASTER_KEY=" in out.out
        assert len(out.out.strip().splitlines()) == 1, "stdout must be exactly one line"
        # Warning banner goes to stderr.
        assert "WARNING" in out.err
        assert "unrecoverable" in out.err

    def test_generated_key_is_a_valid_fernet_key(self, capsys):
        generate_key.run(argparse.Namespace())
        out = capsys.readouterr().out.strip()
        key = out.split("=", 1)[1]
        # Must be accepted by Fernet without error.
        Fernet(key.encode())

    def test_two_calls_produce_different_keys(self, capsys):
        generate_key.run(argparse.Namespace())
        first = capsys.readouterr().out.strip().split("=", 1)[1]
        generate_key.run(argparse.Namespace())
        second = capsys.readouterr().out.strip().split("=", 1)[1]
        assert first != second


# ---------------------------------------------------------------------------
# encrypt-infra
# ---------------------------------------------------------------------------


class TestEncryptInfra:
    def _seed_plaintext_files(self, data_dir: Path) -> list[Path]:
        """Write a couple of plaintext infra files under *data_dir* that
        match paths in ``infra_paths``."""
        (data_dir / "registry").mkdir(parents=True, exist_ok=True)
        (data_dir / "adapters").mkdir(parents=True, exist_ok=True)
        seeded = []
        reg = data_dir / "registry.json"
        reg.write_bytes(json.dumps({"graph1": "abc"}).encode("utf-8"))
        seeded.append(reg)
        speakers = data_dir / "speaker_profiles.json"
        speakers.write_bytes(json.dumps({"speakers": {}, "version": 5}).encode("utf-8"))
        seeded.append(speakers)
        queue = data_dir / "adapters" / "post_session_queue.json"
        queue.write_bytes(json.dumps([]).encode("utf-8"))
        seeded.append(queue)
        return seeded

    def test_refuses_without_key(self, tmp_path, capsys):
        data_dir = tmp_path / "data"
        self._seed_plaintext_files(data_dir)
        rc = encrypt_infra.run(
            argparse.Namespace(data_dir=str(data_dir), config="configs/server.yaml")
        )
        assert rc == 1
        assert "PARAMEM_MASTER_KEY must be set" in capsys.readouterr().err

    def test_encrypts_plaintext_files(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        seeded = self._seed_plaintext_files(data_dir)

        rc = encrypt_infra.run(
            argparse.Namespace(data_dir=str(data_dir), config="configs/server.yaml")
        )
        assert rc == 0

        for path in seeded:
            assert is_pmem1_envelope(path)
            assert path.read_bytes().startswith(PMEM1_MAGIC)

        out = capsys.readouterr().out
        assert "3 converted" in out

    def test_idempotent_on_already_encrypted(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        self._seed_plaintext_files(data_dir)
        encrypt_infra.run(argparse.Namespace(data_dir=str(data_dir), config=""))
        capsys.readouterr()  # drain first run

        rc = encrypt_infra.run(argparse.Namespace(data_dir=str(data_dir), config=""))
        assert rc == 0
        out = capsys.readouterr().out
        assert "0 converted" in out
        assert "already encrypted" in out

    def test_missing_data_dir_returns_error(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        missing = tmp_path / "does-not-exist"
        rc = encrypt_infra.run(
            argparse.Namespace(data_dir=str(missing), config="configs/server.yaml")
        )
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# decrypt-infra
# ---------------------------------------------------------------------------


class TestDecryptInfra:
    def _seed_ciphertext_files(self, data_dir: Path) -> list[Path]:
        """Seed with files written through write_infra_bytes while the key
        is set, then return them with the key preserved so decrypt-infra can
        unwrap them."""
        (data_dir / "registry").mkdir(parents=True, exist_ok=True)
        (data_dir / "adapters").mkdir(parents=True, exist_ok=True)
        seeded = []
        for rel, content in [
            (Path("registry.json"), b'{"graph1":"abc"}'),
            (Path("speaker_profiles.json"), b'{"speakers":{},"version":5}'),
            (Path("adapters/post_session_queue.json"), b"[]"),
        ]:
            p = data_dir / rel
            write_infra_bytes(p, content)
            seeded.append(p)
        return seeded

    def test_refuses_without_flag(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        self._seed_ciphertext_files(data_dir)
        rc = decrypt_infra.run(
            argparse.Namespace(i_accept_plaintext=False, data_dir=str(data_dir), config="")
        )
        assert rc == 1
        assert "--i-accept-plaintext is required" in capsys.readouterr().err

    def test_refuses_without_key(self, tmp_path, capsys):
        # Seed ciphertext with a key, then drop the key to simulate the
        # unrecoverable case.
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        self._seed_ciphertext_files(data_dir)
        _clean_env()

        rc = decrypt_infra.run(
            argparse.Namespace(i_accept_plaintext=True, data_dir=str(data_dir), config="")
        )
        assert rc == 1
        assert "must be set to decrypt" in capsys.readouterr().err

    def test_decrypts_ciphertext_files(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        seeded = self._seed_ciphertext_files(data_dir)

        rc = decrypt_infra.run(
            argparse.Namespace(i_accept_plaintext=True, data_dir=str(data_dir), config="")
        )
        assert rc == 0

        for path in seeded:
            assert not is_pmem1_envelope(path)
        # Reminder about removing the env var appears on stdout.
        assert "remove PARAMEM_MASTER_KEY" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


class TestDump:
    def test_plaintext_passthrough(self, tmp_path, capsysbinary):
        p = tmp_path / "plain.json"
        p.write_bytes(b'{"ok":true}')
        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 0
        assert capsysbinary.readouterr().out == b'{"ok":true}'

    def test_encrypted_decrypted_to_stdout(self, tmp_path, capsysbinary):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        p = tmp_path / "encrypted.json"
        write_infra_bytes(p, b'{"secret":"value"}')

        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 0
        assert capsysbinary.readouterr().out == b'{"secret":"value"}'

    def test_missing_path_returns_error(self, tmp_path, capsys):
        rc = dump.run(argparse.Namespace(path=str(tmp_path / "nope.json")))
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_encrypted_without_key_returns_error(self, tmp_path, capsys):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        p = tmp_path / "encrypted.json"
        write_infra_bytes(p, b"payload")
        _clean_env()

        rc = dump.run(argparse.Namespace(path=str(p)))
        assert rc == 1
        assert MASTER_KEY_ENV_VAR in capsys.readouterr().err


# ---------------------------------------------------------------------------
# End-to-end: encrypt → decrypt roundtrip
# ---------------------------------------------------------------------------


class TestEncryptDecryptRoundtrip:
    def test_encrypt_then_decrypt_returns_original(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        original = json.dumps({"graph1": "abc", "graph2": "def"}, indent=2).encode("utf-8")
        reg = data_dir / "registry.json"
        reg.write_bytes(original)

        encrypt_infra.run(argparse.Namespace(data_dir=str(data_dir), config=""))
        assert is_pmem1_envelope(reg)
        # Content still recoverable via the envelope helper.
        assert read_maybe_encrypted(reg) == original

        decrypt_infra.run(
            argparse.Namespace(i_accept_plaintext=True, data_dir=str(data_dir), config="")
        )
        assert not is_pmem1_envelope(reg)
        assert reg.read_bytes() == original
