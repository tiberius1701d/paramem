"""Tests for the HF checkpoint shard encryption helpers.

Covers:

1. ``encrypt_checkpoint_dir`` — wraps plaintext files in place; idempotent;
   no-op when Security is OFF.
2. ``materialize_checkpoint_to_shm`` — round-trip decrypt/copy into a
   tempdir; handles mixed-state directories; handles subdirectories.
3. Integration: encrypt → materialize yields byte-identical content.
4. ``/dev/shm`` fallback — behaviour when the mount is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from paramem.backup.checkpoint_shard import (
    encrypt_checkpoint_dir,
    materialize_checkpoint_to_shm,
)
from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    _clear_cipher_cache,
    is_pmem1_envelope,
)


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


def _seed_checkpoint(root: Path) -> dict[str, bytes]:
    """Write a synthetic checkpoint-<N> directory and return the expected bytes.

    Emulates an HF PEFT checkpoint: a mix of binary (.safetensors, .pt) and
    text (.json) files plus a README.
    """
    root.mkdir(parents=True, exist_ok=True)
    contents = {
        "adapter_model.safetensors": b"\x00\x01safetensors-stub" + os.urandom(512),
        "optimizer.pt": b"\x80\x02optimizer-stub" + os.urandom(1024),
        "scheduler.pt": b"\x80\x02scheduler-stub" + os.urandom(128),
        "trainer_state.json": b'{"global_step": 42, "epoch": 1}',
        "training_args.bin": b"\x80\x02training-args-stub",
        "rng_state.pth": b"\x80\x02rng-stub",
        "adapter_config.json": b'{"r": 8, "alpha": 16}',
        "README.md": b"# Checkpoint\n",
    }
    for name, body in contents.items():
        (root / name).write_bytes(body)
    return contents


class TestEncryptCheckpointDir:
    def test_encrypt_wraps_every_file_when_key_set(self, tmp_path: Path) -> None:
        """Master key set → every file gains PMEM1 magic; content round-trips."""
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            n = encrypt_checkpoint_dir(ckpt)

        assert n == len(contents)
        for name in contents:
            assert is_pmem1_envelope(ckpt / name), f"{name} should be PMEM1-wrapped"

    def test_encrypt_noop_when_security_off(self, tmp_path: Path) -> None:
        """No master key → no file is modified; return value is 0."""
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        n = encrypt_checkpoint_dir(ckpt)

        assert n == 0
        for name, body in contents.items():
            assert (ckpt / name).read_bytes() == body
            assert not is_pmem1_envelope(ckpt / name)

    def test_encrypt_idempotent(self, tmp_path: Path) -> None:
        """Second call after full encryption encrypts nothing new."""
        ckpt = tmp_path / "checkpoint-42"
        _seed_checkpoint(ckpt)

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            n1 = encrypt_checkpoint_dir(ckpt)
            n2 = encrypt_checkpoint_dir(ckpt)

        assert n1 > 0
        assert n2 == 0

    def test_encrypt_handles_subdirectories(self, tmp_path: Path) -> None:
        """Files in nested subdirs are encrypted recursively."""
        ckpt = tmp_path / "checkpoint-42"
        _seed_checkpoint(ckpt)
        subdir = ckpt / "tokenizer"
        subdir.mkdir()
        (subdir / "tokenizer.json").write_bytes(b'{"vocab_size": 32000}')

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            encrypt_checkpoint_dir(ckpt)

        assert is_pmem1_envelope(subdir / "tokenizer.json")

    def test_encrypt_skips_already_wrapped_files(self, tmp_path: Path) -> None:
        """Mixed state (some wrapped, some plaintext) encrypts only the plaintext."""
        ckpt = tmp_path / "checkpoint-42"
        _seed_checkpoint(ckpt)

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            # Encrypt just one file, then call encrypt_checkpoint_dir.
            from paramem.backup.encryption import write_infra_bytes

            write_infra_bytes(
                ckpt / "adapter_model.safetensors",
                (ckpt / "adapter_model.safetensors").read_bytes(),
            )
            # Re-read after the write since the filesystem state changed.
            # One file is now wrapped, the rest remain plaintext.
            n = encrypt_checkpoint_dir(ckpt)

        # 8 total files seeded, 1 already wrapped → 7 newly encrypted.
        assert n == 7


class TestMaterializeCheckpointToShm:
    def test_materialize_decrypts_all_files(self, tmp_path: Path) -> None:
        """Fully encrypted source → tempdir carries decrypted bytes."""
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            encrypt_checkpoint_dir(ckpt)
            shm = materialize_checkpoint_to_shm(ckpt)

        try:
            for name, body in contents.items():
                assert (shm / name).read_bytes() == body
                assert not is_pmem1_envelope(shm / name)
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)

    def test_materialize_passes_through_plaintext(self, tmp_path: Path) -> None:
        """Unencrypted source → tempdir is a byte-for-byte copy."""
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        shm = materialize_checkpoint_to_shm(ckpt)

        try:
            for name, body in contents.items():
                assert (shm / name).read_bytes() == body
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)

    def test_materialize_handles_mixed_state(self, tmp_path: Path) -> None:
        """Half-wrapped directory → each file decoded via the right branch."""
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        # Wrap half the files; leave the rest plaintext.
        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            from paramem.backup.encryption import write_infra_bytes

            to_wrap = list(contents)[:4]
            for name in to_wrap:
                write_infra_bytes(ckpt / name, (ckpt / name).read_bytes())

            shm = materialize_checkpoint_to_shm(ckpt)

        try:
            for name, body in contents.items():
                assert (shm / name).read_bytes() == body
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)

    def test_materialize_preserves_subdirectory_layout(self, tmp_path: Path) -> None:
        """Nested subdirs survive materialization at the same relative path."""
        ckpt = tmp_path / "checkpoint-42"
        _seed_checkpoint(ckpt)
        subdir = ckpt / "tokenizer"
        subdir.mkdir()
        sub_body = b'{"vocab_size": 32000}'
        (subdir / "tokenizer.json").write_bytes(sub_body)

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            encrypt_checkpoint_dir(ckpt)
            shm = materialize_checkpoint_to_shm(ckpt)

        try:
            assert (shm / "tokenizer" / "tokenizer.json").read_bytes() == sub_body
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)

    def test_shm_fallback_when_dev_shm_unavailable(
        self, tmp_path: Path, monkeypatch, capfd
    ) -> None:
        """Monkeypatched /dev/shm absence → fallback tempdir + WARN."""
        from paramem.backup import checkpoint_shard

        monkeypatch.setattr(checkpoint_shard, "_SHM_ROOT", tmp_path / "nonexistent")

        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        shm = materialize_checkpoint_to_shm(ckpt)

        try:
            for name, body in contents.items():
                assert (shm / name).read_bytes() == body
            captured = capfd.readouterr()
            assert "dev/shm unavailable" in captured.err
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)


class TestEncryptMaterializeRoundTrip:
    def test_round_trip_byte_identical(self, tmp_path: Path) -> None:
        """Encrypt → materialize returns byte-for-byte identical content."""
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            encrypt_checkpoint_dir(ckpt)
            shm = materialize_checkpoint_to_shm(ckpt)

        try:
            materialized = {
                entry.name: entry.read_bytes() for entry in shm.iterdir() if entry.is_file()
            }
            assert materialized == contents
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)
