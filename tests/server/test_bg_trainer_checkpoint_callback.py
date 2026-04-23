"""Unit tests for ``_EncryptCheckpointCallback`` in ``background_trainer``.

The callback is driven by HF Trainer hooks — we don't instantiate a real
Trainer here. Instead we stub the small arg / state surface the callback
actually reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    _clear_cipher_cache,
    is_pmem1_envelope,
)
from paramem.server.background_trainer import _EncryptCheckpointCallback


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


def _seed_two_checkpoints(root: Path) -> tuple[Path, Path]:
    """Write two synthetic HF-style checkpoint directories under *root*."""
    root.mkdir(parents=True, exist_ok=True)
    ckpts = []
    for step in (1, 2):
        ckpt = root / f"checkpoint-{step}"
        ckpt.mkdir()
        (ckpt / "adapter_model.safetensors").write_bytes(f"weights-{step}".encode())
        (ckpt / "optimizer.pt").write_bytes(f"opt-{step}".encode())
        (ckpt / "trainer_state.json").write_bytes(b'{"global_step": 1}')
        ckpts.append(ckpt)
    return tuple(ckpts)  # type: ignore[return-value]


class TestEncryptCheckpointCallbackOnSave:
    def test_on_save_encrypts_all_checkpoint_dirs_when_security_on(self, tmp_path: Path) -> None:
        """on_save walks checkpoint-* subdirs and encrypts every plaintext file."""
        ckpt1, ckpt2 = _seed_two_checkpoints(tmp_path)
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)
        cb = _EncryptCheckpointCallback()

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            cb.on_save(args, state=None, control=None)

        for ckpt in (ckpt1, ckpt2):
            for name in ("adapter_model.safetensors", "optimizer.pt", "trainer_state.json"):
                assert is_pmem1_envelope(ckpt / name), f"{ckpt.name}/{name} not wrapped"

    def test_on_save_noop_when_security_off(self, tmp_path: Path) -> None:
        """No master key → files remain plaintext."""
        ckpt1, _ = _seed_two_checkpoints(tmp_path)
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)

        _EncryptCheckpointCallback().on_save(args, state=None, control=None)

        assert (ckpt1 / "adapter_model.safetensors").read_bytes() == b"weights-1"
        assert not is_pmem1_envelope(ckpt1 / "adapter_model.safetensors")

    def test_on_save_tolerates_encrypt_failure(self, tmp_path: Path, capfd) -> None:
        """An encryption exception is logged but does not propagate.

        The WARN message is asserted via ``capfd`` because paramem's logging
        configuration writes directly to stderr and bypasses caplog's
        propagation path.
        """
        _seed_two_checkpoints(tmp_path)
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            with patch(
                "paramem.backup.checkpoint_shard.encrypt_checkpoint_dir",
                side_effect=OSError("disk full"),
            ):
                _EncryptCheckpointCallback().on_save(args, state=None, control=None)

        captured = capfd.readouterr()
        assert "Failed to encrypt checkpoint files" in captured.err

    def test_on_save_ignores_non_checkpoint_dirs(self, tmp_path: Path) -> None:
        """Only ``checkpoint-*`` subdirs are touched; siblings are left alone."""
        _seed_two_checkpoints(tmp_path)
        # Sibling directory that must be ignored.
        sibling = tmp_path / "runs"
        sibling.mkdir()
        (sibling / "events.out").write_bytes(b"tb-events")

        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)
        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            _EncryptCheckpointCallback().on_save(args, state=None, control=None)

        assert not is_pmem1_envelope(sibling / "events.out")
        assert (sibling / "events.out").read_bytes() == b"tb-events"


class TestEncryptCheckpointCallbackOnTrainBegin:
    def test_refuses_when_load_best_model_at_end_and_security_on(self, tmp_path: Path) -> None:
        """load_best_model_at_end=True + master key loaded → RuntimeError."""
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=True)

        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            with pytest.raises(RuntimeError, match="load_best_model_at_end"):
                _EncryptCheckpointCallback().on_train_begin(args, state=None, control=None)

    def test_allows_load_best_model_at_end_when_security_off(self, tmp_path: Path) -> None:
        """Without a master key the combo is harmless — no exception."""
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=True)
        _EncryptCheckpointCallback().on_train_begin(args, state=None, control=None)

    def test_allows_default_without_master_key(self, tmp_path: Path) -> None:
        """Default args (load_best_model_at_end=False) never raise."""
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)
        with patch.dict(os.environ, {MASTER_KEY_ENV_VAR: _make_key()}):
            _clear_cipher_cache()
            _EncryptCheckpointCallback().on_train_begin(args, state=None, control=None)
