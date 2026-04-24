"""Unit tests for ``_EncryptCheckpointCallback`` in ``background_trainer``.

The callback is driven by HF Trainer hooks — we don't instantiate a real
Trainer here. Instead we stub the small arg / state surface the callback
actually reads.

Security ON is triggered by a loadable daily age identity
(``PARAMEM_DAILY_PASSPHRASE`` set + daily key file on disk).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from paramem.backup.age_envelope import is_age_envelope
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.server.background_trainer import _EncryptCheckpointCallback


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the daily-identity cache and the passphrase env var before and after each test."""
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    _clear_daily_identity_cache()
    yield
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    _clear_daily_identity_cache()


def _setup_daily_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passphrase: str = "test-pw",
) -> None:
    """Mint a daily identity, write it to tmp_path, and wire the env + module default.

    After this call ``daily_identity_loadable(DAILY_KEY_PATH_DEFAULT)`` returns
    ``True`` for the duration of the test.
    """
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()


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
    def test_on_save_encrypts_all_checkpoint_dirs_when_security_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """on_save walks checkpoint-* subdirs and encrypts every plaintext file."""
        ckpt1, ckpt2 = _seed_two_checkpoints(tmp_path)
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)
        cb = _EncryptCheckpointCallback()

        _setup_daily_identity(tmp_path, monkeypatch)
        cb.on_save(args, state=None, control=None)

        for ckpt in (ckpt1, ckpt2):
            for name in ("adapter_model.safetensors", "optimizer.pt", "trainer_state.json"):
                assert is_age_envelope(ckpt / name), f"{ckpt.name}/{name} not age-wrapped"

    def test_on_save_noop_when_security_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No daily identity → files remain plaintext."""
        ckpt1, _ = _seed_two_checkpoints(tmp_path)
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)
        # Point DAILY_KEY_PATH_DEFAULT at a nonexistent path — daily not loadable.
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent_daily_key.age",
        )

        _EncryptCheckpointCallback().on_save(args, state=None, control=None)

        assert (ckpt1 / "adapter_model.safetensors").read_bytes() == b"weights-1"
        assert not is_age_envelope(ckpt1 / "adapter_model.safetensors")

    def test_on_save_tolerates_encrypt_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd, caplog
    ) -> None:
        """An encryption exception is logged but does not propagate.

        Checks both ``capfd.err`` and ``caplog.records`` — pytest's log-capture
        routing differs between local and CI environments; the error message
        lands in one stream or the other, never neither.
        """
        import logging

        _seed_two_checkpoints(tmp_path)
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)
        caplog.set_level(logging.ERROR)

        _setup_daily_identity(tmp_path, monkeypatch)
        with patch(
            "paramem.backup.checkpoint_shard.encrypt_checkpoint_dir",
            side_effect=OSError("disk full"),
        ):
            _EncryptCheckpointCallback().on_save(args, state=None, control=None)

        log_text = capfd.readouterr().err + "\n".join(r.getMessage() for r in caplog.records)
        assert "Failed to encrypt checkpoint files" in log_text

    def test_on_save_ignores_non_checkpoint_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only ``checkpoint-*`` subdirs are touched; siblings are left alone."""
        _seed_two_checkpoints(tmp_path)
        # Sibling directory that must be ignored.
        sibling = tmp_path / "runs"
        sibling.mkdir()
        (sibling / "events.out").write_bytes(b"tb-events")

        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)
        _setup_daily_identity(tmp_path, monkeypatch)
        _EncryptCheckpointCallback().on_save(args, state=None, control=None)

        assert not is_age_envelope(sibling / "events.out")
        assert (sibling / "events.out").read_bytes() == b"tb-events"


class TestEncryptCheckpointCallbackOnTrainBegin:
    def test_refuses_when_load_best_model_at_end_and_security_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_best_model_at_end=True + daily identity loadable → RuntimeError."""
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=True)

        _setup_daily_identity(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="load_best_model_at_end"):
            _EncryptCheckpointCallback().on_train_begin(args, state=None, control=None)

    def test_allows_load_best_model_at_end_when_security_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a daily identity the combo is harmless — no exception."""
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=True)
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent_daily_key.age",
        )
        _EncryptCheckpointCallback().on_train_begin(args, state=None, control=None)

    def test_allows_default_without_daily_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default args (load_best_model_at_end=False) never raise, even with daily loaded."""
        args = SimpleNamespace(output_dir=str(tmp_path), load_best_model_at_end=False)
        _setup_daily_identity(tmp_path, monkeypatch)
        _EncryptCheckpointCallback().on_train_begin(args, state=None, control=None)
