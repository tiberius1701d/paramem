"""Tests for adapter safetensors full-file encryption helpers.

Covers:
- :func:`paramem.models.loader._encrypt_adapter_safetensors` — in-place
  encrypt of ``adapter_model.safetensors`` in a pending slot.
- :func:`paramem.models.loader._adapter_slot_for_load` — context manager
  that transparently decrypts an age-encrypted safetensors into anonymous
  RAM before handing the path to PEFT.
- :func:`paramem.backup.encryption.infra_paths` — verifies that
  ``adapter_model.safetensors`` files are included in the candidate set.
- :func:`paramem.models.loader.atomic_save_adapter` — verifies that Step 3.5
  (encrypt) is wired into the save sequence.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.backup.age_envelope import AGE_MAGIC
from paramem.backup.encryption import infra_paths
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.models.loader import _adapter_slot_for_load, _encrypt_adapter_safetensors

# Fake plaintext safetensors content (not a real tensor — just arbitrary bytes).
_FAKE_SAFETENSORS = b"\x00\x01\x02\x03fake_safetensors_payload"


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point the env + module default at it."""
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


def _make_slot(tmp_path: Path, content: bytes = _FAKE_SAFETENSORS) -> Path:
    """Create a minimal slot directory with ``adapter_model.safetensors``."""
    slot = tmp_path / "slot"
    slot.mkdir()
    (slot / "adapter_model.safetensors").write_bytes(content)
    (slot / "adapter_config.json").write_text('{"r": 8}')
    return slot


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch):
    """Isolate daily identity cache per test."""
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


# ---------------------------------------------------------------------------
# _encrypt_adapter_safetensors — Security OFF (no key)
# ---------------------------------------------------------------------------


class TestEncryptAdapterSafetensorsNoKey:
    """With no daily identity loaded the helper is a no-op rewrite."""

    def test_plaintext_rewrite_when_no_key(self, tmp_path, monkeypatch):
        """No key loaded → file content unchanged after encrypt call."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        slot = _make_slot(tmp_path)
        _encrypt_adapter_safetensors(slot)

        result = (slot / "adapter_model.safetensors").read_bytes()
        assert result == _FAKE_SAFETENSORS

    def test_no_age_magic_when_no_key(self, tmp_path, monkeypatch):
        """No key loaded → file does not acquire age envelope magic."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        slot = _make_slot(tmp_path)
        _encrypt_adapter_safetensors(slot)

        result = (slot / "adapter_model.safetensors").read_bytes()
        assert not result.startswith(AGE_MAGIC)


# ---------------------------------------------------------------------------
# _encrypt_adapter_safetensors — Security ON (key loaded)
# ---------------------------------------------------------------------------


class TestEncryptAdapterSafetensorsWithKey:
    """With a daily identity loaded the helper produces an age envelope."""

    def test_age_magic_after_encrypt(self, tmp_path, monkeypatch):
        """Daily identity loaded → safetensors becomes an age envelope."""
        _setup_daily(tmp_path, monkeypatch)
        slot = _make_slot(tmp_path)
        _encrypt_adapter_safetensors(slot)

        result = (slot / "adapter_model.safetensors").read_bytes()
        assert result.startswith(AGE_MAGIC), f"expected age magic; got {result[:30]!r}"

    def test_roundtrip_via_read_maybe_encrypted(self, tmp_path, monkeypatch):
        """Encrypt then read_maybe_encrypted returns original plaintext."""
        from paramem.backup.encryption import read_maybe_encrypted

        _setup_daily(tmp_path, monkeypatch)
        slot = _make_slot(tmp_path)
        _encrypt_adapter_safetensors(slot)

        recovered = read_maybe_encrypted(slot / "adapter_model.safetensors")
        assert recovered == _FAKE_SAFETENSORS

    def test_other_slot_files_unchanged(self, tmp_path, monkeypatch):
        """Only safetensors is touched; adapter_config.json is not modified."""
        _setup_daily(tmp_path, monkeypatch)
        slot = _make_slot(tmp_path)
        config_before = (slot / "adapter_config.json").read_text()
        _encrypt_adapter_safetensors(slot)
        config_after = (slot / "adapter_config.json").read_text()
        assert config_before == config_after


# ---------------------------------------------------------------------------
# _adapter_slot_for_load — plaintext path (no key or file absent)
# ---------------------------------------------------------------------------


class TestAdapterSlotForLoadPlaintext:
    """When the file is plaintext or absent the original slot is yielded unchanged."""

    def test_yields_original_slot_when_plaintext(self, tmp_path):
        """Plaintext safetensors → original slot path yielded."""
        slot = _make_slot(tmp_path)
        with _adapter_slot_for_load(slot) as load_path:
            assert load_path == slot

    def test_yields_original_slot_when_file_absent(self, tmp_path):
        """No safetensors present → original slot path yielded without error."""
        slot = tmp_path / "slot"
        slot.mkdir()
        (slot / "adapter_config.json").write_text('{"r": 8}')
        with _adapter_slot_for_load(slot) as load_path:
            assert load_path == slot


# ---------------------------------------------------------------------------
# _adapter_slot_for_load — encrypted path (key loaded)
# ---------------------------------------------------------------------------


class TestAdapterSlotForLoadEncrypted:
    """When safetensors is an age envelope a decrypted temp dir is yielded."""

    def test_temp_dir_contains_decrypted_safetensors(self, tmp_path, monkeypatch):
        """Encrypted safetensors → temp dir with decrypted bytes accessible."""
        _setup_daily(tmp_path, monkeypatch)
        slot = _make_slot(tmp_path)
        _encrypt_adapter_safetensors(slot)

        with _adapter_slot_for_load(slot) as load_path:
            # The yielded path must differ from the original slot.
            assert load_path != slot
            # The safetensors inside must be the decrypted plaintext.
            safe = load_path / "adapter_model.safetensors"
            assert safe.exists() or safe.is_symlink()
            data = safe.read_bytes()
            assert data == _FAKE_SAFETENSORS

    def test_temp_dir_cleaned_up_after_context(self, tmp_path, monkeypatch):
        """Temp directory must not survive context exit (no residue on disk)."""
        _setup_daily(tmp_path, monkeypatch)
        slot = _make_slot(tmp_path)
        _encrypt_adapter_safetensors(slot)

        with _adapter_slot_for_load(slot) as load_path:
            temp_recorded = load_path

        # After context exit the temp dir should be gone.
        assert not Path(temp_recorded).exists()

    def test_non_safetensors_files_copied_to_temp_dir(self, tmp_path, monkeypatch):
        """adapter_config.json must be accessible inside the temp dir."""
        _setup_daily(tmp_path, monkeypatch)
        slot = _make_slot(tmp_path)
        _encrypt_adapter_safetensors(slot)

        with _adapter_slot_for_load(slot) as load_path:
            config_path = load_path / "adapter_config.json"
            assert config_path.exists()
            assert config_path.read_text() == '{"r": 8}'


# ---------------------------------------------------------------------------
# infra_paths — includes adapter_model.safetensors
# ---------------------------------------------------------------------------


class TestInfraPathsIncludesSafetensors:
    """infra_paths enumerates adapter_model.safetensors files present on disk."""

    def test_safetensors_included_when_present(self, tmp_path):
        """A safetensors file under adapters/<tier>/<slot>/ is enumerated."""
        data_dir = tmp_path / "data"
        adapters = data_dir / "adapters" / "episodic" / "20240101-000000"
        adapters.mkdir(parents=True)
        safe_path = adapters / "adapter_model.safetensors"
        safe_path.write_bytes(b"fake")

        paths = infra_paths(data_dir)
        assert safe_path in paths

    def test_no_safetensors_when_adapters_absent(self, tmp_path):
        """No adapters root → no safetensors paths in candidate set."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        paths = infra_paths(data_dir)
        safetensors_paths = [p for p in paths if p.name == "adapter_model.safetensors"]
        assert safetensors_paths == []

    def test_multiple_slots_all_enumerated(self, tmp_path):
        """Two slot directories → both safetensors files appear in paths."""
        data_dir = tmp_path / "data"
        slot1 = data_dir / "adapters" / "episodic" / "20240101-000000"
        slot2 = data_dir / "adapters" / "semantic" / "20240101-000001"
        slot1.mkdir(parents=True)
        slot2.mkdir(parents=True)
        safe1 = slot1 / "adapter_model.safetensors"
        safe2 = slot2 / "adapter_model.safetensors"
        safe1.write_bytes(b"a")
        safe2.write_bytes(b"b")

        paths = infra_paths(data_dir)
        assert safe1 in paths
        assert safe2 in paths


# ---------------------------------------------------------------------------
# atomic_save_adapter — encrypt step wired in
# ---------------------------------------------------------------------------


class TestAtomicSaveAdapterEncryptStep:
    """atomic_save_adapter calls _encrypt_adapter_safetensors at Step 3.5."""

    def test_encrypt_called_on_save(self, tmp_path):
        """_encrypt_adapter_safetensors is invoked during atomic_save_adapter."""
        from paramem.models.loader import atomic_save_adapter

        fake_model = MagicMock()
        fake_model.peft_config = {"episodic": MagicMock()}

        def _fake_save_pretrained(slot_str, selected_adapters=None):
            slot = Path(slot_str)
            (slot / "adapter_model.safetensors").write_bytes(_FAKE_SAFETENSORS)
            (slot / "adapter_config.json").write_text('{"r": 8}')

        fake_model.save_pretrained.side_effect = _fake_save_pretrained

        with patch("paramem.models.loader._encrypt_adapter_safetensors") as mock_encrypt:
            atomic_save_adapter(fake_model, tmp_path / "adapters" / "episodic", "episodic")

        mock_encrypt.assert_called_once()
