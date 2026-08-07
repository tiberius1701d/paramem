"""Tests for the infrastructure-envelope primitives.

The encryption module exposes:
- ``envelope_encrypt_bytes`` / ``envelope_decrypt_bytes`` — age envelope helpers.
- ``write_infra_bytes`` — atomic infra writer, delegates to age when daily loaded.
- ``assert_mode_consistency`` — startup refuse on mixed-state (age / plaintext) mismatch.
- ``read_maybe_encrypted`` — universal reader dispatching by envelope magic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paramem.backup.encryption import (
    assert_mode_consistency,
    envelope_decrypt_bytes,
    envelope_encrypt_bytes,
    infra_paths,
    read_maybe_encrypted,
    write_infra_bytes,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.backup.types import FatalConfigError


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point the env + module default at it."""
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch):
    """Isolate daily identity cache per test."""
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


# ---------------------------------------------------------------------------
# envelope_encrypt_bytes — age path
# ---------------------------------------------------------------------------


class TestInfraPathsSkipsPendingDelete:
    """``infra_paths``' safetensors/staging_resume ``rglob`` walks must not
    descend into ``.pending-delete/`` (the reap tombstone directory,
    ``paramem.memory.persistence``) — condemned-but-not-yet-deleted debris
    sitting there must never be listed for encryption/rotation."""

    def test_safetensors_under_pending_delete_excluded(self, tmp_path: Path) -> None:
        adapters_root = tmp_path / "adapters"
        live_slot = adapters_root / "episodic" / "20260421-000000"
        live_slot.mkdir(parents=True)
        (live_slot / "adapter_model.safetensors").write_bytes(b"live")

        stray = adapters_root / ".pending-delete" / "episodic" / "20260101-000000"
        stray.mkdir(parents=True)
        (stray / "adapter_model.safetensors").write_bytes(b"condemned")

        paths = infra_paths(tmp_path)
        assert live_slot / "adapter_model.safetensors" in paths
        assert stray / "adapter_model.safetensors" not in paths

    def test_staging_resume_under_pending_delete_excluded(self, tmp_path: Path) -> None:
        adapters_root = tmp_path / "adapters"
        live_slot = adapters_root / "episodic"
        live_slot.mkdir(parents=True)
        (live_slot / "staging_resume.json").write_bytes(b"{}")

        stray = adapters_root / ".pending-delete" / "episodic"
        stray.mkdir(parents=True)
        (stray / "staging_resume.json").write_bytes(b"{}")

        paths = infra_paths(tmp_path)
        assert live_slot / "staging_resume.json" in paths
        assert stray / "staging_resume.json" not in paths


class TestEnvelopeEncryptBytesHelper:
    """``envelope_encrypt_bytes`` returns plaintext when no key is loaded,
    and an age envelope when the daily identity is loadable.
    """

    def test_plaintext_when_no_keys_loaded(self, tmp_path, monkeypatch):
        """No daily identity loaded → returns plaintext unchanged."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        result = envelope_encrypt_bytes(b"payload")
        assert result == b"payload"

    def test_age_when_daily_loaded(self, tmp_path, monkeypatch):
        """Daily identity loaded → returns age envelope."""
        from paramem.backup.age_envelope import AGE_MAGIC

        _setup_daily(tmp_path, monkeypatch)

        result = envelope_encrypt_bytes(b"payload")
        assert result.startswith(AGE_MAGIC), f"expected age magic; got {result[:30]!r}"


# ---------------------------------------------------------------------------
# envelope_decrypt_bytes — age path
# ---------------------------------------------------------------------------


class TestEnvelopeDecryptBytesHelper:
    def test_dispatches_age_via_loaded_daily(self, tmp_path, monkeypatch):
        """age envelope + daily loaded → decrypts correctly."""
        from paramem.backup.age_envelope import age_encrypt_bytes

        ident = _setup_daily(tmp_path, monkeypatch)

        envelope = age_encrypt_bytes(b"payload", [ident.to_public()])
        assert envelope_decrypt_bytes(envelope) == b"payload"

    def test_age_without_daily_raises_actionable_runtime_error(self, tmp_path, monkeypatch):
        """age envelope without loaded daily → RuntimeError naming the env var."""
        from pyrage import x25519

        from paramem.backup.age_envelope import age_encrypt_bytes

        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        _clear_daily_identity_cache()

        envelope = age_encrypt_bytes(b"x", [x25519.Identity.generate().to_public()])
        with pytest.raises(RuntimeError, match=DAILY_PASSPHRASE_ENV_VAR):
            envelope_decrypt_bytes(envelope)

    def test_non_age_bytes_raise_runtime_error(self):
        """Passing plaintext to envelope_decrypt_bytes raises RuntimeError."""
        with pytest.raises(RuntimeError, match="age magic"):
            envelope_decrypt_bytes(b"not-an-age-envelope")


# ---------------------------------------------------------------------------
# write_infra_bytes / read_maybe_encrypted — plaintext path
# ---------------------------------------------------------------------------


class TestSecurityOffRoundtrip:
    """write_infra_bytes + read_maybe_encrypted round-trip — plaintext (Security OFF)."""

    def test_plaintext_roundtrip_when_no_key(self, tmp_path, monkeypatch):
        """No daily identity → on-disk is plaintext; read_maybe_encrypted returns it."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        target = tmp_path / "infra.json"
        payload = b'{"hello": "world"}'
        write_infra_bytes(target, payload)
        # No key → on-disk is plaintext.
        assert target.read_bytes() == payload
        assert read_maybe_encrypted(target) == payload

    def test_encrypted_roundtrip_when_daily_loaded(self, tmp_path, monkeypatch):
        """Daily identity loaded → on-disk is age envelope; round-trips to plaintext."""
        from paramem.backup.age_envelope import AGE_MAGIC, is_age_envelope

        ident = _setup_daily(tmp_path, monkeypatch)  # noqa: F841

        target = tmp_path / "infra.json"
        payload = b'{"secret": "value"}'
        write_infra_bytes(target, payload)
        # On-disk starts with age magic.
        on_disk = target.read_bytes()
        assert on_disk.startswith(AGE_MAGIC)
        assert on_disk != payload
        assert is_age_envelope(target)
        # Read round-trips to plaintext.
        assert read_maybe_encrypted(target) == payload

    def test_write_is_atomic(self, tmp_path, monkeypatch):
        """Temp file is cleaned up; no partial write visible."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        target = tmp_path / "infra.json"
        write_infra_bytes(target, b"payload")
        assert target.exists()
        assert not (tmp_path / "infra.json.tmp").exists()


# ---------------------------------------------------------------------------
# write_infra_json — the one chokepoint for JSON-shaped infrastructure files
# ---------------------------------------------------------------------------


class TestWriteInfraJson:
    """write_infra_json = json.dumps(..., indent=2) + parent mkdir +
    write_infra_bytes — the single writer collapsed from the former
    byte-identical duplicates in paramem.server.consolidation and
    paramem.training.consolidation."""

    def test_roundtrip_dict_plaintext(self, tmp_path, monkeypatch):
        import json

        from paramem.backup.encryption import write_infra_json

        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        target = tmp_path / "infra.json"
        data = {"cycle_count": 5, "promoted_keys": ["graph1"], "keys": {}}
        write_infra_json(target, data)

        assert target.read_text() == json.dumps(data, indent=2)
        assert json.loads(read_maybe_encrypted(target).decode("utf-8")) == data

    def test_roundtrip_list_encrypted_when_daily_loaded(self, tmp_path, monkeypatch):
        """Same envelope posture as write_infra_bytes -- age-wrapped when a
        daily identity is loaded -- for a list payload too."""
        import json

        from paramem.backup.age_envelope import AGE_MAGIC

        _setup_daily(tmp_path, monkeypatch)

        from paramem.backup.encryption import write_infra_json

        target = tmp_path / "infra.json"
        write_infra_json(target, [1, 2, 3])

        assert target.read_bytes().startswith(AGE_MAGIC)
        assert json.loads(read_maybe_encrypted(target).decode("utf-8")) == [1, 2, 3]

    def test_creates_missing_parent_directory(self, tmp_path, monkeypatch):
        """write_infra_bytes requires the parent to already exist;
        write_infra_json's whole reason to exist is doing that for the
        caller."""
        from paramem.backup.encryption import write_infra_json

        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        target = tmp_path / "nested" / "registry" / "key_metadata.json"
        assert not target.parent.exists()

        write_infra_json(target, {"ok": True})

        assert target.exists()


# ---------------------------------------------------------------------------
# assert_mode_consistency — age-only two-case matrix
# ---------------------------------------------------------------------------


class TestAssertModeConsistency:
    def test_case_age_with_daily_loadable_is_ok(self, tmp_path, monkeypatch):
        """age files + daily identity loadable → proceed."""
        from paramem.backup.age_envelope import age_encrypt_bytes

        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        # Must not raise.
        assert_mode_consistency(data, daily_identity_loadable=True)

    def test_case_unset_plaintext_is_ok(self, tmp_path):
        """No daily identity + plaintext on disk → Security OFF, proceed."""
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(b'{"plain":true}')
        assert_mode_consistency(data, daily_identity_loadable=False)

    def test_case_plaintext_with_daily_loadable_refuses(self, tmp_path, monkeypatch):
        """Daily identity loaded + plaintext on disk → refuse."""
        _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(b'{"plain":true}')
        with pytest.raises(FatalConfigError, match="daily identity is loaded"):
            assert_mode_consistency(data, daily_identity_loadable=True)

    def test_case_age_without_daily_refuses(self, tmp_path, monkeypatch):
        """age on disk + no daily identity → refuse."""
        from pyrage import x25519

        from paramem.backup.age_envelope import age_encrypt_bytes

        ident = x25519.Identity.generate()
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))
        with pytest.raises(FatalConfigError, match=DAILY_PASSPHRASE_ENV_VAR):
            assert_mode_consistency(data, daily_identity_loadable=False)

    def test_mixed_plaintext_and_age_refuses_regardless_of_key(self, tmp_path, monkeypatch):
        """Mixed plaintext + age envelopes → refuse regardless of daily identity."""
        from paramem.backup.age_envelope import age_encrypt_bytes

        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))
        (data / "speaker_profiles.json").write_bytes(b'{"plain":true}')

        with pytest.raises(FatalConfigError, match="Mixed encryption state"):
            assert_mode_consistency(data, daily_identity_loadable=True)
        with pytest.raises(FatalConfigError, match="Mixed encryption state"):
            assert_mode_consistency(data, daily_identity_loadable=False)

    def test_empty_data_dir_is_consistent(self, tmp_path):
        """Fresh deployment with no infra files → OK in both modes."""
        assert_mode_consistency(tmp_path, daily_identity_loadable=False)
        assert_mode_consistency(tmp_path, daily_identity_loadable=True)

    def test_missing_data_dir_is_consistent(self, tmp_path):
        """Non-existent path probe is neutral."""
        ghost = tmp_path / "does-not-exist"
        assert_mode_consistency(ghost, daily_identity_loadable=False)
        assert_mode_consistency(ghost, daily_identity_loadable=True)

    def test_carve_out_state_files_ignored(self, tmp_path, monkeypatch):
        """state/trial.json and state/backup.json are plaintext-by-design and
        must not trigger a mismatch when the rest of the store is age-encrypted."""
        from paramem.backup.age_envelope import age_encrypt_bytes

        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))
        # Write carve-out files as plaintext.
        (data / "state").mkdir()
        (data / "state" / "trial.json").write_bytes(b'{"carve":"out"}')
        (data / "state" / "backup.json").write_bytes(b'{"carve":"out"}')

        # Must not raise — the carve-out files are not in the probed set.
        assert_mode_consistency(data, daily_identity_loadable=True)

    def test_tier_graph_plaintext_with_daily_refuses(self, tmp_path, monkeypatch):
        """Plaintext graph.json under adapter_dir while daily is loaded → refuse.

        After consolidation the tier graph lives at
        ``data/adapters/<tier>/graph.json``.  A mixed-encryption state
        (age-encrypted registry.json + plaintext adapter graph) is detected
        by ``assert_mode_consistency`` because ``infra_paths`` probes the
        per-tier adapter paths.
        """
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        from paramem.backup.age_envelope import age_encrypt_bytes

        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        # Write plaintext under adapter_dir/episodic/ — this is the canonical
        # path that infra_paths probes; mixed state must be detected.
        adapter_ep = data / "adapters" / "episodic"
        adapter_ep.mkdir(parents=True)
        (adapter_ep / "graph.json").write_bytes(b'{"nodes": [], "links": []}')  # plaintext

        with pytest.raises(FatalConfigError, match="Mixed encryption state"):
            assert_mode_consistency(data, daily_identity_loadable=True)

    def test_tier_graph_age_envelope_passes(self, tmp_path, monkeypatch):
        """age envelope in adapter_dir/semantic/graph.json alongside age in data → OK."""
        from paramem.backup.age_envelope import age_encrypt_bytes

        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        # Write an age-encrypted graph under the canonical adapter path.
        adapter_sem = data / "adapters" / "semantic"
        adapter_sem.mkdir(parents=True)
        (adapter_sem / "graph.json").write_bytes(
            age_encrypt_bytes(b'{"nodes": [], "links": []}', [ident.to_public()])
        )

        # Must not raise — both files are age-encrypted, daily identity available.
        assert_mode_consistency(data, daily_identity_loadable=True)

    def test_outside_data_dir_not_probed(self, tmp_path, monkeypatch):
        """Files outside data_dir are not in infra_paths and do not affect the scan.

        A plaintext file in a directory that is not under data_dir must NOT
        trigger a mixed-encryption refusal — only files that infra_paths
        returns are considered.
        """
        ident = _setup_daily(tmp_path, monkeypatch)
        data = tmp_path / "data"
        data.mkdir()
        from paramem.backup.age_envelope import age_encrypt_bytes

        (data / "registry.json").write_bytes(age_encrypt_bytes(b"{}", [ident.to_public()]))

        # A plaintext file entirely outside data_dir — must not be probed.
        rogue = tmp_path / "rogue-dir"
        (rogue / "episodic").mkdir(parents=True)
        (rogue / "episodic" / "quads.json").write_bytes(b"[]")

        # Must not raise — rogue-dir is not in infra_paths(data_dir).
        assert_mode_consistency(data, daily_identity_loadable=True)


# ---------------------------------------------------------------------------
# Integration: backup.write produces age envelope when daily is loaded
# ---------------------------------------------------------------------------


class TestBackupWriteUsesAgeEnvelope:
    def test_age_slot_encrypted_is_true(self, tmp_path, monkeypatch):
        """Writing a backup slot with the daily identity loaded → encrypted=True."""
        from paramem.backup.backup import write as backup_write
        from paramem.backup.meta import read_meta
        from paramem.backup.types import ArtifactKind
        from paramem.server.config import ServerBackupsConfig

        _setup_daily(tmp_path, monkeypatch)
        recovery_path = tmp_path / "absent.pub"
        monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)

        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            b"model: mistral\n",
            meta_fields={"tier": "scheduled"},
            backups_root=tmp_path,
            backups_cfg=ServerBackupsConfig(),
        )
        meta = read_meta(slot_dir)

        assert meta.encrypted is True

    def test_plaintext_slot_encrypted_is_false(self, tmp_path, monkeypatch):
        """No daily identity → plaintext slot with encrypted=False."""
        from paramem.backup.backup import write as backup_write
        from paramem.backup.meta import read_meta
        from paramem.backup.types import ArtifactKind
        from paramem.server.config import ServerBackupsConfig

        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            b"model: mistral\n",
            meta_fields={"tier": "scheduled"},
            backups_root=tmp_path,
            backups_cfg=ServerBackupsConfig(),
        )
        meta = read_meta(slot_dir)

        assert meta.encrypted is False
