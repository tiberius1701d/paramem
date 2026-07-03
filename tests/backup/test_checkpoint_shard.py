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

import pytest

from paramem.backup.age_envelope import is_age_envelope
from paramem.backup.checkpoint_shard import (
    encrypt_checkpoint_dir,
    materialize_checkpoint_to_shm,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch):
    """Clear daily identity env + cache before/after each test."""
    monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


def _setup_daily_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Install a daily identity so the age writer path is active.

    Returns the minted identity so tests can verify round-trip outcomes.
    """
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


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
    def test_encrypt_wraps_every_file_when_daily_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Daily identity loaded → every file becomes an age envelope; count returned."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        n = encrypt_checkpoint_dir(ckpt)

        assert n == len(contents)
        for name in contents:
            assert is_age_envelope(ckpt / name), f"{name} should be age-wrapped"

    def test_encrypt_noop_when_security_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No daily key → no file is modified; return value is 0."""
        # Point the default at a missing file so daily_identity_loadable is False.
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        n = encrypt_checkpoint_dir(ckpt)

        assert n == 0
        for name, body in contents.items():
            assert (ckpt / name).read_bytes() == body
            assert not is_age_envelope(ckpt / name)

    def test_encrypt_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Second call after full encryption encrypts nothing new."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-42"
        _seed_checkpoint(ckpt)

        n1 = encrypt_checkpoint_dir(ckpt)
        n2 = encrypt_checkpoint_dir(ckpt)

        assert n1 > 0
        assert n2 == 0

    def test_encrypt_handles_subdirectories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files in nested subdirs are encrypted recursively."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-42"
        _seed_checkpoint(ckpt)
        subdir = ckpt / "tokenizer"
        subdir.mkdir()
        (subdir / "tokenizer.json").write_bytes(b'{"vocab_size": 32000}')

        encrypt_checkpoint_dir(ckpt)

        assert is_age_envelope(subdir / "tokenizer.json")

    def test_encrypt_skips_already_wrapped_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mixed state (some wrapped, some plaintext) encrypts only the plaintext."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-42"
        _seed_checkpoint(ckpt)

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
    def test_materialize_decrypts_all_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fully encrypted source → tempdir carries decrypted bytes."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)
        encrypt_checkpoint_dir(ckpt)
        shm = materialize_checkpoint_to_shm(ckpt)

        try:
            for name, body in contents.items():
                assert (shm / name).read_bytes() == body
                assert not is_age_envelope(shm / name)
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

    def test_materialize_handles_mixed_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Half-wrapped directory → each file decoded via the right branch."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        # Wrap half the files; leave the rest plaintext.
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

    def test_materialize_preserves_subdirectory_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nested subdirs survive materialization at the same relative path."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-42"
        _seed_checkpoint(ckpt)
        subdir = ckpt / "tokenizer"
        subdir.mkdir()
        sub_body = b'{"vocab_size": 32000}'
        (subdir / "tokenizer.json").write_bytes(sub_body)

        encrypt_checkpoint_dir(ckpt)
        shm = materialize_checkpoint_to_shm(ckpt)

        try:
            assert (shm / "tokenizer" / "tokenizer.json").read_bytes() == sub_body
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)

    def test_shm_fallback_when_dev_shm_unavailable(
        self, tmp_path: Path, monkeypatch, capfd, caplog
    ) -> None:
        """Monkeypatched /dev/shm absence → fallback tempdir + WARN.

        Checks both ``capfd.err`` and ``caplog.records`` so the assertion is
        robust across pytest log-capture variants (local stderr vs CI capture).
        """
        import logging

        from paramem.backup import checkpoint_shard

        monkeypatch.setattr(checkpoint_shard, "_SHM_ROOT", tmp_path / "nonexistent")
        caplog.set_level(logging.WARNING)

        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)

        shm = materialize_checkpoint_to_shm(ckpt)

        try:
            for name, body in contents.items():
                assert (shm / name).read_bytes() == body
            log_text = capfd.readouterr().err + "\n".join(r.getMessage() for r in caplog.records)
            assert "dev/shm unavailable" in log_text
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)


class TestEncryptMaterializeRoundTrip:
    def test_round_trip_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Encrypt → materialize returns byte-for-byte identical content."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-42"
        contents = _seed_checkpoint(ckpt)
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


# ---------------------------------------------------------------------------
# Age-envelope coverage — regression tests for the writer-flip integration.
# ---------------------------------------------------------------------------


class TestAgeEnvelopeCompatibility:
    def test_encrypt_with_daily_loaded_produces_age(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Daily identity loaded → encrypt_checkpoint_dir produces age files."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-1"
        _seed_checkpoint(ckpt)

        encrypt_checkpoint_dir(ckpt)

        for entry in ckpt.iterdir():
            if entry.is_file():
                assert is_age_envelope(entry), f"{entry.name} must be age post-encrypt"

    def test_encrypt_is_idempotent_on_age_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-running encrypt_checkpoint_dir on an age-encrypted tree must NOT
        re-encrypt (would produce nested double-wrapped envelopes)."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-2"
        _seed_checkpoint(ckpt)

        first = encrypt_checkpoint_dir(ckpt)
        assert first > 0
        snapshot = {entry.name: entry.read_bytes() for entry in ckpt.iterdir() if entry.is_file()}

        second = encrypt_checkpoint_dir(ckpt)
        assert second == 0, "second call must no-op; age files already encrypted"
        for entry in ckpt.iterdir():
            if entry.is_file():
                assert entry.read_bytes() == snapshot[entry.name], (
                    f"{entry.name} must NOT be re-encrypted"
                )

    def test_materialize_decrypts_age_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Encrypt → materialize → plaintext bytes in shm dir."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-3"
        contents = _seed_checkpoint(ckpt)
        encrypt_checkpoint_dir(ckpt)

        shm = materialize_checkpoint_to_shm(ckpt)
        try:
            for entry in shm.iterdir():
                if entry.is_file():
                    assert entry.read_bytes() == contents[entry.name], (
                        f"{entry.name}: age file must be decrypted into shm, not copied"
                    )
        finally:
            import shutil

            shutil.rmtree(shm, ignore_errors=True)

    def test_gate_fires_on_daily_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """encrypt_checkpoint_dir must activate when the daily identity is
        loaded — verifies the gate is ``daily_identity_loadable``."""
        _setup_daily_identity(tmp_path, monkeypatch)
        ckpt = tmp_path / "checkpoint-4"
        _seed_checkpoint(ckpt)

        count = encrypt_checkpoint_dir(ckpt)
        assert count > 0, "encrypt must fire when the daily identity is loaded"

    def test_gate_noop_when_no_keys_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Security OFF: no daily key. encrypt_checkpoint_dir leaves every file plaintext."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)

        ckpt = tmp_path / "checkpoint-5"
        contents = _seed_checkpoint(ckpt)

        count = encrypt_checkpoint_dir(ckpt)
        assert count == 0
        for entry in ckpt.iterdir():
            if entry.is_file():
                assert entry.read_bytes() == contents[entry.name]


# ---------------------------------------------------------------------------
# purge_partial_checkpoints — boot-time reconciliation of crash-interrupted
# checkpoint saves (paramem.training.trainer).
# ---------------------------------------------------------------------------


class TestPurgePartialCheckpoints:
    """Boot-time reconciliation of mid-crash checkpoint debris.

    A completed ``EncryptCheckpointCallback.on_save`` (via
    ``encrypt_checkpoint_dir``) leaves every file in a ``checkpoint-*/`` tree
    age-encrypted, so any plaintext file inside one proves the save was
    interrupted mid-crash. ``purge_partial_checkpoints`` deletes those dirs
    and clears any dangling ``staging_resume.json`` pointer into them.
    """

    def _base_state(self, **overrides) -> dict:
        state = {
            "adapter_name": "episodic",
            "dataset_fingerprint": "aabbccdd" * 8,
            "training_config_fingerprint": "11223344" * 8,
            "ram_checkpoint_path": "",
            "disk_checkpoint_path": "",
            "started_at": "2026-07-03T00:00:00+00:00",
            "updated_at": "2026-07-03T00:00:00+00:00",
        }
        state.update(overrides)
        return state

    def test_partial_checkpoint_purged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd, caplog
    ) -> None:
        """A checkpoint-*/ dir with a plaintext file among age siblings is purged.

        Checks both ``capfd.err`` and ``caplog.records`` so the assertion is
        robust across pytest log-capture variants (local stderr vs CI capture),
        mirroring ``test_shm_fallback_when_dev_shm_unavailable`` above.
        """
        import logging

        from paramem.training.trainer import purge_partial_checkpoints

        _setup_daily_identity(tmp_path, monkeypatch)
        adapters_root = tmp_path / "adapters"
        ckpt = adapters_root / "episodic" / "interim_20260703T1200" / "checkpoint-14"
        contents = _seed_checkpoint(ckpt)
        # Encrypt some but not all files — mixed state (mid-encrypt crash).
        from paramem.backup.encryption import write_infra_bytes

        write_infra_bytes(ckpt / "adapter_config.json", (ckpt / "adapter_config.json").read_bytes())

        caplog.set_level(logging.WARNING)
        purged = purge_partial_checkpoints(adapters_root)

        assert purged == [ckpt]
        assert not ckpt.exists()
        log_text = capfd.readouterr().err + "\n".join(r.getMessage() for r in caplog.records)
        assert "Purging partial checkpoint" in log_text
        # Sanity: seeded contents existed before the purge (precondition proof).
        assert len(contents) == 8

    def test_partial_checkpoint_purged_nested_per_adapter_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real live-debris shape: HF writes every co-resident PeftModel
        adapter into its own subdirectory of the checkpoint dir
        (``checkpoint-N/<adapter_name>/adapter_model.safetensors``), not a
        flat layout. A plaintext file nested one level deeper than
        ``_seed_checkpoint`` produces must still mark the whole
        ``checkpoint-N`` dir partial and purge it."""
        from paramem.training.trainer import purge_partial_checkpoints

        _setup_daily_identity(tmp_path, monkeypatch)
        adapters_root = tmp_path / "adapters"
        ckpt = adapters_root / "episodic" / "interim_20260703T1200" / "checkpoint-14"
        for adapter_name in ("episodic_interim_20260702T1200", "episodic_interim_20260703T0000"):
            adapter_dir = ckpt / adapter_name
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_model.safetensors").write_bytes(
                b"\x00\x01safetensors-stub" + os.urandom(256)
            )
            (adapter_dir / "adapter_config.json").write_bytes(b'{"r": 8, "alpha": 16}')
            (adapter_dir / "README.md").write_bytes(b"# Checkpoint\n")

        purged = purge_partial_checkpoints(adapters_root)

        assert purged == [ckpt]
        assert not ckpt.exists()

    def test_complete_checkpoint_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fully age-encrypted checkpoint-*/ dir survives and is not returned."""
        from paramem.training.trainer import purge_partial_checkpoints

        _setup_daily_identity(tmp_path, monkeypatch)
        adapters_root = tmp_path / "adapters"
        ckpt = adapters_root / "episodic" / "interim_20260703T1200" / "checkpoint-14"
        _seed_checkpoint(ckpt)
        encrypt_checkpoint_dir(ckpt)

        purged = purge_partial_checkpoints(adapters_root)

        assert purged == []
        assert ckpt.is_dir()
        for entry in ckpt.iterdir():
            if entry.is_file():
                assert is_age_envelope(entry)

    def test_durable_plaintext_not_purged_but_still_trips_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plaintext file with no checkpoint-* path component is untouched by
        purge, and still fails assert_mode_consistency via the MIXED age/plaintext
        branch (encryption.py:419) — the branch this whole change is motivated
        by (a genuinely inconsistent durable store must still hard-fail, only
        checkpoint-scratch plaintext should ever be silently reconciled)."""
        from paramem.backup.encryption import assert_mode_consistency, write_infra_bytes
        from paramem.backup.types import FatalConfigError
        from paramem.training.trainer import purge_partial_checkpoints

        _setup_daily_identity(tmp_path, monkeypatch)
        data_dir = tmp_path / "data"
        adapters_root = data_dir / "adapters"
        durable_slot = adapters_root / "episodic" / "20260703-000000"
        durable_slot.mkdir(parents=True)
        durable_file = durable_slot / "adapter_model.safetensors"
        durable_file.write_bytes(b"\x00\x01plaintext-safetensors")

        # A second, age-encrypted durable file (fixed infra_paths candidate,
        # not scratch) so the store is genuinely mixed on disk, not merely
        # plaintext-while-daily-loaded.
        semantic_graph = adapters_root / "semantic" / "graph.json"
        semantic_graph.parent.mkdir(parents=True)
        write_infra_bytes(semantic_graph, b'{"nodes": [], "edges": []}')
        assert is_age_envelope(semantic_graph)

        purged = purge_partial_checkpoints(adapters_root)

        assert purged == []
        assert durable_file.exists()
        assert not is_age_envelope(durable_file)
        assert is_age_envelope(semantic_graph)
        with pytest.raises(FatalConfigError) as exc_info:
            assert_mode_consistency(data_dir, daily_identity_loadable=True)
        assert "Mixed encryption state" in str(exc_info.value), (
            f"expected the mixed age/plaintext branch (encryption.py:419-438), "
            f"got: {exc_info.value}"
        )

    def test_pointer_clear_disk_checkpoint_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """staging_resume.json's disk_checkpoint_path into a purged dir is cleared."""
        from paramem.training.trainer import (
            _read_staging_resume,
            _write_staging_resume,
            purge_partial_checkpoints,
        )

        _setup_daily_identity(tmp_path, monkeypatch)
        adapters_root = tmp_path / "adapters"
        slot = adapters_root / "episodic" / "interim_20260703T1200"
        ckpt = slot / "checkpoint-14"
        _seed_checkpoint(ckpt)

        resume_path = slot / "staging_resume.json"
        state = self._base_state(disk_checkpoint_path=str(ckpt))
        _write_staging_resume(resume_path, state)
        assert is_age_envelope(resume_path)

        purge_partial_checkpoints(adapters_root)

        assert not ckpt.exists()
        reread = _read_staging_resume(resume_path)
        assert reread["disk_checkpoint_path"] == ""
        assert reread["ram_checkpoint_path"] == ""
        assert reread["adapter_name"] == "episodic"
        assert reread["dataset_fingerprint"] == state["dataset_fingerprint"]
        assert is_age_envelope(resume_path)

    def test_pointer_clear_ram_checkpoint_path_nested_bg_mirror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ram_checkpoint_path nested at <slot>/bg_checkpoint_epoch/checkpoint-N
        is cleared even though the marker lives at the enclosing <slot>/ root
        (no enclosing-directory proximity matching — resolves by path
        containment)."""
        from paramem.training.trainer import (
            _read_staging_resume,
            _write_staging_resume,
            purge_partial_checkpoints,
        )

        _setup_daily_identity(tmp_path, monkeypatch)
        adapters_root = tmp_path / "adapters"
        slot = adapters_root / "episodic" / "interim_20260703T1200"
        ckpt = slot / "bg_checkpoint_epoch" / "checkpoint-9"
        _seed_checkpoint(ckpt)

        resume_path = slot / "staging_resume.json"
        state = self._base_state(ram_checkpoint_path=str(ckpt))
        _write_staging_resume(resume_path, state)

        purged = purge_partial_checkpoints(adapters_root)

        assert purged == [ckpt]
        assert not ckpt.exists()
        reread = _read_staging_resume(resume_path)
        assert reread["ram_checkpoint_path"] == ""
        assert reread["disk_checkpoint_path"] == ""

    def test_pointer_not_cleared_for_surviving_complete_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pointer resolving to a complete (surviving) checkpoint elsewhere is
        left untouched even when purging occurs for an unrelated slot."""
        from paramem.training.trainer import (
            _read_staging_resume,
            _write_staging_resume,
            purge_partial_checkpoints,
        )

        _setup_daily_identity(tmp_path, monkeypatch)
        adapters_root = tmp_path / "adapters"

        # Slot A: partial checkpoint that will be purged.
        slot_a = adapters_root / "episodic" / "interim_20260703T1200"
        partial_ckpt = slot_a / "checkpoint-14"
        _seed_checkpoint(partial_ckpt)

        # Slot B: complete checkpoint, pointer must survive untouched.
        slot_b = adapters_root / "semantic" / "interim_20260703T1200"
        complete_ckpt = slot_b / "checkpoint-10"
        _seed_checkpoint(complete_ckpt)
        encrypt_checkpoint_dir(complete_ckpt)

        resume_path_b = slot_b / "staging_resume.json"
        state_b = self._base_state(disk_checkpoint_path=str(complete_ckpt))
        _write_staging_resume(resume_path_b, state_b)

        purged = purge_partial_checkpoints(adapters_root)

        assert purged == [partial_ckpt]
        assert complete_ckpt.is_dir()
        reread_b = _read_staging_resume(resume_path_b)
        assert reread_b["disk_checkpoint_path"] == str(complete_ckpt)

    def test_security_off_returns_empty_and_purges_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No loadable daily key → self-gate returns [] and leaves plaintext
        checkpoints untouched, even though they would otherwise be flagged
        partial."""
        from paramem.training.trainer import purge_partial_checkpoints

        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", tmp_path / "absent.age"
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)

        adapters_root = tmp_path / "adapters"
        ckpt = adapters_root / "episodic" / "interim_20260703T1200" / "checkpoint-14"
        _seed_checkpoint(ckpt)

        purged = purge_partial_checkpoints(adapters_root)

        assert purged == []
        assert ckpt.is_dir()
        assert list(ckpt.iterdir())
