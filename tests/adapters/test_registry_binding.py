"""Unit tests for paramem.adapters.registry_binding.

Covers:
- verify_tier_binding: one test per verdict (VERIFIED, VERIFIED with
  key_count=UNKNOWN, KEY_COUNT_MISMATCH, NO_MATCHING_SLOT, NO_CANDIDATES
  with a present registry (simulate venue), NO_CANDIDATES on a fresh
  install, REGISTRY_ABSENT_WITH_SLOTS, REGISTRY_UNREADABLE).
- Edge cases: absent registry + ""-stamped slot -> VERIFIED; an
  unparseable (raising) registry file with no slots -> REGISTRY_UNREADABLE,
  not NO_CANDIDATES; an encrypted registry verifies; a slot manifest with
  no adapter weight payload still verifies; a donor-named tier_root raises.
- Totality: a matched slot's meta.json that fails to parse (race arm) ->
  NO_MATCHING_SLOT; a mode-000 meta.json (permission failure during slot
  enumeration/read) resolves to a verdict instead of propagating OSError.
- verify_adapter_tree: yields main tiers + interims, never a donor.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UNKNOWN,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    TokenizerFingerprint,
    tier_registry_sha256,
    write_manifest,
)
from paramem.adapters.registry_binding import (
    KEY_COUNT_MISMATCH,
    NO_CANDIDATES,
    NO_MATCHING_SLOT,
    REGISTRY_ABSENT_WITH_SLOTS,
    REGISTRY_UNREADABLE,
    VERIFIED,
    verify_adapter_tree,
    verify_tier_binding,
)
from paramem.training.donor import DONOR_STORE_PREFIX
from paramem.training.key_registry import KeyRegistry

_ROOT_OR_NON_POSIX = os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0)
_SKIP_CHMOD = pytest.mark.skipif(
    _ROOT_OR_NON_POSIX,
    reason="chmod 000 is ineffective for root / non-POSIX platforms",
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_registry(tier_root: Path, keys: list[str]) -> Path:
    """Build a KeyRegistry with *keys* active and save it under *tier_root*."""
    reg = KeyRegistry()
    for key in keys:
        reg.add(key)
    path = tier_root / "indexed_key_registry.json"
    reg.save(path)
    return path


def _write_slot(
    tier_root: Path,
    *,
    registry_sha256: str,
    key_count: "int | str",
    stamp: str = "20260421-040000",
    name: str = "episodic",
) -> Path:
    """Write a manifest slot under *tier_root* with the given stamp fields."""
    slot = tier_root / stamp
    slot.mkdir(parents=True, exist_ok=True)
    manifest = AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=name,
        trained_at="2026-04-21T04:00:00Z",
        base_model=BaseModelFingerprint(repo="hf/model", sha="abc123", hash="sha256:deadbeef"),
        tokenizer=TokenizerFingerprint(
            name_or_path="hf/model", vocab_size=32000, merges_hash="cafebabe"
        ),
        lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
        registry_sha256=registry_sha256,
        key_count=key_count,
    )
    write_manifest(slot, manifest)
    return slot


def _mint_daily_identity(tmp_path: Path, monkeypatch, passphrase: str = "pw"):
    """Mint a daily identity, point module defaults at it, and return it."""
    from pyrage import x25519

    from paramem.backup.key_store import (
        DAILY_PASSPHRASE_ENV_VAR,
        mint_daily_identity,
        wrap_daily_identity,
        write_daily_key_file,
        write_recovery_pub_file,
    )

    daily = mint_daily_identity()
    recovery = x25519.Identity.generate()
    daily_path = tmp_path / "daily_key.age"
    recovery_path = tmp_path / "recovery.pub"
    write_daily_key_file(wrap_daily_identity(daily, passphrase), daily_path)
    write_recovery_pub_file(recovery.to_public(), recovery_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", daily_path)
    monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)
    return daily


@pytest.fixture(autouse=True)
def _isolate_daily_identity_cache():
    """Encrypted-registry tests mint/rotate a daily identity — isolate the
    module-level cache and env var so they never leak into other tests."""
    import os

    from paramem.backup.key_store import (
        DAILY_PASSPHRASE_ENV_VAR,
        _clear_daily_identity_cache,
    )

    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_daily_identity_cache()
    yield
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_daily_identity_cache()


# ---------------------------------------------------------------------------
# verify_tier_binding — one test per verdict
# ---------------------------------------------------------------------------


class TestVerifyTierBindingVerdicts:
    def test_verified_hash_and_count_agree(self, tmp_path: Path) -> None:
        """Matching hash + matching int key_count -> VERIFIED, publishable."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])
        live_hash = tier_registry_sha256(tier_root)
        slot = _write_slot(tier_root, registry_sha256=live_hash, key_count=1)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == VERIFIED
        assert binding.publishable is True
        assert binding.slot == slot
        assert binding.registry is not None
        assert binding.registry.list_active() == ["graph1"]
        assert binding.manifest is not None
        assert binding.manifest.key_count == 1
        assert binding.registry_present is True

    def test_verified_with_key_count_unknown(self, tmp_path: Path) -> None:
        """An UNKNOWN key_count stamp is never a mismatch -> VERIFIED."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1", "graph2"])
        live_hash = tier_registry_sha256(tier_root)
        _write_slot(tier_root, registry_sha256=live_hash, key_count=UNKNOWN)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == VERIFIED
        assert binding.publishable is True
        assert binding.manifest is not None
        assert binding.manifest.key_count == UNKNOWN

    def test_key_count_mismatch(self, tmp_path: Path) -> None:
        """Matching hash but disagreeing int key_count -> KEY_COUNT_MISMATCH."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])
        live_hash = tier_registry_sha256(tier_root)
        slot = _write_slot(tier_root, registry_sha256=live_hash, key_count=3)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == KEY_COUNT_MISMATCH
        assert binding.publishable is False
        assert binding.slot == slot
        assert binding.manifest is not None
        assert binding.manifest.key_count == 3
        assert binding.registry.list_active() == ["graph1"]

    def test_no_matching_slot(self, tmp_path: Path) -> None:
        """Registry present, one candidate slot, hash does not match -> NO_MATCHING_SLOT."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])
        _write_slot(tier_root, registry_sha256="a-stale-hash", key_count=1)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_MATCHING_SLOT
        assert binding.publishable is False
        assert binding.slot is None
        assert binding.candidate_count == 1
        assert binding.registry_present is True

    def test_no_candidates_with_registry_present_simulate_venue(self, tmp_path: Path) -> None:
        """A registry with active keys but zero weight-slot candidates
        (the simulate venue never writes a meta.json) -> NO_CANDIDATES,
        publishable — nothing on disk contradicts the registry."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_CANDIDATES
        assert binding.publishable is True
        assert binding.registry is not None
        assert binding.registry.list_active() == ["graph1"]
        assert binding.candidate_count == 0

    def test_no_candidates_fresh_install(self, tmp_path: Path) -> None:
        """Nothing on disk at all -> NO_CANDIDATES, registry_present False,
        publishable — the true never-trained shape."""
        tier_root = tmp_path / "episodic"

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_CANDIDATES
        assert binding.publishable is True
        assert binding.registry_present is False
        assert binding.registry is not None
        assert binding.registry.list_active() == []
        assert binding.candidate_count == 0

    def test_registry_absent_with_slots(self, tmp_path: Path) -> None:
        """Candidate slot(s) present, but the tier's registry file does not
        exist at all -> REGISTRY_ABSENT_WITH_SLOTS, not publishable."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_slot(tier_root, registry_sha256="some-other-hash", key_count=1)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == REGISTRY_ABSENT_WITH_SLOTS
        assert binding.publishable is False
        assert binding.registry_present is False
        assert binding.slot is None
        assert binding.candidate_count == 1

    def test_registry_unreadable_undecryptable(self, tmp_path: Path) -> None:
        """An age-encrypted registry with no daily identity loaded -> REGISTRY_UNREADABLE."""
        from paramem.backup.encryption import age_encrypt_bytes
        from paramem.backup.key_store import mint_daily_identity

        encrypter = mint_daily_identity()
        ciphertext = age_encrypt_bytes(b'{"active_keys":["graph1"]}', [encrypter.to_public()])

        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        (tier_root / "indexed_key_registry.json").write_bytes(ciphertext)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == REGISTRY_UNREADABLE
        assert binding.publishable is False
        assert binding.registry is None
        assert binding.manifest is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestVerifyTierBindingEdgeCases:
    def test_absent_registry_empty_hash_slot_is_verified(self, tmp_path: Path) -> None:
        """No registry file, slot stamped with the empty-hash fresh-install
        convention -> VERIFIED. A replay-disabled install must not be
        flagged as unhealthy."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        slot = _write_slot(tier_root, registry_sha256="", key_count=0)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == VERIFIED
        assert binding.publishable is True
        assert binding.slot == slot
        assert binding.registry_present is False
        assert binding.registry.list_active() == []

    def test_unparseable_registry_with_no_slots_is_registry_unreadable(
        self, tmp_path: Path
    ) -> None:
        """A registry file present but not valid JSON, with zero slot
        candidates, resolves to REGISTRY_UNREADABLE — the registry is read
        (and fails) before any candidate count is consulted, so this is
        never conflated with NO_CANDIDATES."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        (tier_root / "indexed_key_registry.json").write_bytes(b"not json at all")

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == REGISTRY_UNREADABLE
        assert binding.publishable is False
        assert binding.registry is None

    def test_encrypted_registry_verifies(self, tmp_path: Path, monkeypatch) -> None:
        """With a daily identity loaded, an encrypted registry matched by a
        correctly-hashed (plaintext-hashed) slot manifest verifies."""
        self._mint(tmp_path, monkeypatch)

        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])
        live_hash = tier_registry_sha256(tier_root)
        slot = _write_slot(tier_root, registry_sha256=live_hash, key_count=1)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == VERIFIED
        assert binding.slot == slot
        assert binding.registry.list_active() == ["graph1"]

    def _mint(self, tmp_path, monkeypatch):
        return _mint_daily_identity(tmp_path, monkeypatch)

    def test_manifest_without_payload_still_verifies(self, tmp_path: Path) -> None:
        """A slot carrying only meta.json (no adapter_model.safetensors) is
        still VERIFIED — payload completeness is out of scope here (see
        module docstring)."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])
        live_hash = tier_registry_sha256(tier_root)
        slot = _write_slot(tier_root, registry_sha256=live_hash, key_count=1)

        assert not (slot / "adapter_model.safetensors").exists()

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == VERIFIED

    def test_donor_named_tier_root_raises(self, tmp_path: Path) -> None:
        """A tier_root whose directory name starts with DONOR_STORE_PREFIX
        raises ValueError — donor stores are never a memory tier."""
        tier_root = tmp_path / f"{DONOR_STORE_PREFIX}abc123"
        tier_root.mkdir()

        with pytest.raises(ValueError, match="donor"):
            verify_tier_binding("donor", tier_root)

    def test_registry_hash_failure_after_successful_load_is_unreadable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A registry that loads successfully but whose hash step raises
        (e.g. the daily identity cache clears mid-read) still resolves to
        REGISTRY_UNREADABLE with registry=None — the invariant holds
        regardless of which of the two read steps failed."""
        import paramem.adapters.registry_binding as registry_binding_mod

        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])

        def _raise(_tier_root):
            raise RuntimeError("simulated mid-operation hash failure")

        monkeypatch.setattr(registry_binding_mod, "tier_registry_sha256", _raise)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == REGISTRY_UNREADABLE
        assert binding.registry is None
        assert binding.manifest is None

    def test_matched_slot_manifest_parse_failure_is_race_arm(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A candidate slot whose registry_sha256 MATCHES the live hash, but
        whose meta.json fails to parse when re-read at step 5 (a race —
        find_live_slot's own internal scan, via manifest.read_manifest,
        already succeeded moments earlier), resolves to NO_MATCHING_SLOT
        with slot=None and the parse failure named in detail. Only
        registry_binding's OWN bound read_manifest is patched — find_live_slot
        keeps using the real, unpatched one, so its internal scan still
        succeeds and returns the match."""
        import paramem.adapters.registry_binding as registry_binding_mod
        from paramem.adapters.manifest import ManifestSchemaError

        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])
        live_hash = tier_registry_sha256(tier_root)
        _write_slot(tier_root, registry_sha256=live_hash, key_count=1)

        def _raise(_path):
            raise ManifestSchemaError("simulated race: meta.json changed mid-read")

        monkeypatch.setattr(registry_binding_mod, "read_manifest", _raise)

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_MATCHING_SLOT
        assert binding.slot is None
        assert binding.manifest is None
        assert "race" in binding.detail

    @_SKIP_CHMOD
    def test_mode_000_meta_json_resolves_to_a_verdict_not_a_crash(self, tmp_path: Path) -> None:
        """A slot whose meta.json is unreadable (mode 000) must not crash
        verify_tier_binding — the whole point of the totality guarantee is
        that one tier's broken permissions can't take down verify_adapter_tree
        for every other tier."""
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()
        _write_registry(tier_root, ["graph1"])
        slot = tier_root / "20260421-040000"
        slot.mkdir()
        meta_path = slot / "meta.json"
        meta_path.write_text("{}")
        meta_path.chmod(0o000)

        try:
            binding = verify_tier_binding("episodic", tier_root)
        finally:
            meta_path.chmod(0o644)

        assert binding.status in (NO_MATCHING_SLOT,)
        assert binding.publishable is False
        assert binding.slot is None


# ---------------------------------------------------------------------------
# verify_adapter_tree
# ---------------------------------------------------------------------------


class TestVerifyAdapterTree:
    def test_yields_main_tiers_and_interims_never_donor(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()

        # Main tiers.
        for tier in ("episodic", "semantic", "procedural"):
            (adapter_dir / tier).mkdir()

        # One interim tier under episodic/interim_<stamp>/.
        interim_dir = adapter_dir / "episodic" / "interim_20260421T0400"
        interim_dir.mkdir(parents=True)
        _write_registry(interim_dir, ["graph9"])

        # A donor store sibling — must never be yielded.
        donor_dir = adapter_dir / f"{DONOR_STORE_PREFIX}xyz"
        donor_dir.mkdir()

        bindings = verify_adapter_tree(adapter_dir)

        assert set(bindings) == {
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260421T0400",
        }
        assert bindings["episodic_interim_20260421T0400"].status == NO_CANDIDATES
        assert bindings["episodic_interim_20260421T0400"].registry.list_active() == ["graph9"]
        assert all(not name.startswith(DONOR_STORE_PREFIX) for name in bindings)
