"""Tests for paramem.adapters.registry_binding -- the venue-uniform boot
binding verdict.

Covers the verdict vocabulary (``VERIFIED``, ``NO_CANDIDATES``,
``KEYS_WITHOUT_SLOT``, ``NO_MATCHING_SLOT``, ``PAYLOAD_MISMATCH``,
``REGISTRY_ABSENT_WITH_SLOTS``), the payload digest check that runs for
both venues alike, the registry-absent-with-candidates resolution order
(resolves before any empty-digest slot-match attempt, even on a keyless
boot), parse-failure totality (a wrongly-typed candidate manifest yields a
verdict, never a crash), and the migrated-tree end-to-end path (the real
migration script's ``migrate()`` over a prior-schema fixture tree, then a
fresh boot-time ``verify_adapter_tree`` + ``verify_infrastructure_integrity``
read -- no model, no GPU).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from paramem.adapters.manifest import find_live_slot, tier_registry_sha256
from paramem.adapters.registry_binding import (
    NO_CANDIDATES,
    NO_MATCHING_SLOT,
    PAYLOAD_MISMATCH,
    REGISTRY_ABSENT_WITH_SLOTS,
    VERIFIED,
    verify_adapter_tree,
    verify_tier_binding,
)
from paramem.adapters.slot import write_slot
from paramem.server.manifest_status import ROW_STATUS_FOR_VERDICT, UNBOUND_ROW_STATUSES
from paramem.training.key_registry import KeyRegistry
from tests._manifest_fixtures import make_train_manifest, write_slot_files

# Ensure repo root is on sys.path so scripts/ is importable -- same pattern
# as tests/migrate/test_stamp_slot_manifests_v5.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_registry_and_train_slot(tier_root: Path, key: str, *, name: str) -> Path:
    """Registry with one active key + simhash, then a REAL written train
    (weights) slot bound to it -- through the same envelope production
    uses. Returns the written slot directory."""
    registry = KeyRegistry()
    registry.add(key)
    registry.set_simhash(key, 12345)
    tier_root.mkdir(parents=True, exist_ok=True)
    registry.save(tier_root / "indexed_key_registry.json")
    registry_hash = tier_registry_sha256(tier_root)
    manifest = make_train_manifest(name=name, registry_sha256=registry_hash, key_count=1)
    write_slot(
        tier_root, manifest=manifest, write_payload=lambda pending: write_slot_files(pending)
    )
    return find_live_slot(tier_root, registry_hash)


class TestATierWithNoKeysAndNoSlotStillPublishes:
    """A genuinely fresh tier -- no registry file, no slot candidates at
    all -- resolves NO_CANDIDATES, which is publishable: a fresh install
    must never block store publish."""

    def test_a_tier_with_no_keys_and_no_slot_still_publishes(self, tmp_path: Path) -> None:
        tier_root = tmp_path / "adapters" / "episodic"

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_CANDIDATES
        assert binding.publishable
        assert binding.registry is not None
        assert binding.registry.list_active() == []


class TestAPayloadWhoseBytesChangedIsWithheld:
    """Step 6's payload check runs for both venues alike: a bound train
    slot whose weight bytes no longer hash to the manifest's stamped
    digest resolves PAYLOAD_MISMATCH and is left unpublishable."""

    def test_a_payload_whose_bytes_changed_is_withheld(self, tmp_path: Path) -> None:
        tier_root = tmp_path / "adapters" / "episodic"
        slot = _write_registry_and_train_slot(tier_root, "graph1", name="episodic")

        # Corrupt the written payload bytes in place -- the digest stamped
        # into the manifest at write time no longer matches.
        (slot / "adapter_model.safetensors").write_bytes(b"corrupted-not-what-was-written")

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == PAYLOAD_MISMATCH
        assert not binding.publishable
        assert binding.slot == slot
        assert binding.manifest is not None
        assert binding.detail


class TestATornMemberIsWithheldAndReportedWhenTheLedgerIsLost:
    """The crash-window shape: write -> publish interrupted mid-bundle ->
    ledger lost. The member whose registry landed while its
    manifest rebind did not is the torn one -- its live registry digest
    matches no on-disk manifest, so it resolves NO_MATCHING_SLOT
    (unpublishable) rather than silently binding a stale slot."""

    def test_a_torn_member_is_withheld_and_reported_when_the_ledger_is_lost(
        self, tmp_path: Path
    ) -> None:
        tier_root = tmp_path / "adapters" / "episodic"
        _write_registry_and_train_slot(tier_root, "graph1", name="episodic")

        # Simulate the torn crash window: this member's REGISTRY lands
        # (rewritten with new content) but no new slot is ever written to
        # rebind it -- the manifest rebind that should have followed never
        # ran before the crash.
        registry = KeyRegistry.load(tier_root / "indexed_key_registry.json")
        registry.add("graph2")
        registry.set_simhash("graph2", 999)
        registry.save(tier_root / "indexed_key_registry.json")

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_MATCHING_SLOT
        assert not binding.publishable
        assert binding.detail  # reported: names the mismatch, non-empty
        # This verdict is one the operator-visible row vocabulary can name
        # (paramem.server.manifest_status) -- "reported" is not silent.
        assert ROW_STATUS_FOR_VERDICT[binding.status] in UNBOUND_ROW_STATUSES


class TestAScalarPayloadManifestYieldsAVerdictNotACrash:
    """A JSON-valid but wrongly-typed ``meta.json`` (``"payload": 5``) must
    not crash the tree walk -- ``find_live_slot``'s ``ManifestSchemaError``-
    only catch skips the unreadable candidate, so the tier resolves
    NO_MATCHING_SLOT (registry present, one candidate present but none
    readable/matching) rather than propagating a bare ``TypeError``."""

    def test_a_scalar_payload_manifest_yields_a_verdict_not_a_crash(self, tmp_path: Path) -> None:
        from tests._manifest_fixtures import write_raw_meta

        tier_root = tmp_path / "adapters" / "episodic"
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 1)
        tier_root.mkdir(parents=True, exist_ok=True)
        registry.save(tier_root / "indexed_key_registry.json")

        slot = tier_root / "20260101-000000"
        write_raw_meta(
            slot,
            {
                "schema_version": 5,
                "name": "episodic",
                "trained_at": "2026-01-01T00:00:00Z",
                "window_stamp": "",
                "payload": 5,
                "registry_sha256": "",
                "key_count": 0,
            },
        )

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_MATCHING_SLOT
        assert not binding.publishable
        assert binding.slot is None
        assert binding.candidate_count == 1


class TestAnUnhashablePayloadKindYieldsAVerdictNotACrash:
    """A JSON-valid but wrongly-typed ``payload.kind`` (a list, not a
    string) must not crash the tree walk. ``PayloadFingerprint.__post_init__``
    tests ``kind not in _PAYLOAD_KINDS`` -- a frozenset membership check
    that raises a bare (uncaught) ``TypeError`` on an unhashable value
    instead of the intended ``ManifestSchemaError``. ``_dict_to_manifest``
    now type-checks ``payload.kind`` before constructing
    ``PayloadFingerprint``, so this candidate is skipped like any other
    unreadable manifest rather than crashing the walk."""

    def test_an_unhashable_payload_kind_yields_a_verdict_not_a_crash(self, tmp_path: Path) -> None:
        from tests._manifest_fixtures import write_raw_meta

        tier_root = tmp_path / "adapters" / "episodic"
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 1)
        tier_root.mkdir(parents=True, exist_ok=True)
        registry.save(tier_root / "indexed_key_registry.json")

        slot = tier_root / "20260101-000000"
        write_raw_meta(
            slot,
            {
                "schema_version": 5,
                "name": "episodic",
                "trained_at": "2026-01-01T00:00:00Z",
                "window_stamp": "",
                "payload": {"kind": ["train"], "sha256": "a"},
                "registry_sha256": "",
                "key_count": 0,
            },
        )

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_MATCHING_SLOT
        assert not binding.publishable
        assert binding.slot is None
        assert binding.candidate_count == 1


class TestANonIterableTargetModulesYieldsAVerdictNotACrash:
    """A JSON-valid but wrongly-typed ``lora.target_modules`` (an int, not
    an array) must not crash the tree walk. ``tuple(lo["target_modules"])``
    raises a bare (uncaught) ``TypeError`` on a non-iterable value instead
    of the intended ``ManifestSchemaError``. ``_dict_to_manifest`` now
    type-checks ``lora.target_modules`` before constructing ``LoRAShape``,
    so this candidate is skipped like any other unreadable manifest rather
    than crashing the walk."""

    def test_a_non_iterable_target_modules_yields_a_verdict_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        from tests._manifest_fixtures import write_raw_meta

        tier_root = tmp_path / "adapters" / "episodic"
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 1)
        tier_root.mkdir(parents=True, exist_ok=True)
        registry.save(tier_root / "indexed_key_registry.json")

        slot = tier_root / "20260101-000000"
        write_raw_meta(
            slot,
            {
                "schema_version": 5,
                "name": "episodic",
                "trained_at": "2026-01-01T00:00:00Z",
                "window_stamp": "",
                "payload": {"kind": "train", "sha256": "a"},
                "registry_sha256": "",
                "key_count": 0,
                "base_model": {"repo": "r", "sha": "s", "hash": "h"},
                "tokenizer": {"name_or_path": "n", "vocab_size": 1, "merges_hash": "m"},
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": 5},
            },
        )

        binding = verify_tier_binding("episodic", tier_root)

        assert binding.status == NO_MATCHING_SLOT
        assert not binding.publishable
        assert binding.slot is None
        assert binding.candidate_count == 1


class TestKeylessBootShapeResolvesRegistryAbsentBeforeEmptyDigestBind:
    """A tier whose registry file is absent but which carries a slot
    candidate must resolve REGISTRY_ABSENT_WITH_SLOTS -- BEFORE any
    empty-digest bind attempt -- even when that candidate's manifest
    carries the donor/fresh-install ``registry_sha256=""`` convention.
    Binding via the empty digest here would let step 7 attempt the boot's
    FIRST decrypt and misdiagnose a missing daily passphrase as
    PAYLOAD_MISMATCH corruption."""

    def test_keyless_boot_shape_resolves_registry_absent_not_payload_mismatch(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from pyrage import x25519

        from paramem.backup.encryption import envelope_encrypt_bytes
        from paramem.backup.key_store import (
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
            write_recovery_pub_file,
        )

        daily = mint_daily_identity()
        recovery = x25519.Identity.generate()
        daily_path = tmp_path / "daily_key.age"
        recovery_path = tmp_path / "recovery.pub"
        write_daily_key_file(wrap_daily_identity(daily, "pw"), daily_path)
        write_recovery_pub_file(recovery.to_public(), recovery_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", daily_path)
        monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)

        tier_root = tmp_path / "adapters" / "episodic"
        manifest = make_train_manifest(name="episodic", registry_sha256="", key_count=0)

        def _write_encrypted_payload(pending: Path) -> None:
            write_slot_files(pending)
            plaintext = (pending / "adapter_model.safetensors").read_bytes()
            (pending / "adapter_model.safetensors").write_bytes(envelope_encrypt_bytes(plaintext))

        write_slot(tier_root, manifest=manifest, write_payload=_write_encrypted_payload)
        # No indexed_key_registry.json is ever written for this tier -- the
        # keyless-boot shape: registry file absent, encrypted payload, slot
        # candidate present.

        try:
            # Simulate the keyless boot: no daily identity loadable.
            monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
            _clear_daily_identity_cache()

            binding = verify_tier_binding("episodic", tier_root)

            assert binding.status == REGISTRY_ABSENT_WITH_SLOTS
            assert not binding.publishable
            assert binding.slot is None
            assert binding.manifest is None
        finally:
            _clear_daily_identity_cache()


class TestAMigratedTrainTreeMountsAndVerifiesEndToEnd:
    """Run the real migration script's ``migrate()`` over a prior-schema
    (v4) fixture tree spanning all three main tiers, then a fresh boot-time
    verification read: every tier VERIFIED, payload digests checked, and
    the integrity report carries no failed-level row. No model load --
    "mounts" is asserted at the verification/decision level only."""

    def test_a_migrated_train_tree_mounts_and_verifies_end_to_end(self, tmp_path: Path) -> None:
        from scripts.migrate.stamp_slot_manifests_v5 import migrate
        from tests._manifest_fixtures import v4_train_meta_dict, write_raw_meta

        adapter_root = tmp_path / "adapters"

        for tier_name in ("episodic", "semantic", "procedural"):
            tier_root = adapter_root / tier_name
            key = f"{tier_name[:3]}1"
            registry = KeyRegistry()
            registry.add(key)
            registry.set_simhash(key, 42)
            registry_bytes = registry.save_bytes()
            old_digest = _sha256_hex(registry_bytes)
            tier_root.mkdir(parents=True, exist_ok=True)
            (tier_root / "indexed_key_registry.json").write_bytes(registry_bytes)
            slot = tier_root / "20260101-000000"
            write_slot_files(slot, weight_bytes=f"real-weights-{tier_name}".encode())
            write_raw_meta(
                slot,
                v4_train_meta_dict(name=tier_name, registry_sha256=old_digest, key_count=1),
            )

        migrate(adapter_root, dry_run=False)

        tier_bindings = verify_adapter_tree(adapter_root)
        for tier_name in ("episodic", "semantic", "procedural"):
            binding = tier_bindings[tier_name]
            assert binding.status == VERIFIED, (tier_name, binding.status, binding.detail)
            assert binding.publishable
            assert binding.manifest is not None
            assert binding.manifest.payload.kind == "train"

        from unittest.mock import MagicMock

        from paramem.backup.integrity import verify_infrastructure_integrity

        cfg = MagicMock()
        cfg.adapter_dir = adapter_root
        cfg.paths.data = tmp_path / "data"

        report = verify_infrastructure_integrity(cfg, store=None, daily_loadable=False)

        assert report.ok is True, report.failures
        assert report.failures == []
