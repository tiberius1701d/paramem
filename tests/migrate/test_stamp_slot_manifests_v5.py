"""Tests for scripts/migrate/stamp_slot_manifests_v5.py -- the one-shot,
offline metadata migration to schema v5.

Most tests call :func:`migrate` (the walk + per-tier logic) directly, never
``main()``. ``TestMainLivenessGuard`` is the exception: it exercises
``main()``'s own liveness guard, which routes through the project's single
``systemctl`` transport seam (:mod:`paramem.utils.systemctl`) -- the
autouse host-isolation guard (``tests/conftest.py``) already stubs that
seam for every test, so this file only needs to override its return value,
never reach real systemd. The guard's other input, ``pgrep``, is a
read-only host process scan and is left real.

Covers the v5 migration script's design: raw pre-migration reads on both
artifacts (never the current-schema-only readers), a registry-last write
per tier, bound-slot resolution before any write with slot mtimes
preserved, whole-tree idempotence, and fail-loud on anything outside
boot's own partial-slot classification -- the three tier-shape rules
(partial-scratch skip, zero-candidate registry-only migration,
candidates-present-but-unbound stop).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    find_live_slot,
    read_manifest,
    tier_registry_sha256,
    write_manifest,
)
from paramem.training.donor import DONOR_STORE_PREFIX
from paramem.training.key_registry import KeyRegistry

# Ensure repo root is on sys.path so scripts/ is importable (same pattern as
# tests/test_ingest_cli.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.migrate.stamp_slot_manifests_v5 import (  # noqa: E402  # isort: skip
    TierMigrationBlocked,
    _migrate_registry_bytes,
    main,
    migrate,
)
from tests._manifest_fixtures import (  # noqa: E402  # isort: skip
    make_train_manifest,
    v4_train_meta_dict,
    write_canonical_registry,
    write_noncanonical_registry,
    write_pre_change_stale_registry,
    write_raw_meta,
    write_slot_files,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "migrate" / "stamp_slot_manifests_v5.py"
)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot(root: Path) -> dict[str, tuple[bytes, float]]:
    """(relative path -> (bytes, mtime)) for every file under *root*."""
    out: dict[str, tuple[bytes, float]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[str(p.relative_to(root))] = (p.read_bytes(), p.stat().st_mtime)
    return out


class TestPayloadDigestStamping:
    def test_migration_stamps_the_real_payload_digest_from_the_bytes_on_disk(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_noncanonical_registry(tier_root, ["graph1", "graph2"])
        old_digest = _sha256_hex(registry_bytes)

        slot = tier_root / "20260101-000000"
        weight_bytes = b"real donor-style plaintext weights"
        write_slot_files(slot, weight_bytes=weight_bytes)
        write_raw_meta(slot, v4_train_meta_dict(name="episodic", registry_sha256=old_digest))

        migrate(adapter_root, dry_run=False)

        manifest = read_manifest(slot)
        assert manifest.payload.sha256 == _sha256_hex(weight_bytes)


class TestFingerprintAndBindingFieldsCarryOver:
    def test_migration_carries_the_existing_fingerprints_and_binding_fields_over(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_noncanonical_registry(tier_root, ["graph1"])
        old_digest = _sha256_hex(registry_bytes)
        new_digest = _sha256_hex(KeyRegistry.load_from_bytes(registry_bytes).save_bytes())

        bound_slot = tier_root / "20260101-000000"
        write_slot_files(bound_slot)
        write_raw_meta(
            bound_slot,
            v4_train_meta_dict(name="episodic", registry_sha256=old_digest, key_count=1),
        )

        # A sibling slot whose registry_sha256/key_count already carry the
        # documented UNKNOWN sentinel -- must be carried across unchanged,
        # not overwritten with a resolved value it never had.
        unknown_slot = tier_root / "20260102-000000"
        write_slot_files(unknown_slot)
        write_raw_meta(
            unknown_slot,
            v4_train_meta_dict(name="episodic", registry_sha256="unknown", key_count="unknown"),
        )

        migrate(adapter_root, dry_run=False)

        bound_manifest = read_manifest(bound_slot)
        assert bound_manifest.name == "episodic"
        assert bound_manifest.base_model.repo == "test-org/test-base-model"
        assert bound_manifest.tokenizer.name_or_path == "test-org/test-base-model"
        assert bound_manifest.lora.rank == 8
        assert bound_manifest.registry_sha256 == new_digest

        unknown_manifest = read_manifest(unknown_slot)
        assert unknown_manifest.registry_sha256 == "unknown"
        assert unknown_manifest.key_count == "unknown"


class TestCoversEveryStoreShape:
    def test_migration_covers_main_tiers_interim_slots_and_donor_stores(self, tmp_path):
        adapter_root = tmp_path / "adapters"

        def _tier(name: str, keys: list[str]) -> tuple[Path, str]:
            root = adapter_root / name
            registry_bytes = write_noncanonical_registry(root, keys)
            digest = _sha256_hex(registry_bytes)
            slot = root / "20260101-000000"
            write_slot_files(slot)
            write_raw_meta(slot, v4_train_meta_dict(name=name, registry_sha256=digest))
            return slot, digest

        episodic_slot, _ = _tier("episodic", ["k1"])
        semantic_slot, _ = _tier("semantic", ["k2"])
        procedural_slot, _ = _tier("procedural", ["k3"])

        interim_root = adapter_root / "episodic" / "interim_20260101T0000"
        interim_registry = write_noncanonical_registry(interim_root, ["k4"])
        interim_digest = _sha256_hex(interim_registry)
        interim_slot = interim_root / "20260101-010000"
        write_slot_files(interim_slot)
        write_raw_meta(
            interim_slot,
            v4_train_meta_dict(
                name="episodic_interim_20260101T0000", registry_sha256=interim_digest
            ),
        )

        donor_root = adapter_root / f"{DONOR_STORE_PREFIX}r8-a16-2mod-deadbeef-btest1234"
        donor_slot = donor_root / "20260101-020000"
        write_slot_files(donor_slot)
        write_raw_meta(donor_slot, v4_train_meta_dict(name="donor_build", registry_sha256=""))

        results = migrate(adapter_root, dry_run=False)

        labels = {r.label for r in results}
        assert {"episodic", "semantic", "procedural"} <= labels
        assert any("interim" in label for label in labels)
        assert any(label.startswith(DONOR_STORE_PREFIX) for label in labels)

        for slot in (episodic_slot, semantic_slot, procedural_slot, interim_slot, donor_slot):
            assert read_manifest(slot).schema_version == MANIFEST_SCHEMA_VERSION


class TestIdempotence:
    def test_migration_is_idempotent_on_an_already_current_tree(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_canonical_registry(tier_root, ["k1", "k2"])
        digest = _sha256_hex(registry_bytes)

        slot = tier_root / "20260101-000000"
        write_slot_files(slot, weight_bytes=b"already-current-weights")
        manifest = make_train_manifest(
            name="episodic",
            registry_sha256=digest,
            key_count=2,
            payload_sha256=_sha256_hex(b"already-current-weights"),
        )
        write_manifest(slot, manifest)

        before = _snapshot(adapter_root)
        results_1 = migrate(adapter_root, dry_run=False)
        after_1 = _snapshot(adapter_root)
        results_2 = migrate(adapter_root, dry_run=False)
        after_2 = _snapshot(adapter_root)

        assert all(r.action in ("already_migrated", "absent") for r in results_1)
        assert all(r.action in ("already_migrated", "absent") for r in results_2)
        assert after_1 == before, "an already-current tree must not be rewritten at all"
        assert after_2 == before


class TestStopsLoudly:
    def test_migration_stops_loudly_on_a_slot_it_cannot_read_or_decrypt(self, tmp_path):
        # Sub-case 1: unreadable (corrupt JSON) meta.json on an otherwise
        # complete slot.
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        write_canonical_registry(tier_root, ["k1"])
        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        (slot / "meta.json").write_text("{not valid json", encoding="utf-8")

        with pytest.raises(TierMigrationBlocked) as exc_info:
            migrate(adapter_root, dry_run=False)
        assert exc_info.value.tier == "episodic"

        # Sub-case 2: a registry that is age-encrypted but cannot be
        # decrypted (no daily identity loaded in this test process).
        adapter_root_2 = tmp_path / "adapters_2"
        tier_root_2 = adapter_root_2 / "episodic"
        tier_root_2.mkdir(parents=True)
        from paramem.backup.age_envelope import AGE_MAGIC

        (tier_root_2 / "indexed_key_registry.json").write_bytes(AGE_MAGIC + b"not-real-ciphertext")
        slot_2 = tier_root_2 / "20260101-000000"
        write_slot_files(slot_2)
        write_raw_meta(slot_2, v4_train_meta_dict(name="episodic", registry_sha256="whatever"))

        with pytest.raises(TierMigrationBlocked) as exc_info_2:
            migrate(adapter_root_2, dry_run=False)
        assert exc_info_2.value.tier == "episodic"


class TestMissingRequiredTopLevelFieldStopsLoudly:
    def test_migration_stops_loudly_on_a_meta_json_missing_a_required_top_level_field(
        self, tmp_path
    ):
        """One representative case (``trained_at``) -- ``name``, ``key_count``
        and ``registry_sha256`` are hard-required the same way; a v1-v4
        meta.json was never legitimately missing any of the four."""
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_noncanonical_registry(tier_root, ["k1"])
        old_digest = _sha256_hex(registry_bytes)

        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        raw = v4_train_meta_dict(name="episodic", registry_sha256=old_digest)
        del raw["trained_at"]
        write_raw_meta(slot, raw)

        with pytest.raises(TierMigrationBlocked, match="trained_at"):
            migrate(adapter_root, dry_run=False)


class TestSkipsPartialScratch:
    def test_migration_skips_a_partial_slot_boot_would_reap(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_noncanonical_registry(tier_root, ["k1"])
        old_digest = _sha256_hex(registry_bytes)

        complete_slot = tier_root / "20260101-000000"
        write_slot_files(complete_slot)
        write_raw_meta(
            complete_slot, v4_train_meta_dict(name="episodic", registry_sha256=old_digest)
        )

        # A partial slot: has a meta.json (so it IS a slot candidate) but is
        # missing adapter_model.safetensors -- the same scratch predicate
        # cleanup_partial_slots applies at boot.
        partial_slot = tier_root / "20260102-000000"
        partial_slot.mkdir(parents=True)
        (partial_slot / "adapter_config.json").write_bytes(b"{}")
        write_raw_meta(
            partial_slot, v4_train_meta_dict(name="episodic", registry_sha256=old_digest)
        )
        partial_meta_before = (partial_slot / "meta.json").read_bytes()

        results = migrate(adapter_root, dry_run=False)

        assert read_manifest(complete_slot).schema_version == MANIFEST_SCHEMA_VERSION
        # The partial slot must be left completely untouched.
        assert (partial_slot / "meta.json").read_bytes() == partial_meta_before
        migrated_names = {name for r in results for name in r.migrated_slots}
        assert partial_slot.name not in migrated_names


class TestAllScratchTierStopsLoudly:
    def test_migration_stops_on_a_tier_whose_only_candidates_are_partial_scratch(self, tmp_path):
        """A tier with a keyed registry but ONLY partial-trained scratch
        slots (no complete slot at all): partial scratch is excluded from
        the bound-slot match set regardless of what registry_sha256 it
        carries, so this is NOT the zero-candidate registry-only case --
        it is a tier that is already unbound before migration."""
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_noncanonical_registry(tier_root, ["k1"])
        old_digest = _sha256_hex(registry_bytes)

        partial_slot = tier_root / "20260101-000000"
        partial_slot.mkdir(parents=True)
        (partial_slot / "adapter_config.json").write_bytes(b"{}")
        write_raw_meta(
            partial_slot, v4_train_meta_dict(name="episodic", registry_sha256=old_digest)
        )

        with pytest.raises(TierMigrationBlocked) as exc_info:
            migrate(adapter_root, dry_run=False)
        assert exc_info.value.tier == "episodic"


class TestUnresolvableBoundSlot:
    def test_migration_stops_on_a_tier_whose_bound_slot_cannot_be_resolved(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        write_noncanonical_registry(tier_root, ["k1"])

        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        # Deliberately wrong -- does not match the tier's real registry
        # digest under either the old or the new (re-serialized) form.
        write_raw_meta(slot, v4_train_meta_dict(name="episodic", registry_sha256="f" * 64))

        with pytest.raises(TierMigrationBlocked) as exc_info:
            migrate(adapter_root, dry_run=False)
        assert exc_info.value.tier == "episodic"
        assert "already unbound" in str(exc_info.value)


class TestInventsNoManifestForSimulateTree:
    def test_migration_invents_no_manifest_for_a_simulate_tree(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "semantic"
        write_noncanonical_registry(tier_root, ["graph1", "graph2"])

        from paramem.adapters.manifest import count_slot_candidates

        assert count_slot_candidates(tier_root) == 0

        results = migrate(adapter_root, dry_run=False)

        assert count_slot_candidates(tier_root) == 0, (
            "the pre-upgrade simulate shape must not have a manifest invented for it"
        )
        semantic_results = [r for r in results if r.label == "semantic"]
        assert semantic_results
        assert semantic_results[0].action in (
            "migrated_registry_only",
            "already_migrated_registry_only",
        )


class TestMtimeTieBreakPreserved:
    def test_migration_keeps_the_same_slot_bound_when_two_slots_share_a_registry_digest(
        self, tmp_path
    ):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_noncanonical_registry(tier_root, ["k1"])
        old_digest = _sha256_hex(registry_bytes)
        new_digest = _sha256_hex(KeyRegistry.load_from_bytes(registry_bytes).save_bytes())

        slot_a = tier_root / "20260101-000000"
        slot_b = tier_root / "20260102-000000"
        for slot in (slot_a, slot_b):
            write_slot_files(slot)
            write_raw_meta(slot, v4_train_meta_dict(name="episodic", registry_sha256=old_digest))

        t_old = 1_700_000_000.0
        t_new = 1_700_100_000.0
        os.utime(slot_a, (t_old, t_old))
        os.utime(slot_b, (t_new, t_new))

        # Pre-migration, `read_manifest` (and therefore `find_live_slot`)
        # cannot read the prior (schema_version=4) shape at all -- exactly
        # the "an unmigrated slot reads as unreadable" consequence the plan
        # documents. The pre-migration tie-break is checked directly against
        # the slot directories' own mtimes instead.
        assert slot_b.stat().st_mtime > slot_a.stat().st_mtime, (
            "fixture must establish slot_b as the newer slot before migration"
        )

        migrate(adapter_root, dry_run=False)

        post_migration_live = find_live_slot(tier_root, new_digest)
        assert post_migration_live == slot_b, (
            "the migration must preserve slot directory mtimes so the SAME "
            "slot resolves live before and after"
        )


class TestConvergesOnRerun:
    def test_a_partially_migrated_tier_converges_on_re_run(self, tmp_path):
        adapter_root = tmp_path / "adapters"

        # --- Direction 1: manifest already rewritten to v5, registry not
        # yet re-serialized (simulates a crash between step 3 and step 4). ---
        episodic_root = adapter_root / "episodic"
        raw_registry = write_noncanonical_registry(episodic_root, ["k1", "k2"])
        old_digest = _sha256_hex(raw_registry)
        new_registry_bytes = KeyRegistry.load_from_bytes(raw_registry).save_bytes()
        new_digest = _sha256_hex(new_registry_bytes)
        assert old_digest != new_digest, "fixture must exercise a genuine re-serialization"

        episodic_slot = episodic_root / "20260101-000000"
        weight_bytes = b"episodic weights"
        write_slot_files(episodic_slot, weight_bytes=weight_bytes)
        write_manifest(
            episodic_slot,
            make_train_manifest(
                name="episodic",
                registry_sha256=new_digest,
                key_count=2,
                payload_sha256=_sha256_hex(weight_bytes),
            ),
        )

        # --- Direction 2: registry already at its final (canonical) bytes,
        # one sibling slot in the tier not yet stamped to v5. ---
        semantic_root = adapter_root / "semantic"
        canonical_bytes = write_canonical_registry(semantic_root, ["k3"])
        semantic_digest = _sha256_hex(canonical_bytes)

        semantic_bound = semantic_root / "20260101-000000"
        semantic_weight_bytes = b"semantic weights bound"
        write_slot_files(semantic_bound, weight_bytes=semantic_weight_bytes)
        write_manifest(
            semantic_bound,
            make_train_manifest(
                name="semantic",
                registry_sha256=semantic_digest,
                key_count=1,
                payload_sha256=_sha256_hex(semantic_weight_bytes),
            ),
        )

        semantic_sibling = semantic_root / "20260102-000000"
        write_slot_files(semantic_sibling)
        write_raw_meta(
            semantic_sibling,
            v4_train_meta_dict(name="semantic", registry_sha256="f" * 64),
        )

        migrate(adapter_root, dry_run=False)

        # Direction 1 converged: registry now matches the manifest's digest.
        assert tier_registry_sha256(episodic_root) == new_digest
        assert find_live_slot(episodic_root, new_digest) == episodic_slot

        # Direction 2 converged: the sibling is now readable at v5, and the
        # tier still boots publishable off its bound slot.
        assert read_manifest(semantic_sibling).schema_version == MANIFEST_SCHEMA_VERSION
        assert find_live_slot(semantic_root, semantic_digest) == semantic_bound

        # A further re-run is a true no-op -- fully converged.
        before = _snapshot(adapter_root)
        migrate(adapter_root, dry_run=False)
        after = _snapshot(adapter_root)
        assert after == before


class TestDryRun:
    def test_migration_dry_run_writes_nothing(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_noncanonical_registry(tier_root, ["k1"])
        old_digest = _sha256_hex(registry_bytes)

        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        write_raw_meta(slot, v4_train_meta_dict(name="episodic", registry_sha256=old_digest))

        before = _snapshot(adapter_root)
        results = migrate(adapter_root, dry_run=True)
        after = _snapshot(adapter_root)

        assert after == before, "a dry run must never touch the filesystem"
        assert any(r.action.startswith("would_migrate") for r in results)

    def test_dry_run_raises_on_a_malformed_tier_the_same_as_a_real_run(self, tmp_path):
        """Dry-run must exercise the SAME per-tier validation a real run
        does (required-field checks, payload digest computation) and skip
        only the writes -- an operator who sees a clean 'would migrate'
        report must not then hit a stop partway through the real run."""
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        registry_bytes = write_noncanonical_registry(tier_root, ["k1"])
        old_digest = _sha256_hex(registry_bytes)

        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        raw = v4_train_meta_dict(name="episodic", registry_sha256=old_digest)
        del raw["name"]
        write_raw_meta(slot, raw)

        with pytest.raises(TierMigrationBlocked):
            migrate(adapter_root, dry_run=True)


class TestNeverParsesPreMigrationInputWithCurrentSchemaReaders:
    def test_migration_never_parses_pre_migration_input_with_the_current_schema_readers(self):
        """Structural pin (the tests/test_extraction_pipeline_guard.py shape):
        the migration script must never call `read_manifest` (current-
        schema-only) or `KeyRegistry.load` (file-path, current-shape-only)
        on pre-migration input -- it uses raw `json.loads` / `read_maybe_encrypted`
        plus `KeyRegistry.load_from_bytes` instead."""
        tree = ast.parse(_SCRIPT_PATH.read_text())

        offenders: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "read_manifest":
                offenders.append(("read_manifest", node.lineno))
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and isinstance(func.value, ast.Name)
                and func.value.id == "KeyRegistry"
            ):
                offenders.append(("KeyRegistry.load", node.lineno))

        assert offenders == [], (
            "stamp_slot_manifests_v5.py must never call the current-schema-only "
            f"readers on pre-migration input: {offenders}"
        )

        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "read_manifest" not in imported_names, (
            "the script must not even import read_manifest -- the reader for "
            "the shape it exists to migrate away from"
        )


# ---------------------------------------------------------------------------
# The registry pass -- _migrate_registry_bytes -- direct unit coverage
# against the pre-change ("stale" as a dict of per-id records) shape.
# ---------------------------------------------------------------------------


class TestMigrateRegistryBytes:
    def test_the_migration_rewrites_every_tier_registry_into_the_one_shape(self):
        """Records -> sorted bare ids; stale_since and withheld-id
        fingerprints (whether embedded in the record or duplicated in the
        top-level 'simhash' section) are all gone from the migrated payload."""
        raw = json.dumps(
            {
                "active_keys": ["graph1", "graph2"],
                "stale": {
                    "graph9": {
                        "stale_since": "2026-01-01T00:00:00Z",
                        "simhash": 777,
                    },
                    "graph3": {},
                },
                # A "simhash"-section duplicate: graph9 also carries a
                # top-level fingerprint entry, independent of its embedded
                # record fingerprint above.
                "simhash": {"graph1": 1, "graph2": 2, "graph9": 888},
            }
        ).encode("utf-8")

        registry_obj, migrated_bytes = _migrate_registry_bytes(
            raw, tier="episodic", registry_path=Path("indexed_key_registry.json")
        )

        data = json.loads(migrated_bytes)
        assert data["stale"] == ["graph3", "graph9"]
        assert data["simhash"] == {"graph1": 1, "graph2": 2}
        assert registry_obj.list_active() == ["graph1", "graph2"]
        assert registry_obj.list_stale() == ["graph3", "graph9"]
        assert registry_obj.simhash_for("graph9") is None

    def test_a_migrated_registry_round_trips_byte_stably(self):
        raw = json.dumps(
            {
                "active_keys": ["graph1"],
                "stale": {"graph2": {"stale_since": "2026-01-01T00:00:00Z"}},
                "simhash": {"graph1": 5},
            }
        ).encode("utf-8")

        _registry_obj, migrated_bytes = _migrate_registry_bytes(
            raw, tier="episodic", registry_path=Path("indexed_key_registry.json")
        )

        reparsed = KeyRegistry.load_from_bytes(migrated_bytes)
        assert reparsed.save_bytes() == migrated_bytes

    def test_a_registry_with_no_stale_section_migrates_to_an_empty_one(self):
        """A legitimate pre-stale-extension file -- 'stale' absent entirely
        -- migrates to an empty list, not an error."""
        raw = json.dumps({"active_keys": ["graph1"], "simhash": {"graph1": 1}}).encode("utf-8")

        _registry_obj, migrated_bytes = _migrate_registry_bytes(
            raw, tier="episodic", registry_path=Path("indexed_key_registry.json")
        )

        assert json.loads(migrated_bytes)["stale"] == []

    def test_an_id_in_both_the_active_list_and_the_stale_section_fails_loud(self):
        """Reachable today because add() only guards the active list --
        dropping the withheld-side fingerprint would leave an active key
        with none, so this is a hard stop, not an auto-heal."""
        raw = json.dumps(
            {
                "active_keys": ["graph1"],
                "stale": {"graph1": {"stale_since": "2026-01-01T00:00:00Z"}},
                "simhash": {},
            }
        ).encode("utf-8")

        with pytest.raises(TierMigrationBlocked, match="graph1"):
            _migrate_registry_bytes(
                raw, tier="episodic", registry_path=Path("indexed_key_registry.json")
            )

    def test_a_non_int_fingerprint_fails_loud(self):
        raw = json.dumps(
            {"active_keys": ["graph1"], "stale": {}, "simhash": {"graph1": "not-an-int"}}
        ).encode("utf-8")

        with pytest.raises(TierMigrationBlocked):
            _migrate_registry_bytes(
                raw, tier="episodic", registry_path=Path("indexed_key_registry.json")
            )

    def test_the_registry_pass_never_parses_pre_migration_input_through_the_new_reader(self):
        """Structural: _migrate_registry_bytes reads the pre-migration
        payload as raw JSON and calls KeyRegistry.load_from_bytes exactly
        once, on its OWN migrated output -- never on the raw parameter."""
        tree = ast.parse(_SCRIPT_PATH.read_text())
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_migrate_registry_bytes"
        )

        calls: list[ast.Call] = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_from_bytes"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "KeyRegistry"
        ]
        assert len(calls) == 1, (
            f"expected exactly one KeyRegistry.load_from_bytes call, found {len(calls)}"
        )
        arg = calls[0].args[0]
        assert isinstance(arg, ast.Name) and arg.id != "raw_bytes", (
            "KeyRegistry.load_from_bytes must be called on the migrated bytes, "
            f"not the raw parameter -- got argument {ast.dump(arg)}"
        )


class TestMigrateTierWithPreChangeShape:
    """`migrate()` end-to-end against a genuine pre-change ('stale' as a
    dict of per-id records) tier -- follows the same fixture pattern as
    TestFingerprintAndBindingFieldsCarryOver / TestIdempotence, using
    write_pre_change_stale_registry instead of write_noncanonical_registry."""

    def test_the_migration_restamps_each_migrated_tiers_bound_slot(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        raw_bytes = write_pre_change_stale_registry(
            tier_root,
            active_keys=["graph1"],
            stale_records={"graph2": {"stale_since": "2026-01-01T00:00:00Z"}},
        )
        old_digest = _sha256_hex(raw_bytes)
        new_digest = _sha256_hex(
            KeyRegistry.load_from_bytes(
                _migrate_registry_bytes(raw_bytes, tier="episodic", registry_path=tier_root / "x")[
                    1
                ]
            ).save_bytes()
        )

        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        write_raw_meta(slot, v4_train_meta_dict(name="episodic", registry_sha256=old_digest))

        migrate(adapter_root, dry_run=False)

        manifest = read_manifest(slot)
        assert manifest.registry_sha256 == new_digest

        on_disk = (tier_root / "indexed_key_registry.json").read_bytes()
        migrated_data = json.loads(on_disk)
        assert migrated_data["stale"] == ["graph2"]
        assert isinstance(migrated_data["stale"], list)

    def test_a_second_run_of_the_migration_changes_no_byte_and_restamps_nothing(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        write_pre_change_stale_registry(
            tier_root,
            active_keys=["graph1"],
            stale_records={"graph2": {"stale_since": "2026-01-01T00:00:00Z"}},
        )
        raw_bytes = (tier_root / "indexed_key_registry.json").read_bytes()
        old_digest = _sha256_hex(raw_bytes)

        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        write_raw_meta(slot, v4_train_meta_dict(name="episodic", registry_sha256=old_digest))

        results_1 = migrate(adapter_root, dry_run=False)
        assert any(r.action == "migrated" for r in results_1)

        after_1 = _snapshot(adapter_root)
        results_2 = migrate(adapter_root, dry_run=False)
        after_2 = _snapshot(adapter_root)

        episodic_result = next(r for r in results_2 if r.label == "episodic")
        assert episodic_result.action == "already_migrated"
        assert episodic_result.migrated_slots == ()
        assert after_2 == after_1, "a re-run on an already-migrated tier must change nothing"

    def test_a_tier_with_an_already_current_schema_bound_manifest_still_migrates(self, tmp_path):
        """The proven blocker: the bound slot's manifest is ALREADY at the
        current schema version, but the tier's registry is still
        pre-change-shape. Restamping must not be gated on schema_version
        parity -- idempotence is binding-inclusive (module docstring): a
        bound slot not yet carrying the tier's final registry digest is
        restamped regardless of its own schema_version. Without the fix
        this slot is excluded from `pending` (its schema_version already
        matches current), so the registry pass still rewrites the registry
        to `new_digest` but the bound manifest is left holding `old_digest`
        -- and since the pre-migration bytes are gone from disk, no re-run
        can ever repair the binding again."""
        adapter_root = tmp_path / "adapters"
        tier_root = adapter_root / "episodic"
        raw_bytes = write_pre_change_stale_registry(
            tier_root,
            active_keys=["graph1"],
            stale_records={"graph2": {"stale_since": "2026-01-01T00:00:00Z"}},
        )
        old_digest = _sha256_hex(raw_bytes)
        new_digest = _sha256_hex(
            KeyRegistry.load_from_bytes(
                _migrate_registry_bytes(raw_bytes, tier="episodic", registry_path=tier_root / "x")[
                    1
                ]
            ).save_bytes()
        )
        assert old_digest != new_digest, "fixture must exercise a genuine re-serialization"

        slot = tier_root / "20260101-000000"
        weight_bytes = b"already-v5-weights"
        write_slot_files(slot, weight_bytes=weight_bytes)
        write_manifest(
            slot,
            make_train_manifest(
                name="episodic",
                registry_sha256=old_digest,
                key_count=1,
                payload_sha256=_sha256_hex(weight_bytes),
            ),
        )
        assert read_manifest(slot).schema_version == MANIFEST_SCHEMA_VERSION, (
            "fixture must start with an already-current-schema bound manifest"
        )

        migrate(adapter_root, dry_run=False)

        manifest = read_manifest(slot)
        assert manifest.registry_sha256 == new_digest, (
            "the bound slot must be restamped to the rewritten registry's digest "
            "even though its schema_version was already current"
        )
        on_disk = (tier_root / "indexed_key_registry.json").read_bytes()
        migrated_data = json.loads(on_disk)
        assert migrated_data["stale"] == ["graph2"]
        assert isinstance(migrated_data["stale"], list)

        assert find_live_slot(tier_root, new_digest) == slot, (
            "the tier must bind after migration -- an unrepaired unbinding is the "
            "proven blocker this fix closes"
        )

        # A second run converges to a true no-op: the registry is already
        # rewritten and the bound slot already carries its digest.
        before = _snapshot(adapter_root)
        results_2 = migrate(adapter_root, dry_run=False)
        after = _snapshot(adapter_root)
        episodic_result = next(r for r in results_2 if r.label == "episodic")
        assert episodic_result.action == "already_migrated"
        assert after == before


def _fake_systemctl_run(returncode: int):
    """A ``systemctl.run``-shaped stub reporting *returncode* for every call."""

    def _run(*args, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout="", stderr="")

    return _run


class TestMainLivenessGuard:
    """``main()``'s own liveness guard -- routed through the project's one
    ``systemctl`` transport seam (fix for the direct ``subprocess.run``
    call this script used to make). The autouse host-isolation guard
    already stubs :func:`paramem.utils.systemctl.run`; these tests override
    its return value to report the server active/inactive."""

    def test_main_refuses_unconditionally_when_the_server_reports_active(
        self, tmp_path, monkeypatch
    ):
        from paramem.utils import systemctl

        monkeypatch.setattr(systemctl, "run", _fake_systemctl_run(returncode=0))

        rc = main(["--adapter-root", str(tmp_path / "adapters")])

        assert rc == 1

    def test_main_proceeds_when_the_server_reports_inactive(self, tmp_path, monkeypatch):
        import scripts.migrate.stamp_slot_manifests_v5 as script_mod
        from paramem.utils import systemctl

        monkeypatch.setattr(systemctl, "run", _fake_systemctl_run(returncode=3))
        # This host's own pgrep patterns can legitimately match a real
        # process (e.g. the project's own running server) -- isolate this
        # test from host process state, since it exercises the systemctl
        # seam only, not the pgrep half of the guard.
        monkeypatch.setattr(script_mod, "_pgrep_alive", lambda patterns: [])
        adapter_root = tmp_path / "adapters"
        adapter_root.mkdir()

        rc = main(["--adapter-root", str(adapter_root)])

        assert rc == 0
