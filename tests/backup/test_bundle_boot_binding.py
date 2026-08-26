"""write_bundle -> restore_bundle -> boot binding verification round trip.

Exercises the uniform slot envelope end to end: capture a real written slot
(train or simulate payload), restore it into a clean tree, and confirm the
result is exactly what boot binding verification
(:func:`paramem.adapters.registry_binding.verify_tier_binding`) expects --
one candidate slot carrying ``meta.json`` plus its payload, no legacy
tier-root ``graph.json``.  No existing backup test file combines
``write_bundle`` + ``restore_bundle`` + registry-binding verification
together, hence a new home rather than folding into an unrelated file.

No GPU required -- every slot here is written with stub payload bytes
through the real :func:`~paramem.adapters.slot.write_slot` envelope.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from paramem.adapters.manifest import (
    count_slot_candidates,
    find_live_slot,
    tier_registry_sha256,
)
from paramem.adapters.registry_binding import VERIFIED, verify_tier_binding
from paramem.adapters.slot import write_slot
from paramem.backup.backup import (
    SLOT_DURABLE_FILES,
    TIER_ROOT_FILES_ORDERED,
    read_bundle_manifest,
    restore_bundle,
    write_bundle,
)
from paramem.backup.integrity import cleanup_partial_slots
from paramem.training.key_registry import KeyRegistry
from tests._fold_fixtures import _write_graph
from tests._manifest_fixtures import make_train_manifest, write_slot_files


def _write_registry_and_rows(
    tier_root: Path,
    keys: "str | list[str]",
    *,
    stale: "list[str] | None" = None,
    fingerprints: "dict[str, int] | None" = None,
) -> KeyRegistry:
    """Write a canonical registry -- one or more active keys, with an
    optional withheld subset -- plus matching ``key_metadata.json`` rows for
    every KNOWN key (active union stale) at *tier_root* -- the
    every-known-key-has-a-row invariant a bundle capture enforces.

    *keys* is a single key (the original one-key call shape) or a list of
    keys. *stale* names the subset (a subset of *keys*) withheld via
    :meth:`KeyRegistry.stale` -- the real API, which drops the withheld
    key's fingerprint on the active->stale transition by design; a withheld
    key's bookkeeping row is retained regardless, via :meth:`list_known`'s
    active-union-stale enumeration. *fingerprints* supplies a real per-key
    SimHash fingerprint (default ``12345`` for every key, matching the
    original single-key call's fixed value) -- set on every key BEFORE any
    ``stale()`` call, so the fingerprint-drop-on-withhold contract is
    exercised for a withheld key rather than skipped.

    Returns the registry so a caller (e.g. :func:`_write_real_simulate_slot`)
    can reuse it to write a matching bound slot.
    """
    key_list = [keys] if isinstance(keys, str) else list(keys)
    stale = stale or []
    fingerprints = fingerprints or {}
    registry = KeyRegistry()
    for key in key_list:
        registry.add(key)
        registry.set_simhash(key, fingerprints.get(key, 12345))
    for key in stale:
        registry.stale(key)
    tier_root.mkdir(parents=True, exist_ok=True)
    registry.save(tier_root / "indexed_key_registry.json")
    rows = {
        key: {
            "speaker_id": "speaker0",
            "relation_type": "factual",
            "reinforcement_count": 1,
            "last_reinforced_cycle": 0,
            "last_seen": "2026-01-01T00:00:00Z",
            "first_seen": "2026-01-01T00:00:00Z",
            "promoted": False,
        }
        for key in registry.list_known()
    }
    (tier_root / "key_metadata.json").write_text(
        json.dumps({"tier_cycle": 0, "keys": rows}),
        encoding="utf-8",
    )
    return registry


def _write_real_simulate_slot(
    tier_root: Path,
    keys: "str | list[str]",
    *,
    stale: "list[str] | None" = None,
    entries: "dict[str, dict] | None" = None,
) -> Path:
    """Registry + rows (one or more keys, optional withheld subset) + a REAL
    written simulate (graph.json) slot carrying every ACTIVE key's fact,
    bound to that registry, through the same envelope production uses.

    *entries* supplies each key's ``{subject, predicate, object}`` content
    (default: the fixed alice/lives_in/berlin fact, matching the original
    single-key call's content). Each key's registered fingerprint is the
    REAL :func:`~paramem.memory.entry.entry_simhash` of its own fact, so a
    probe against this slot verifies and lands the entry instead of being
    gated out by a placeholder fingerprint. A withheld (*stale*) key's fact
    is used only to compute the fingerprint it briefly carries before
    ``KeyRegistry.stale`` drops it -- it is NOT written into ``graph.json``,
    matching production (a withheld key is excluded from serving content)
    and keeping the slot manifest's ``key_count`` equal to the registry's
    active count, as :func:`~paramem.adapters.registry_binding.verify_tier_binding`
    requires.
    """
    from paramem.memory.entry import entry_simhash

    key_list = [keys] if isinstance(keys, str) else list(keys)
    stale = stale or []
    entries = entries or {}
    default_fact = {"subject": "alice", "predicate": "lives_in", "object": "berlin"}

    quads = []
    fingerprints: dict[str, int] = {}
    for key in key_list:
        fact = entries.get(key, default_fact)
        entry = {"key": key, **fact, "speaker_id": "speaker0"}
        fingerprints[key] = entry_simhash(entry)
        if key not in stale:
            quads.append(entry)

    _write_registry_and_rows(tier_root, key_list, stale=stale, fingerprints=fingerprints)
    _write_graph(tier_root, quads)
    return find_live_slot(tier_root, tier_registry_sha256(tier_root))


def _write_real_train_slot(tier_root: Path, key: str) -> Path:
    """Registry + rows + a REAL written train (adapter weights) slot, bound to
    that registry, through the same envelope production uses."""
    _write_registry_and_rows(tier_root, key)
    registry_hash = tier_registry_sha256(tier_root)
    manifest = make_train_manifest(name="episodic", registry_sha256=registry_hash, key_count=1)
    write_slot(
        tier_root, manifest=manifest, write_payload=lambda pending: write_slot_files(pending)
    )
    return find_live_slot(tier_root, registry_hash)


class TestSlotDurableFilesTierRootFilesDisjoint:
    """The plan's own risk note: a filename claimed by both sets would be
    ambiguous between a slot payload and a tier-root file."""

    def test_slot_durable_files_and_tier_root_files_are_disjoint(self) -> None:
        assert set(SLOT_DURABLE_FILES).isdisjoint(TIER_ROOT_FILES_ORDERED)


class TestCaptureRestoreBootBinding:
    """Capture a real written tier, restore into a clean tree, verify boot
    binding sees exactly the uniform slot shape -- no tier-root payload
    file, one candidate slot, VERIFIED."""

    def test_capture_restore_boot_binds_a_graph_payload(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        episodic_dir = src / "adapters" / "episodic"
        _write_real_simulate_slot(episodic_dir, "graph1")

        bundle_slot = write_bundle(
            config_path=tmp_path / "no-such-config.yaml",
            adapter_dirs={"episodic": episodic_dir},
            backups_root=src / "backups",
            backups_cfg=None,
            meta_fields={"tier": "manual", "label": "graph-capture"},
        )

        target = tmp_path / "target"
        restore_bundle(
            bundle_slot_dir=bundle_slot,
            data_dir=target,
            config_path=target / "server.yaml",
        )

        target_episodic = target / "adapters" / "episodic"
        # Exactly one candidate slot, carrying meta.json + graph.json.
        slot_dirs = [
            p for p in target_episodic.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
        assert len(slot_dirs) == 1
        restored_slot = slot_dirs[0]
        assert (restored_slot / "meta.json").exists()
        assert (restored_slot / "graph.json").exists()

        # No tier-root graph.json -- graph.json is a SLOT file now.
        assert not (target_episodic / "graph.json").exists()

        binding = verify_tier_binding("episodic", target_episodic)
        assert binding.status == VERIFIED
        assert binding.slot == restored_slot

    def test_capture_restore_boot_binds_a_weight_payload(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        episodic_dir = src / "adapters" / "episodic"
        _write_real_train_slot(episodic_dir, "graph1")

        bundle_slot = write_bundle(
            config_path=tmp_path / "no-such-config.yaml",
            adapter_dirs={"episodic": episodic_dir},
            backups_root=src / "backups",
            backups_cfg=None,
            meta_fields={"tier": "manual", "label": "weight-capture"},
        )

        target = tmp_path / "target"
        restore_bundle(
            bundle_slot_dir=bundle_slot,
            data_dir=target,
            config_path=target / "server.yaml",
        )

        target_episodic = target / "adapters" / "episodic"
        slot_dirs = [
            p for p in target_episodic.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
        assert len(slot_dirs) == 1
        restored_slot = slot_dirs[0]
        assert (restored_slot / "meta.json").exists()
        assert (restored_slot / "adapter_model.safetensors").exists()
        assert (restored_slot / "adapter_config.json").exists()
        assert not (target_episodic / "graph.json").exists()

        binding = verify_tier_binding("episodic", target_episodic)
        assert binding.status == VERIFIED
        assert binding.slot == restored_slot


class TestRestoringAPreUpgradeSimulateBundle:
    """A legacy bundle recording a simulate tier's ``graph.json`` as a plain
    tier-root file (no accompanying slot ``meta.json``).  Restoring it lands
    that file inside a
    freshly allocated slot dir (``SLOT_DURABLE_FILES`` classifies by
    filename alone) that carries no ``meta.json`` -- ``cleanup_partial_slots``
    removes it as scratch, and the tier is left with active keys but no
    slot -- the keyed-slotless shape :data:`~paramem.adapters.registry_binding.KEYS_WITHOUT_SLOT`
    exists to name.
    """

    def _write_pre_upgrade_bundle(self, bundle_dir: Path, key: str) -> Path:
        """Hand-construct a bundle.meta.json + files in the legacy shape:
        ``adapters/episodic/graph.json`` captured as a bare tier-root
        file, no ``adapters/episodic/meta.json`` slot entry -- the shape a
        bundle carries when ``graph.json`` is recorded as a bare tier-root
        file instead of as a slot member. ``bundle_schema_version`` (the
        outer bundle format) is independent of the per-slot manifest
        schema, so it stays current."""
        import hashlib

        from paramem.backup.types import BUNDLE_SCHEMA_VERSION

        bundle_dir.mkdir(parents=True, exist_ok=True)

        registry = KeyRegistry()
        registry.add(key)
        registry.set_simhash(key, 999)
        registry_bytes = registry.save_bytes()

        rows_bytes = json.dumps(
            {
                "tier_cycle": 0,
                "keys": {
                    key: {
                        "speaker_id": "speaker0",
                        "relation_type": "factual",
                        "reinforcement_count": 1,
                        "last_reinforced_cycle": 0,
                        "last_seen": "2026-01-01T00:00:00Z",
                        "first_seen": "2026-01-01T00:00:00Z",
                        "promoted": False,
                    }
                },
            }
        ).encode("utf-8")

        graph_bytes = json.dumps({"directed": True, "multigraph": True, "graph": {}}).encode(
            "utf-8"
        )

        files = []
        for rel_path, content in (
            ("adapters/episodic/indexed_key_registry.json", registry_bytes),
            ("adapters/episodic/key_metadata.json", rows_bytes),
            ("adapters/episodic/graph.json", graph_bytes),
        ):
            dst = bundle_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(content)
            files.append(
                {
                    "path": rel_path,
                    "content_sha256": hashlib.sha256(content).hexdigest(),
                    "encrypted": False,
                    "size_bytes": len(content),
                }
            )

        manifest_dict = {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": "2026-01-01T00:00:00Z",
            "tier": "manual",
            "label": "pre_upgrade_simulate",
            "base_model": {},
            "files": files,
            "adapters": {
                "episodic": {
                    "slot_source": "",
                    "registry_sha256": "",
                    "key_count": "unknown",
                    "indexed_key_registry_present": True,
                    "keyed_pairs_present": False,
                    "weightless_cause": None,
                }
            },
            "excluded": [],
        }
        (bundle_dir / "bundle.meta.json").write_text(json.dumps(manifest_dict), encoding="utf-8")
        return bundle_dir

    def test_restoring_a_pre_upgrade_simulate_bundle_quarantines_loudly(
        self, tmp_path: Path
    ) -> None:
        bundle_dir = self._write_pre_upgrade_bundle(tmp_path / "bundle", "graph1")

        target = tmp_path / "target"
        restore_bundle(
            bundle_slot_dir=bundle_dir,
            data_dir=target,
            config_path=target / "server.yaml",
        )

        target_episodic = target / "adapters" / "episodic"
        # The legacy graph.json landed inside a freshly allocated slot dir
        # with no meta.json -- a candidate by neither name.
        allocated_slots = [
            p for p in target_episodic.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
        assert len(allocated_slots) == 1
        assert (allocated_slots[0] / "graph.json").exists()
        assert not (allocated_slots[0] / "meta.json").exists()

        removed = cleanup_partial_slots(target / "adapters")
        assert len(removed) == 1
        assert removed[0]["tier"] == "episodic"
        assert not allocated_slots[0].exists(), "the graph-only slot is removed as scratch"

        # The tier keeps its active-keyed registry but now has zero slot
        # candidates at all.
        assert count_slot_candidates(target_episodic) == 0
        registry = KeyRegistry.load(target_episodic / "indexed_key_registry.json")
        assert registry.list_active() == ["graph1"]

        binding = verify_tier_binding("episodic", target_episodic)
        from paramem.adapters.registry_binding import KEYS_WITHOUT_SLOT

        assert binding.status == KEYS_WITHOUT_SLOT
        assert binding.tier == "episodic"
        assert not binding.publishable


class TestRestoreSweepRemovesStaleTierRootGraph:
    """A target tree carrying a stale tier-root ``graph.json`` is swept clean
    by the restore's clean-slate sweep, so the tier is left in the uniform
    shape and the next capture reports no torn slot."""

    def test_restore_sweep_removes_a_stale_tier_root_graph(self, tmp_path: Path) -> None:
        # Source bundle carries a real, current-shape simulate slot for episodic.
        src = tmp_path / "src"
        episodic_dir = src / "adapters" / "episodic"
        _write_real_simulate_slot(episodic_dir, "graph1")

        bundle_slot = write_bundle(
            config_path=tmp_path / "no-such-config.yaml",
            adapter_dirs={"episodic": episodic_dir},
            backups_root=src / "backups",
            backups_cfg=None,
            meta_fields={"tier": "manual", "label": "graph-capture"},
        )

        # Target tree already carries a stale tier-root graph.json under
        # episodic/, outside any slot.
        target = tmp_path / "target"
        target_episodic = target / "adapters" / "episodic"
        target_episodic.mkdir(parents=True)
        (target_episodic / "graph.json").write_text('{"directed": true}')

        restore_bundle(
            bundle_slot_dir=bundle_slot,
            data_dir=target,
            config_path=target / "server.yaml",
        )

        assert not (target_episodic / "graph.json").exists(), (
            "the stale tier-root graph.json must be swept by the restore"
        )

        binding = verify_tier_binding("episodic", target_episodic)
        assert binding.status == VERIFIED

        # The next capture from the now-clean target reports no torn slot.
        next_bundle_slot = write_bundle(
            config_path=tmp_path / "no-such-config.yaml",
            adapter_dirs={"episodic": target_episodic},
            backups_root=target / "backups",
            backups_cfg=None,
            meta_fields={"tier": "manual", "label": "post-restore-capture"},
        )
        next_manifest = read_bundle_manifest(next_bundle_slot)
        assert next_manifest.adapters["episodic"]["weightless_cause"] is None


class TestRestoreHydratesIdenticalStoreState:
    """A bundle round trip -- write -> publish, then restore -- must not just
    reproduce the right FILES; the tree it lands must boot-hydrate to the
    identical in-RAM state a fresh boot of the source tree would produce.
    Drives both trees through ``_build_store_contents`` (the single
    canonical boot-store builder,
    :func:`paramem.server.app._build_store_contents`) and compares content
    projections, never slot dir names or paths (restore mints fresh
    timestamps by design).

    Two main tiers are seeded: ``episodic`` carries one active key plus a
    SECOND key withheld via the real :meth:`KeyRegistry.stale` API (so the
    round trip exercises the stale set, and the withheld key's bookkeeping
    row surviving alongside it) and ``semantic`` carries one active key.
    """

    _EPISODIC_ACTIVE = {
        "graph1": {"subject": "alice", "predicate": "lives_in", "object": "berlin"},
    }
    _EPISODIC_STALE = {
        "graph2": {"subject": "alice", "predicate": "works_at", "object": "acme corp"},
    }
    _SEMANTIC_ACTIVE = {
        "graph3": {"subject": "bob", "predicate": "likes", "object": "coffee"},
    }

    def _seed_tree(self, adapters_root: Path) -> None:
        _write_real_simulate_slot(
            adapters_root / "episodic",
            list(self._EPISODIC_ACTIVE) + list(self._EPISODIC_STALE),
            stale=list(self._EPISODIC_STALE),
            entries={**self._EPISODIC_ACTIVE, **self._EPISODIC_STALE},
        )
        _write_real_simulate_slot(
            adapters_root / "semantic",
            list(self._SEMANTIC_ACTIVE),
            entries=self._SEMANTIC_ACTIVE,
        )

    def _hydrate(self, adapters_root: Path) -> dict:
        """Boot-hydrate *adapters_root* through ``_build_store_contents``
        (simulate venue, ``preload_cache=True``, ``model=None`` -- the
        MagicMock-config pattern from
        ``tests/test_server.py::TestBuildStoreContents._make_config``) and
        project each main tier's content-only state."""
        from paramem.server.app import _build_store_contents

        cfg = MagicMock()
        cfg.adapter_dir = adapters_root
        cfg.consolidation.mode = "simulate"
        cfg.consolidation.recall_probe_batch_size = 8
        cfg.inference.preload_cache = True
        cfg.paths.data = adapters_root

        new_entries, new_registry, new_bookkeeping, stats = _build_store_contents(
            cfg, model=None, tokenizer=None
        )
        assert stats["preload_complete"] is True, (
            "the simulate venue never defers -- a False here means the fill "
            "silently short-circuited rather than actually probing disk"
        )

        projection = {}
        for tier in ("episodic", "semantic"):
            registry = new_registry[tier]
            known = sorted(registry.list_known())
            projection[tier] = {
                "active_keys": sorted(registry.list_active()),
                "stale_keys": registry.list_stale(),
                "simhash": {key: registry.simhash_for(key) for key in registry.list_active()},
                "entries": new_entries.get(tier, {}),
                "bookkeeping": {key: new_bookkeeping[key] for key in known},
                "tier_cycle": json.loads((adapters_root / tier / "key_metadata.json").read_text())[
                    "tier_cycle"
                ],
            }
        return projection

    def test_restore_hydrates_identical_in_ram_state_across_two_tiers(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        self._seed_tree(src / "adapters")

        source_projection = self._hydrate(src / "adapters")

        # The stale key's post-write state, before any restore is even
        # attempted: withheld from active/simhash, but its bookkeeping row
        # and its membership in the tier's known-key set survive.
        assert source_projection["episodic"]["active_keys"] == ["graph1"]
        assert source_projection["episodic"]["stale_keys"] == ["graph2"]
        assert "graph2" not in source_projection["episodic"]["simhash"]
        assert "graph2" not in source_projection["episodic"]["entries"]
        assert "graph2" in source_projection["episodic"]["bookkeeping"]

        bundle_slot = write_bundle(
            config_path=tmp_path / "no-such-config.yaml",
            adapter_dirs={
                "episodic": src / "adapters" / "episodic",
                "semantic": src / "adapters" / "semantic",
            },
            backups_root=src / "backups",
            backups_cfg=None,
            meta_fields={"tier": "manual", "label": "two-tier-round-trip"},
        )

        target = tmp_path / "target"
        result = restore_bundle(
            bundle_slot_dir=bundle_slot,
            data_dir=target,
            config_path=target / "server.yaml",
        )

        assert set(result.restored_adapters) == {"episodic", "semantic"}
        assert result.weightless_adapters == {}

        target_projection = self._hydrate(target / "adapters")

        # The stale key's post-restore state matches the source exactly --
        # still withheld, still fingerprintless, its bookkeeping row still
        # carried across the round trip.
        assert target_projection["episodic"]["stale_keys"] == ["graph2"]
        assert "graph2" not in target_projection["episodic"]["simhash"]
        assert "graph2" in target_projection["episodic"]["bookkeeping"]

        assert target_projection == source_projection
