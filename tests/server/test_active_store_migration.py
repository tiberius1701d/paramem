"""Unit tests for paramem.server.active_store_migration's standalone helpers.

``_delete_orphaned_simulate_slots``'s logic (venue-blind manifest-kind
classification, keep-slot exclusion, dot-dir exclusion) is directly
unit-testable without a live model, GPU, or ConsolidationLoop -- unlike the
migration functions that call it, which require the full fold/training
surface and are exercised end-to-end via the scoped integration suites
instead (``tests/server/test_migration_confirm.py``,
``tests/test_consolidation.py``).

No GPU required -- every slot here is written with stub payload bytes through
the real :func:`~paramem.adapters.slot.write_slot` envelope.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.adapters.manifest import tier_registry_sha256
from paramem.adapters.slot import write_slot
from paramem.server.active_store_migration import (
    MigrationState,
    _delete_orphaned_simulate_slots,
    _migrate_tier_simulate_to_train,
    _TierSkipped,
    clear_state,
    detect_mode_switch,
    load_state,
    save_state,
    state_path,
)
from paramem.training.key_registry import KeyRegistry
from tests._fold_fixtures import _make_loop, _wire_fakes, _write_graph
from tests._manifest_fixtures import make_train_manifest, write_canonical_registry, write_slot_files


def _write_simulate_slot(
    tier_root: Path, *, name: str = "episodic", registry_sha256: str = ""
) -> Path:
    """Write a real simulate-payload slot under *tier_root*.

    *registry_sha256* defaults to ``""`` -- unbound to any registry, since
    most callers of this helper (the orphan-cleanup tests below) classify by
    the slot's OWN manifest kind, not by binding. Passing the tier's live
    registry digest (:func:`~paramem.adapters.manifest.tier_registry_sha256`)
    produces a slot that :func:`~paramem.adapters.manifest.find_live_slot`
    actually binds to -- what ``detect_mode_switch`` requires.
    """
    from paramem.adapters.manifest import graph_payload_manifest

    manifest = graph_payload_manifest(
        name=name, key_count=0, registry_sha256=registry_sha256, window_stamp=""
    )
    return write_slot(
        tier_root,
        manifest=manifest,
        write_payload=lambda pending: (pending / "graph.json").write_bytes(b"{}"),
    )


def _write_train_slot(
    tier_root: Path, *, name: str = "episodic", registry_sha256: str = ""
) -> Path:
    manifest = make_train_manifest(name=name, registry_sha256=registry_sha256, key_count=0)
    return write_slot(
        tier_root, manifest=manifest, write_payload=lambda pending: write_slot_files(pending)
    )


def _cfg(adapter_dir: Path, *, mode: str) -> MagicMock:
    """Minimal ``ServerConfig`` stand-in for ``detect_mode_switch``, which
    reads only ``config.adapter_dir`` and ``config.consolidation.mode``."""
    cfg = MagicMock()
    cfg.adapter_dir = adapter_dir
    cfg.consolidation.mode = mode
    return cfg


class TestDeleteOrphanedSimulateSlots:
    def test_deletes_simulate_slots_other_than_keep(self, tmp_path: Path) -> None:
        """A stale simulate slot left behind by an interrupted migration is
        removed; the freshly-committed train slot (``keep``) is untouched."""
        tier_root = tmp_path / "episodic"
        orphan = _write_simulate_slot(tier_root)
        kept = _write_train_slot(tier_root)

        deleted = _delete_orphaned_simulate_slots(tier_root, keep=kept)

        assert deleted == 1
        assert not orphan.exists()
        assert kept.exists()

    def test_never_deletes_the_keep_slot_even_if_simulate(self, tmp_path: Path) -> None:
        """keep is never removed regardless of its own payload kind -- the
        exclusion is by identity, not by kind."""
        tier_root = tmp_path / "episodic"
        kept = _write_simulate_slot(tier_root)

        deleted = _delete_orphaned_simulate_slots(tier_root, keep=kept)

        assert deleted == 0
        assert kept.exists()

    def test_skips_dot_prefixed_directories(self, tmp_path: Path) -> None:
        """A .pending staging directory is never treated as an orphan slot,
        matching iter_slot_candidates' dot-dir exclusion rule."""
        tier_root = tmp_path / "episodic"
        pending = tier_root / ".pending" / "20260101-000000"
        pending.mkdir(parents=True)
        (pending / "meta.json").write_bytes(b"{}")
        (pending / "graph.json").write_bytes(b"{}")
        kept = _write_train_slot(tier_root)

        deleted = _delete_orphaned_simulate_slots(tier_root, keep=kept)

        assert deleted == 0
        assert pending.exists()

    def test_skips_unreadable_manifest_best_effort(self, tmp_path: Path) -> None:
        """A subdirectory whose meta.json is missing or unparseable is left
        alone -- this is a best-effort orphan sweep, not
        cleanup_partial_slots' scratch-removal contract."""
        tier_root = tmp_path / "episodic"
        no_manifest = tier_root / "20260101-000000"
        no_manifest.mkdir(parents=True)
        (no_manifest / "graph.json").write_bytes(b"{}")
        kept = _write_train_slot(tier_root)

        deleted = _delete_orphaned_simulate_slots(tier_root, keep=kept)

        assert deleted == 0
        assert no_manifest.exists()

    def test_skips_interim_dir_family(self, tmp_path: Path) -> None:
        """An interim_<stamp>/ container under the main tier's slot root
        survives untouched even when it carries a simulate-payload slot --
        it is a sibling tier owned by find_live_slot + the boot-time
        keyless-tier sweep, never this main-tier orphan cleanup (mirrors
        integrity.py's INTERIM_DIR_PREFIX skip in its own partial-slot
        sweep)."""
        tier_root = tmp_path / "episodic"
        interim_root = tier_root / "interim_20260101T0000"
        interim_simulate_slot = _write_simulate_slot(
            interim_root, name="episodic_interim_20260101T0000"
        )
        kept = _write_train_slot(tier_root)

        deleted = _delete_orphaned_simulate_slots(tier_root, keep=kept)

        assert deleted == 0
        assert interim_simulate_slot.exists()
        assert kept.exists()

    def test_absent_slot_root_returns_zero(self, tmp_path: Path) -> None:
        """A tier root that does not exist on disk yields no deletions,
        never an exception."""
        tier_root = tmp_path / "episodic"
        keep = tier_root / "20260101-000000"

        deleted = _delete_orphaned_simulate_slots(tier_root, keep=keep)

        assert deleted == 0


class TestMigrateTierSimulateToTrainEntryFilter:
    """Unit coverage for ``_migrate_tier_simulate_to_train``'s
    registry-authoritative entry filter (``active_keys =
    set(loop.store.registry(name).list_active())``) -- a
    withheld key's graph.json edge must never reach the training funnel.
    No GPU: ``create_adapter`` is faked the same way
    ``tests/_fold_fixtures.py`` documents (a real ``get_peft_model()`` cold
    wrap needs a real ``torch.nn.Module``, which the fake driver model is
    not), and the training mock is forced to return ``(None, None)`` so the
    function raises ``_TierSkipped`` immediately after the funnel call --
    before ``staged_weights``/``promote_staging_adapter``/``commit_tier_slot``,
    none of which this test needs to exercise."""

    def test_a_withheld_keys_graph_entry_is_not_migrated_back_into_a_trained_tier(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from paramem.models import loader as loader_mod

        loop = _make_loop(tmp_path)
        _wire_fakes(loop, monkeypatch, train_side_effect=lambda entries, **kwargs: (None, None))

        def _fake_create_adapter(model, adapter_config, adapter_name="default"):
            # Mirrors real create_adapter's own add_adapter call, on the fake
            # driver model's (tests/_fold_fixtures.py::_make_fake_driver_model)
            # public seeding contract -- instead of taking the real cold
            # get_peft_model() path, which needs a real torch.nn.Module.
            model.add_adapter(adapter_name, adapter_config)
            model.set_adapter(adapter_name)
            return model

        monkeypatch.setattr(loader_mod, "create_adapter", _fake_create_adapter)

        tier_root = loop.output_dir / "episodic"

        # graph1 active, graph2 withheld -- both on-disk (the registry
        # authority) and mirrored on loop.store (the filter's own read
        # site), exactly as a production caller hydrates before migrate().
        disk_registry = KeyRegistry()
        disk_registry.add("graph1")
        disk_registry.add("graph2")
        disk_registry.stale("graph2")
        disk_registry.save(tier_root / "indexed_key_registry.json")

        store_registry = KeyRegistry()
        store_registry.add("graph1")
        store_registry.add("graph2")
        store_registry.stale("graph2")
        loop.store.load_registry("episodic", store_registry)

        # The bound simulate slot's graph.json still carries BOTH edges --
        # the withheld key's content lingers there (an operator erase
        # stale-marks the registry, it never touches graph.json).
        _write_graph(
            tier_root,
            [
                {
                    "key": "graph1",
                    "subject": "alice",
                    "predicate": "lives_in",
                    "object": "berlin",
                    "speaker_id": "speaker0",
                },
                {
                    "key": "graph2",
                    "subject": "bob",
                    "predicate": "lives_in",
                    "object": "paris",
                    "speaker_id": "speaker0",
                },
            ],
        )

        cfg = MagicMock()
        cfg.adapter_dir = loop.output_dir

        with pytest.raises(_TierSkipped):
            _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        loop._train_tier_adapter.assert_called_once()
        migrated_entries = loop._train_tier_adapter.call_args.args[0]
        migrated_keys = {e["key"] for e in migrated_entries}
        assert migrated_keys == {"graph1"}

        # The withheld key's marker survives untouched -- migration mutates
        # nothing about it.
        assert loop.store.registry("episodic").knows("graph2") is True
        assert "graph2" not in loop.store.registry("episodic")

    def test_a_marker_only_registry_skips_its_migration_without_calling_itself_empty(
        self, tmp_path: Path
    ) -> None:
        """No ACTIVE key anywhere in the tier, but the registry file is
        present (a marker-only registry, e.g. every key withheld) --
        _TierSkipped names it "no active key", not "empty" (the "empty"
        wording is reserved for the earlier, unrelated empty-graph.json
        skip a few lines up in the same function)."""
        loop = _make_loop(tmp_path)
        tier_root = loop.output_dir / "episodic"

        disk_registry = KeyRegistry()
        disk_registry.add("graph1")
        disk_registry.stale("graph1")
        disk_registry.save(tier_root / "indexed_key_registry.json")

        store_registry = KeyRegistry()
        store_registry.add("graph1")
        store_registry.stale("graph1")
        loop.store.load_registry("episodic", store_registry)

        # graph.json still carries the withheld key's leftover edge -- a
        # non-empty graph is required to reach the "no active key" branch
        # rather than the earlier "empty graph.json" skip.
        _write_graph(
            tier_root,
            [
                {
                    "key": "graph1",
                    "subject": "alice",
                    "predicate": "lives_in",
                    "object": "berlin",
                    "speaker_id": "speaker0",
                }
            ],
        )

        cfg = MagicMock()
        cfg.adapter_dir = loop.output_dir

        with pytest.raises(_TierSkipped) as exc_info:
            _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        message = str(exc_info.value)
        assert "no active key" in message
        assert "empty" not in message.lower()


class TestDetectModeSwitch:
    """Behavioral pins for ``detect_mode_switch``'s manifest-bound
    classification (paramem/server/active_store_migration.py:189-254) --
    mode-switch detection reads the slot's bound manifest, never a
    filename at the tier root."""

    def test_simulate_bound_tier_with_target_train_arms_simulate_to_train(
        self, tmp_path: Path
    ) -> None:
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        write_canonical_registry(tier_root, ["k1"])
        _write_simulate_slot(tier_root, registry_sha256=tier_registry_sha256(tier_root))

        result = detect_mode_switch(_cfg(adapter_dir, mode="train"))

        assert result is not None
        assert result.direction == "simulate_to_train"
        assert result.source_mode == "simulate"
        assert result.target_mode == "train"

    def test_train_bound_tier_with_target_simulate_arms_train_to_simulate(
        self, tmp_path: Path
    ) -> None:
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "semantic"
        write_canonical_registry(tier_root, ["k1"])
        _write_train_slot(
            tier_root, name="semantic", registry_sha256=tier_registry_sha256(tier_root)
        )

        result = detect_mode_switch(_cfg(adapter_dir, mode="simulate"))

        assert result is not None
        assert result.direction == "train_to_simulate"
        assert result.source_mode == "train"
        assert result.target_mode == "simulate"

    def test_tier_already_bound_to_the_configured_mode_arms_nothing(self, tmp_path: Path) -> None:
        """Both venues: a tier already carrying the configured mode's own
        payload kind is not a mismatch -- detect_mode_switch arms nothing."""
        train_dir = tmp_path / "train_adapters"
        train_root = train_dir / "procedural"
        write_canonical_registry(train_root, ["k1"])
        _write_train_slot(
            train_root, name="procedural", registry_sha256=tier_registry_sha256(train_root)
        )
        assert detect_mode_switch(_cfg(train_dir, mode="train")) is None

        simulate_dir = tmp_path / "simulate_adapters"
        simulate_root = simulate_dir / "episodic"
        write_canonical_registry(simulate_root, ["k1"])
        _write_simulate_slot(simulate_root, registry_sha256=tier_registry_sha256(simulate_root))
        assert detect_mode_switch(_cfg(simulate_dir, mode="simulate")) is None

    def test_existing_state_file_takes_precedence_over_fresh_detection(
        self, tmp_path: Path
    ) -> None:
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        write_canonical_registry(tier_root, ["k1"])
        _write_simulate_slot(tier_root, registry_sha256=tier_registry_sha256(tier_root))

        prior = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        prior.completed_tiers = ["procedural"]
        save_state(adapter_dir, prior)

        # Fresh detection against this tree (simulate-bound + mode=train)
        # would arm simulate_to_train -- the state file wins instead.
        result = detect_mode_switch(_cfg(adapter_dir, mode="train"))

        assert result is not None
        assert result.direction == "train_to_simulate"
        assert result.completed_tiers == ["procedural"]

    def test_unsupported_mode_returns_none(self, tmp_path: Path) -> None:
        result = detect_mode_switch(_cfg(tmp_path / "adapters", mode="bogus"))
        assert result is None

    def test_unreadable_bound_manifest_is_treated_as_unbound_not_raised(
        self, tmp_path: Path
    ) -> None:
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        write_canonical_registry(tier_root, ["k1"])
        bound = _write_simulate_slot(tier_root, registry_sha256=tier_registry_sha256(tier_root))
        (bound / "meta.json").write_bytes(b"not valid json")

        result = detect_mode_switch(_cfg(adapter_dir, mode="train"))

        assert result is None

    def test_interim_family_never_satisfies_a_main_tier(self, tmp_path: Path) -> None:
        """A tier root holding ONLY an interim family (its manifests one
        level deeper than a main-tier slot) plus a canonical registry arms
        nothing -- iter_slot_candidates only yields direct children carrying
        their OWN meta.json, and an interim_<stamp>/ family root has none
        (its manifests live at interim_<stamp>/<ts>/meta.json)."""
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        interim_root = tier_root / "interim_20260101T000000"
        _write_simulate_slot(interim_root, name="episodic_interim_20260101T000000")
        write_canonical_registry(tier_root, ["k1"])

        result = detect_mode_switch(_cfg(adapter_dir, mode="train"))

        assert result is None


class TestStateFileIO:
    """Round-trip coverage for the state-file primitives (save_state /
    load_state / clear_state)."""

    def test_save_then_load_round_trips_equal(self, tmp_path: Path) -> None:
        original = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        original.completed_tiers = ["episodic"]
        original.failed_tiers = {"semantic": "boom"}

        save_state(tmp_path, original)
        loaded = load_state(tmp_path)

        assert loaded == original

    def test_clear_state_removes_the_file(self, tmp_path: Path) -> None:
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        save_state(tmp_path, state)
        assert state_path(tmp_path).exists()

        clear_state(tmp_path)

        assert not state_path(tmp_path).exists()
        assert load_state(tmp_path) is None

    def test_load_state_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert load_state(tmp_path) is None
